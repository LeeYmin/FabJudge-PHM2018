"""Create the staged, executable 06d experiment notebook."""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()


def md(s):
    nb.cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    nb.cells.append(nbf.v4.new_code_cell(s.strip()))


md("""
# 06d · LSTM–JEV OOF edge

**사전 질문:** 누수 없는 RF 확률을 쓸 때 JEV P2/P3가 RF에 판별력을 더하는가? RF 입력이 없는 P3가 독립 신호인가? JEV는 경보의 진실 판정기가 아니라 검토 우선순위 신호다.

실행 순서는 RF 재현·OOF → 질문과 평가 기준 사전 등록 → dev JEV → dev 설정 고정 → test 한 번 평가다. `data/raw/`는 건드리지 않는다. 시퀀스 NPZ에서는 현재·과거의 `x`, `time`, `raw_row_index`만 읽고 `samples_to_fault`는 열지 않는다. 라벨은 별도 event metadata에서 얻는다.
""")

md("""
## 0. 캐시 직렬화 확인 및 RF 재현

**가정:** 04b RF는 현재 측정의 인코딩 21열과 압력–유량 residual 1열을 쓴다. LSTM 출력은 RF 입력이 아니다. 저장된 fault/normal 샘플에서 원본 train 시퀀스만 선택하고 boundary는 학습에서 제외한다. residual regressor는 normal만 학습한다. RF 분류기 64 trees, depth 16, leaf 5; residual 64 trees, depth 10, leaf 20; seed 42. fault/normal 역빈도 가중치를 재현한다. 저장된 test 확률 첫 20행 오차가 0.05보다 크면 중단한다. 키 v2는 12자리 유효숫자 state를 HTTP body와 SHA-256 양쪽에 동일하게 쓴다.
""")

code(r"""
import sys, json, hashlib, difflib
from pathlib import Path
import joblib, numpy as np, pandas as pd
from IPython.display import display
PROJECT = Path.cwd() if (Path.cwd() / 'src/fabjudge/paths.py').exists() else Path.cwd().parent
sys.path.insert(0, str(PROJECT / 'src'))
from fabjudge.huang2018_models import train_evaluate_rf, _rf_matrix_encoded
from fabjudge.jev_oof_edge import (stage1, stage2_calls, stage4_freeze, stage5_test,
    make_questions, make_prereg, config_hash, write_json, log)
from fabjudge.jev_gate import canonical, request_spec
ROOT = PROJECT.resolve(); ART = ROOT / 'artifacts/huang2018'
OUT = ROOT / 'artifacts/06d_lstm_jev_oof_edge'; OUT.mkdir(parents=True, exist_ok=True)
SPLIT = json.loads((ART / 'split_metadata.json').read_text(encoding='utf-8'))['splits']['fault3']
META = pd.read_csv(ART / 'sequence_metadata.csv')
INPUT_COLUMNS = json.loads((ART / 'reproduction_config.json').read_text(encoding='utf-8'))['model_input_columns']
assert len(INPUT_COLUMNS) == 21
PREVIOUS = json.loads((ROOT / 'artifacts/06c_lstm_jev_semantic_gate/frozen_config.json').read_text(encoding='utf-8'))
assert PREVIOUS['selected_features'] == ['ROTATIONSPEED','ROTATIONSPEED__rms']
print('RF inputs: 21 current-row columns + pressure-flow residual; no LSTM output')
print('RF seed=42, classifier=(64, depth 16, leaf 5), residual=(64, depth 10, leaf 20)')

rf, rf_metrics = train_evaluate_rf('fault3', 3, ART, OUT / 'models', SPLIT, seed=42)
saved = pd.read_parquet(ART / 'fault3_test_predictions.parquet').head(20)
lookup = META.set_index('sequence_id'); reproduced = []
for row in saved.itertuples():
    with np.load(ROOT / str(lookup.loc[row.sequence_id,'sequence_path'])) as data:
        positions = np.flatnonzero(data['raw_row_index'] == row.raw_row_index)
        assert len(positions) == 1
        x = data['x'][positions]
    reproduced.append(rf['classifier'].predict_proba(_rf_matrix_encoded(x, rf['residual_model']))[0,1])
error = np.abs(np.array(reproduced) - saved.rf_probability.to_numpy())
parity = {'first_20_max_abs_error':float(error.max()),
          'first_20_mean_abs_error':float(error.mean()),
          'within_0_05':bool(error.max() <= .05),
          'exact_reproduction':bool(error.max() <= 1e-10),
          'seed':42, 'train_counts':rf_metrics['stored_train_counts'],
          'uses_lstm_output':False,
          'rf_inputs':'21 encoded current-row inputs + FLOWCOOLFLOWRATE minus fitted pressure-flow residual'}
write_json(OUT / 'rf_parity.json', parity)
print('First 20 saved test RF probabilities, maximum absolute error:', error.max())
if error.max() > .05: raise RuntimeError('Stop: RF parity error > 0.05')
""")

md("""
## 1. 원본 시퀀스 단위 RF OOF 및 calibration

**가정:** train의 36개 원본 시퀀스를 GroupKFold(6)로 나누고, 빠진 시퀀스의 LSTM 경보 행에만 RF OOF 확률을 붙인다. validation의 RF와 LSTM은 원래 out-of-sample이다. 06c에서 저장된 train LSTM 예측·센서 상태를 재사용한다. train LSTM 출력과 경보 집합은 여전히 in-sample이라는 한계가 있다. train OOF와 validation 경보를 합치되 출처를 보존한다. TP·FP 중 하나가 20건 미만이면 중단한다.
""")

code(r"""
dev, dev_alarm, calibration, fold_log = stage1(ROOT, OUT, SPLIT, META, INPUT_COLUMNS)
print('Fold training:', fold_log)
print('Dev alarm counts:', dev_alarm.groupby(['source','actual_fail_5000']).size().to_dict())
display(calibration)
""")

md("""
## 2. 질문과 test 정합성

**가정:** 06c의 센서 두 개·경계값·P2 질문 원문을 고정한다. P2의 RF 값만 OOF로 교체한다. P3는 P2에서 RF 설명·상태 변수·criteria의 RF 구절만 제거한다. 기존 06c test 80행의 LSTM 경보를 test 모집단으로 쓰며 정합성 TP 15, FP 8, FN 1, TN 56을 먼저 확인한다. 06c의 JEV test 결과는 후보 선택에 쓰지 않는다.
""")

code(r"""
p2, p3 = make_questions(PREVIOUS)
qdir = OUT / 'questions'; qdir.mkdir(exist_ok=True)
write_json(qdir / 'P2.json', p2); write_json(qdir / 'P3.json', p3)
print('P2 → P3 diff:')
print(''.join(difflib.unified_diff(json.dumps(p2,indent=2).splitlines(True),
                                  json.dumps(p3,indent=2).splitlines(True),
                                  fromfile='P2',tofile='P3')))
test = pd.read_csv(ROOT / 'artifacts/06c_lstm_jev_semantic_gate/test_results.csv',
                   float_precision='round_trip')
assert len(test) == 80 and test.sequence_id.nunique() == 8
truth = test.actual_fail_5000.to_numpy(bool); alarm = test.lstm_alarm.to_numpy(bool)
counts = (int((truth & alarm).sum()),int((~truth & alarm).sum()),
          int((truth & ~alarm).sum()),int((~truth & ~alarm).sum()))
print('06a/06b/06c test parity TP,FP,FN,TN:', counts)
if counts != (15,8,1,56): raise RuntimeError('Stop: test parity failed')
""")

md("""
## 3. JEV 호출 전에 판정 기준 사전 등록

**가정:** 후보 B1/B3/S2/S3/J2/J3를 고정한다. 로지스틱은 표준화, C=1이고 dev 보고치는 원본 시퀀스 GroupKFold OOF만 쓴다. AUC와 차이는 시퀀스 bootstrap 1,000회 95% CI로 평가한다. 점수가 높을수록 근거가 강하다. 사용자와 합의한 순서로 `t_low`는 dev TP recall≥0.98의 최고 임계값, `t_keep`은 ≥0.90의 최고 임계값이다. 점수 < `t_low`는 하향, ≥`t_keep`은 유지, 사이는 검토다. 하향만 FP 제거로 집계한다. test TP 손실 허용은 1건이다. 사전 판정 기준은 config에 기록한다. 예상 호출은 dev 경보×2 + test 46 + P3 반복 15, 비용은 06c cache의 요청당 약 $0.000025를 사용한다.
""")

code(r"""
config, prereg_hash = make_prereg(OUT, PREVIOUS, dev_alarm, p2, p3, 'swapped')
MAX_NEW_CALLS = len(dev_alarm)*2 + 46 + 15
print({'dev_calls':len(dev_alarm)*2,'test_calls':46,'stability_calls':15,
       'MAX_NEW_CALLS':MAX_NEW_CALLS,'estimated_USD':MAX_NEW_CALLS*2.5e-5,
       'prereg_sha256':prereg_hash})
assert config_hash(OUT / 'frozen_config.json') == prereg_hash
""")

md("""
## 4. Dev JEV 실행과 설정 고정

첫 응답 schema 실패, 미해결 dev 행, 호출 상한 도달 시 중단한다. 반복 5행×3회는 API body를 같은 질문으로 보내되 별도 반복 cache 폴더에 저장한다. 후보 AUC, 상관, 3구간 수, TP 손실, FP 제거를 모두 기록한다. 이후 test 전에 모델 계수와 임계값이 든 frozen_config의 SHA-256을 확정한다.
""")

code(r"""
dev, stability, client, calls = stage2_calls(ROOT, OUT, dev, test, p2, p3, MAX_NEW_CALLS)
alarms, dev_candidates, config, frozen_hash = stage4_freeze(OUT, dev, config, prereg_hash)
print('Frozen config SHA-256:', frozen_hash)
display(dev_candidates)
display(stability)
""")

md("""
## 5. 고정된 설정으로 test 한 번 평가

test의 RF 확률은 04b 저장값을 사용한다. RF 재학습이 저장값을 사실상 정확히 재현했으므로 양쪽은 동일하다. frozen_config 해시를 확인하고 23개 경보에 P2/P3를 각각 호출한다. 80행 전체 혼동행렬, 경보 3구간, 시퀀스 bootstrap CI, 시퀀스별 첫 경보를 기록한다. 06c P2 수치는 참고 열이다.
""")

code(r"""
comparison, summary = stage5_test(ROOT, OUT, test, config, frozen_hash, client, calls)
display(comparison)
print('Summary:', summary)
""")

md("""
## 6. 결론

아래 수치는 위에서 고정한 판정식으로 자동 생성한다. CI가 겹치는 FP 제거 후보는 우열을 단정하지 않는다. 이 결과는 fault3의 8개 test 시퀀스와 in-sample LSTM 경보 집합에 한정된다.
""")

code(r"""
train_cal = calibration.loc[(calibration.source=='train_oof') & (calibration.field=='rf_probability_oof')]
old_cal = calibration.loc[(calibration.source=='train_oof') & (calibration.field=='rf_probability')]
val_cal = calibration.loc[calibration.source=='validation']
print('1. RF 중앙값 (TP/FP) train in-sample:', old_cal.set_index('class')['median'].to_dict(),
      '→ train OOF:', train_cal.set_index('class')['median'].to_dict(),
      '; validation:', val_cal.set_index('class')['median'].to_dict())
print('2. 결합 효과 S2/S3:', config['auc_differences_vs_B1'],
      '; 최종 판정:', summary['combination_effect'])
j3 = dev_candidates.loc[(dev_candidates.candidate=='J3') & dev_candidates.source.isna()].iloc[0]
print('3. 독립 신호 J3: dev AUC 95% CI lower=',j3.auc_ci_low,
      ', RF Spearman=',summary['j3_rf_spearman'],', 판정=',summary['independent_signal'])
eligible = comparison.loc[comparison.constraint_met]
if len(eligible):
    top = eligible.loc[eligible.removed_fp.eq(eligible.removed_fp.max())]
    print('4. recall 제약 충족 최대 FP 제거:',top[['candidate','removed_fp','removed_fp_ci_low','removed_fp_ci_high']].to_dict('records'))
    ranges = eligible[['candidate','removed_fp_ci_low','removed_fp_ci_high']].to_dict('records')
    print('   후보 간 95% CI가 겹치면 구분 불가:',ranges)
else: print('4. recall 제약 충족 후보 없음')
print('5. 한계: train LSTM 출력·경보 집합 in-sample; test 8개 시퀀스; fault3 단일 분포.')
if not any(summary['combination_effect'].values()) and not summary['independent_signal']:
    print('판정: fault3에서 JEV 강점 없음. 다음 탐색: RF 불확실 구간과 M01↔M02 분포 이동.')
""")

nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata.language_info = {"name": "python"}
conclusion = root / "artifacts/06d_lstm_jev_oof_edge/conclusion.md"
if conclusion.exists():
    md(conclusion.read_text(encoding="utf-8"))
path = root / "notebooks/06d_lstm_jev_oof_edge.ipynb"
nbf.write(nb, path)
print(path)
