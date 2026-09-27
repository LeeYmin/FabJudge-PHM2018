"""Build the reviewable 06e notebook; execution reuses verified stage artifacts."""

from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parents[1]
cells = [
    nbf.v4.new_markdown_cell("""# 06e · 06d 버그 정정과 약점 진단

1부는 06d의 dev P2 RF 입력만 OOF로 고쳐 재실행한다. 2부는 JEV를 호출하지 않고 train·validation 시퀀스의 causal endpoint를 진단한다. `data/raw/`와 기존 06a~06d 산출물은 수정하지 않는다. 예외로 06d 결과 폴더에 정오표 하나만 추가한다."""),
    nbf.v4.new_markdown_cell("""## 1. 06d 정정의 고정 가정

- P2 질문·센서·v2 직렬화와 기존 frozen_config의 후보, 임계값, 판정식은 그대로 사용한다.
- dev의 `train_oof`에서 요청 state의 `rf_probability`는 `rf_probability_oof`에서 읽는다. 호출 직전 v2 직렬화 값을 검증한다.
- test의 23개 P2 요청은 기존 캐시 body SHA-256과 저장된 out-of-sample RF를 검증하고 재호출하지 않는다.
- dev P2는 최대 145회 재호출한다. 첫 응답 스키마 오류, 상한 초과, frozen 해시 불일치면 중단한다. S2/J2만 다시 적합한다."""),
    nbf.v4.new_code_cell("""import sys, json
from pathlib import Path
import pandas as pd
from IPython.display import display, Markdown
ROOT = Path.cwd() if (Path.cwd() / 'src/fabjudge').exists() else Path.cwd().parent
ROOT = ROOT.resolve()
sys.path.insert(0, str(ROOT / 'src'))
OUT = ROOT / 'artifacts/06e_fix_and_weakness_diagnosis'
OUT.mkdir(parents=True, exist_ok=True)
from fabjudge.fix_06e import run as run_fix
from fabjudge.weakness_06e import preregister, run as run_diagnosis
fix = json.loads((OUT / 'fix_summary.json').read_text(encoding='utf-8')) if (OUT / 'fix_summary.json').exists() else run_fix(ROOT, OUT)
print('test RF/cache verified:', fix['test_state_verified'], 'new P2 calls:', fix['new_jev_calls'])
print('06d config SHA-256:', fix['source_06d_config_sha256'])
print('corrected frozen SHA-256:', fix['frozen_config_sha256'])"""),
    nbf.v4.new_code_cell("""display(pd.read_csv(OUT / 'dev_metric_comparison.csv'))
display(pd.read_csv(OUT / 'metric_comparison_06d_fix.csv')[['candidate','version','constraint_met','removed_fp','lost_tp','auc']])
print('S2 combination effect, original -> corrected:', fix['s2_combination_effect_original'], '->', fix['s2_combination_effect_corrected'])"""),
    nbf.v4.new_markdown_cell("""## 2. 진단의 사전 등록: 계산 전 정의

04b의 300행, 15원본행 간격 LSTM 입력은 통상 약 17,940초에 해당한다. 전체 15행 간격 endpoint와 마지막 행을 사용한다. train RF는 시퀀스별 OOF, validation RF는 저장된 out-of-sample 모델을 사용한다. train LSTM은 in-sample이므로 모든 진단 표를 낙관적 추정으로 표시한다. fault3 test 8개 시퀀스는 진단에 쓰지 않는다.

TTF 구간은 0–5k, 5–10k, 10–20k, 20–50k, >50k초다. 행 단위와 시퀀스 단위로 LSTM signed error의 중앙값·IQR, 조기/지연 비율, 5k 경보율, 가능한 경우 RF AUC를 계산한다. PHM 비대칭 비용은 적용 가능한 확정 식이 없으면 미적용으로 둔다."""),
    nbf.v4.new_markdown_cell("""## 3. 긴 창 feature와 과제 정의: 계산 전 고정

FLOWCOOLPRESSURE, FLOWCOOLFLOWRATE, 04b pressure–flow residual의 04b 입력 스케일을 사용한다. train residual의 압력–유량 관계 모델은 해당 시퀀스를 뺀 RF OOF fold에서, validation은 저장된 train 모델에서 얻는다. 각 센서에서 과거 현재 1,000초 평균의 baseline z, 과거 현재 10,000초 평균의 baseline z, 10,000초 실제 시간축 기울기×1,000초/baseline std, 10,000초 std/baseline std를 만든다. baseline은 시퀀스 시작의 처음 20,000초다. 경과 시간이 30,000초보다 짧거나 baseline std가 0이면 `insufficient_history`로 두고 대치하지 않는다.

T1은 fault3 LSTM 5k 경보의 실제 5k TP/FP다. T2는 fault3 LSTM 예측 >5k 행에서 실제 20k 이내 고장 여부다. T3는 T2 과제를 M01 학습→M02 평가 및 반대 방향으로 본다. T4는 다른 fault artifact가 있을 때 같은 T2 정의로 본다. 원래 라벨은 변경하지 않는다."""),
    nbf.v4.new_markdown_cell("""## 4. 사전 검사와 06f 선택 규칙: 계산 전 고정

입력 E는 LSTM 예측초와 RF OOF, L은 긴 창 12개, E+L은 결합이다. 표준화 로지스틱(C=1)과 작은 RF(100 trees, depth 4, seed 42)를 모두 사용한다. 시퀀스 GroupKFold(5) OOF로 AUC와 시퀀스 bootstrap 1,000회 95% CI, 결합−E의 paired CI를 구한다. T3는 정의상 반대 장비 holdout을 쓴다. 어느 쪽 클래스든 평가 시퀀스가 5개 미만이면 해당 과제만 제외한다. T2는 recall 0.8 기준 오경보율도 낸다.

1순위는 E+L−E AUC CI 하한 >0인 과제이며 차이가 큰 순이다. 없으면 L AUC CI 하한 >0.6이고 E AUC <0.75인 과제다. 둘 다 없으면 추가 신호가 확인되지 않은 것으로 기록하고 06f를 설계하지 않는다. 선택되면 E+L 로지스틱과 RF를 06f 기준선으로 고정한다. 이 정의와 규칙은 아래 계산 셀 전에 `prereg_06e.json`에 해시와 함께 저장한다. 초기 v1의 train residual 관계 모델이 전체 train으로 적합된 점을 발견해 v1을 보존하고, 시퀀스 OOF residual을 명시한 v2를 계산 전에 다시 고정했다."""),
    nbf.v4.new_code_cell("""prereg_sha = preregister(OUT)
print('prereg SHA-256:', prereg_sha)
diagnosis = ({'prereg_sha256': prereg_sha, 'selection': json.loads((OUT / 'task_selection.json').read_text(encoding='utf-8'))}
             if (OUT / 'task_selection.json').exists() else run_diagnosis(ROOT, OUT))
assert preregister(OUT) == prereg_sha
check = json.loads((OUT / 'lstm_input_check.json').read_text(encoding='utf-8'))
print('LSTM input:', check['lstm_max_input_rows'], 'rows; nominal span:', check['lstm_nominal_span_seconds'], 'seconds')
print('sample interval quantiles:', check['sample_interval_seconds_quantiles'])
print('other fault artifacts:', check['available_fault_artifacts'])
print('training seconds:', check['lstm_training_seconds_by_fault'])
print('long-window overlap:', check['window_overlap_note'])"""),
    nbf.v4.new_code_cell("""long_rows = pd.read_parquet(OUT / 'long_window_features.parquet')
print('causal dev endpoints:', len(long_rows), 'insufficient history:', int(long_rows.insufficient_history.sum()))
display(long_rows.groupby('fault', as_index=False).agg(rows=('sequence_id','size'), sequences=('sequence_id','nunique'), insufficient=('insufficient_history','sum')))
weak = pd.read_csv(OUT / 'weakness_map.csv')
display(weak.loc[(weak.fault == 'fault3') & (weak.unit == 'row') & (weak.dimension == 'ttf_band')])
display(weak.loc[(weak.unit == 'row') & weak.dimension.isin(['machine','source'])])"""),
    nbf.v4.new_code_cell("""screen = pd.read_csv(OUT / 'prescreen_results.csv')
excluded = pd.read_csv(OUT / 'excluded_tasks.csv')
selection = json.loads((OUT / 'task_selection.json').read_text(encoding='utf-8'))
print('eligible prescreens:'); display(screen)
print('excluded tasks:'); display(excluded)
print('06f selection:'); print(json.dumps(selection, ensure_ascii=False, indent=2))"""),
    nbf.v4.new_markdown_cell("""## 5. 결론

아래 결론은 저장된 수치에서 생성한다. 긴 창 과제가 이력 부족으로 제외되면 성능 부재로 해석하지 않는다. 모든 train LSTM 수치는 in-sample이고 fault3 validation은 4개 시퀀스이며, 2부에서 test는 사용하지 않았다."""),
    nbf.v4.new_code_cell("""from runpy import run_path
run_path(str(ROOT / 'scripts/finalize_06e.py'))
display(Markdown((OUT / 'conclusion.md').read_text(encoding='utf-8')))"""),
]
notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
nbf.write(notebook, root / "notebooks/06e_fix_and_weakness_diagnosis.ipynb")
