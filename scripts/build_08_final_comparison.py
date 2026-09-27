"""Build notebook 08 without executing any historical training/API notebooks."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md('''# 08 · 최종 비교 — 저장 artifact 검증 및 보고서 산출물

**새 API 호출·모델 학습·재학습·추론 없음.** 기존 artifact만 읽으며 threshold, label,
E, 가설, routing, 검정 순서를 변경하지 않는다. 원본 artifact의 SHA-256을 마지막에 다시 확인한다.
수치 불일치는 `atol=1e-9, rtol=0` assert로 즉시 중단한다. 전체 endpoint가 누락되면 T5 이후를 진행하지 않는다.

**분석 단위:** 07b는 LSTM 경보 중 시퀀스별 percentile 표본 720 endpoint,
전체 분석은 04b causal test endpoint 전부이다. 한 고장 사건에는 여러 endpoint가 속한다.
두 모집단의 recall을 직접 합치지 않는다. fault1+2가 확증 주 모집단이며 fault3은 과거 test 재사용으로 탐색이다.

**통계적 지위:** H1~H3만 사전등록 확증 검정이다. E10/E30, fault별 결과, 대안 routing,
AUC, 위치 진단, 06 회고, 전체 endpoint 무작위 rescue 시나리오는 탐색이다.
‘확증 표본’이라는 표 제목은 그 표의 모든 효과가 가설검정되었다는 뜻이 아니다.

**산출물:** `artifacts/08_final_comparison/tables/*.csv`, `figures/*.png` (300 dpi),
`manifest.csv`. 재사용 가능한 읽기·검증·집계 함수는 `src/fabjudge/final_comparison.py`에 있다.
''')
code('''from pathlib import Path
import sys, platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

ROOT = Path.cwd().resolve()
if not (ROOT / 'src' / 'fabjudge').is_dir():
    ROOT = ROOT.parent
assert (ROOT / 'src' / 'fabjudge').is_dir()
sys.path.insert(0, str(ROOT / 'src'))
from fabjudge.final_comparison import FinalComparison, COLORS, RANDOM_SEED

analysis = FinalComparison(ROOT)
# 모든 figure는 아래 단일 mapping을 재사용한다 (Okabe–Ito 계열).
METHOD_COLORS = COLORS
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_colwidth', 110)
print('Python:', platform.python_version(), '| NumPy:', np.__version__, '| pandas:', pd.__version__)
print('Seed:', RANDOM_SEED, '| Font:', plt.rcParams['font.family'])
display(pd.DataFrame(METHOD_COLORS.items(), columns=['method', 'color']))''')
md('''## 1. Artifact inventory — 분석 전에 존재와 실제 경로 확인

`huang2018/`, `06*/`, `07*/`의 요약·표·가공 예측을 탐색한다. API 캐시와 실패 raw 응답은 분석 대상으로 열지 않는다.
필수 입력이 없으면 즉시 중단한다. 07a notebook은 현재 저장소에 없으므로 실제 존재하는
`artifacts/07a_pipeline_skeleton/dev_sample_statistics.json`과 `scripts/run_07a.py`로 이력을 추적한다.
입력별 SHA-256은 마지막 `source_manifest.csv`에도 남긴다.''')
code('''inventory = analysis.inventory()
display(inventory)
print('필수 artifact:', int(inventory.required.sum()), '| 누락:', int((~inventory.exists).sum()))''')
md('''## 2. 07b integrity — 이후 모든 07b 표·그림의 실행 전제

출처: `artifacts/07b_rescue_lane_confirmatory/`의 `summary.json`, `hypothesis_tests.json`,
`prereg_07_v2.json`, `prereg_07_v2_1.json`, `tier_split.csv`, `eval_labels.csv`, `eval_sample.csv`,
`rankings.csv`, `effect_vs_04b.csv`, `exploratory.csv`, `cost_curve.csv`, `jev_R2_scores.csv`,
`llm_noRF_results.jsonl`, `api_attempt_log.jsonl`.

row key와 fault/sequence/원본 위치로 일대일 결합한다. 저장 순위는 변경하지 않는다.
혼동행렬·precision·recall·rescue precision·검토량은 저장 label/순위에서 재구성하며,
cost curve의 검토량/FN 및 summary의 효과와 대조한다. H1 하한은 원래 seed·시퀀스 재표집
10,000회·half-up E20으로 재현한다. H2/H3는 저장된 미검정 판정을 유지한다.
실제 호출/재시도/비용은 저장 요청 로그와 대조한다.

**07b와 전체 분석의 반올림 차이:** 07b는 사전등록 `floor(n×E+0.5)`를 유지한다.
T5는 요청된 Python `round(n×E)`를 사용하며 혼용하지 않는다.''')
code('''integrity = analysis.load_and_verify_07b()
print(f'PASS: {len(integrity):,}개 수치 비교; 모든 assert atol=1e-9, rtol=0')
display(integrity.groupby('status').size().rename('검증 수').to_frame())
display(pd.read_csv(analysis.tables / '07b_confusion_verified.csv'))''')
md('''## 3. T1. 04b 대 최종 구조 레인 — 07b 확증 test 표본

fault1+2와 fault3을 분리한다. **E10/E30 및 fault3은 탐색**, E20 recall은 확증 표본에서의 기술통계이다.
H1의 실제 검정 대상은 E20 rescue precision과 기저율의 차이이다.

출처: `effect_vs_04b.csv`, `summary.json`; 실제 fault group별 호출·비용의 출처는 `api_attempt_log.jsonl`이다.
‘회수 TP’는 baseline TP를 포함한 총 TP이다. baseline rescue precision은 적용 불가로 빈칸이다.

**비용 해석:** 07b는 전체 rescue에 LLM을 호출해 모든 routing 비교에 재사용했다.
따라서 ‘LLM 호출 수’는 해당 fault group의 실제 main+재시도+안정성 요청,
‘실제 API 비용’은 해당 group의 JEV+LLM 실험 총비용(USD)이다.
E별 독립 실행 비용이 아니므로 E10/E20/E30에 반복 기재된 값을 합산하면 안 된다.
04b 행의 0은 추가 API 비용이며 과거 학습/추론 비용을 뜻하지 않는다.

## T2. 07b 가설 검정 — 확증

출처: `hypothesis_tests.json`, `prereg_07_v2_1.json`. 판정 문자열을 그대로 보존한다.
하한 공란은 H2/H3가 gatekeeping으로 검정되지 않았음을 뜻한다.''')
code('''T1, T2 = analysis.comparison_tables()
display(T1)
display(T2)
display(Markdown('**사전등록 개정 v2 → v2.1:** ' + analysis.prereg['amendment_reason']))
print('MODEL_UNAVAILABLE:', analysis.summary['main_MODEL_UNAVAILABLE'], '건')
print('사전등록 고정 검정 순서:', ' → '.join(analysis.prereg['hypotheses']['order']))
print('H2/H3는 H1 통과 후에만 검정. 추가 검정 또는 새 판정 기준 없음.')''')
md('''## 4. T3. FabJudge 실험 경과 요약 — 06a~07b (탐색 회고; 07b 확증 판정 별도)

## T4. 실험 과정에서 확인한 주요 함정 — 탐색 회고

각 행의 실제 근거 경로를 저장한다. 06d 전체가 아니라 in-sample RF가 들어간 J2/S2 dev 및
이에 적합된 설정이 무효이다. 영향받지 않은 P3, B1, out-of-sample test 요청까지 무효로 확대하지 않는다.
06e의 시작점 baseline은 위치 누수 **가능성**이며, 새 성능 검정 결과로 해석하지 않는다.
early-warning은 기존 공통 80행의 LSTM false positive를 확인하며 5,000초 label을 바꾸지 않는다.

F6 후보는 과거의 연속된 P2/J2 질문을 고정하여 비교한다. 단계별 최댓값을 새로 선택하지 않는다.
06b에는 같은 P2가 없으므로 당시 sensor prompt임을 별도로 밝힌다.''')
code('''T3, T4 = analysis.history_tables()
display(T3)
display(T4)
display(analysis.history_auc)''')
md('''## 5. 전체 test endpoint 사전 검증 — 실패하면 T5 이후 중단

출처: `artifacts/huang2018/fault{1,2,3}_test_predictions.parquet`, `split_metadata.json`,
`sequence_metadata.csv`, `sequences/*.npz`.
04b의 `src/fabjudge/huang2018_models.py::combined_pipeline_metrics_causal`은
시퀀스 위치 `0,15,30,...`와 마지막 행을 endpoint로 저장한다.
**모든 원본 센서 행이라는 뜻이 아니라 04b가 정의한 causal test endpoint 전부**라는 뜻이다.

각 test sequence의 저장 가공 NPZ에서 원본 행 key 및 시간을 읽어 기대 endpoint 배열 전체와 비교한다.
샘플링·모델 추론·raw 데이터 읽기 없이 key, 시간, fault-time label, sequence 수, 전체 행 수를 검증한다.
또한 07b 720행과 예측/label/판정을 대조하고, 06a의 저장 `sample_plan.csv`에 담긴
정확한 fault3 80행을 key join하여 TP=9, FP=4, FN=7, TN=60을 확인한다.

고정 규칙: `y = wall_ttf_seconds <= 5000`, `lstm_alarm = lstm_pred_seconds <= 5000`,
`alarm_04b = lstm_alarm & (rf_probability >= 0.5)`, `rescue = lstm_alarm & ~alarm_04b`.''')
code('''# parquet별 schema와 수를 assert 전에 먼저 출력한다.
for fault in ('fault1', 'fault2', 'fault3'):
    path = ROOT / 'artifacts' / 'huang2018' / f'{fault}_test_predictions.parquet'
    frame = pd.read_parquet(path)
    print(fault, '| endpoints:', len(frame), '| sequences:', frame.sequence_id.nunique())
    print('columns:', frame.columns.tolist())
schema, coverage, exact80 = analysis.verify_full_endpoints()
display(schema)
display(coverage)
print('06a exact 80-row confusion:', exact80)''')
md('''## 6. F1~F6 — 07b 표본과 06 이력의 보고서용 그림

- **F1 (확증 표본 기술통계/탐색):** 04b와 E10/E20/E30. 무작위 순서 기대값은 각 fault의
  rescue 기저율×검토량을 합산한다. F2의 ‘무작위 라우팅’ 1,000회 평균과 다른 개념이다.
- **F2 (확증 H1 + 탐색):** JEV 막대만 H1 확증 대상이다. 전체 LLM, 무작위, LSTM-only,
  logistic은 탐색이며 H2/H3 미검정을 유의성 또는 동등성으로 해석하지 않는다.
- **F3 (탐색):** `cost_curve.csv`의 fault1+2, 검토 비용 1과 FN 비용 r (API 비용 별도).
- **F4 (탐색):** fault1+2 rescue의 endpoint key로 명시적 join 후 AUC 계산.
  LLM의 `MODEL_UNAVAILABLE`만 LLM AUC에서 제외하며 각 신호의 n과 제외 수를 기록한다.
- **F5 (탐색):** 경보 percentile별 양성률 및 전체 test sequence의 경보 endpoint 행 수.
  무경보 sequence도 길이 0으로 포함한다. 고정 최대 길이와 failure-anchored 구성의 위치 누수 진단이며
  실제 길이가 모두 같다고 가정하지 않는다. **사용 불가 / leakage diagnostic only**.
- **F6 (탐색):** 단계마다 LSTM 경보 내 JEV/RF 공통 유효행에서 AUC를 계산한다.
  06d J2는 ‘버그로 무효’; RF는 비교 기준인 OOF 값이다. 서로 다른 prompt·표본·split으로 인해
  단계 간 차이를 순수한 알고리즘 개선량으로 해석할 수 없다.

모든 그림은 Malgun Gothic, 동일 method-color mapping, PNG 300 dpi,
`bbox_inches="tight"`로 저장한다. F4/F5/F6의 수치와 표본 수는 별도 source CSV에도 남긴다.''')
code('''figures = analysis.sample_figures()
for fig in figures:
    display(fig)
    plt.close(fig)
display(analysis.signal_auc)''')
md('''## 7. T5. 전체 test endpoint 기준 04b 대 검토 확대 시나리오 — 탐색

**표 각주:** 07b H1은 JEV precision이 rescue 기저율보다 높다는 증거를 확보하지 못했다.
H3의 무작위 routing 직접 검정은 gatekeeping으로 미실시되었으므로 ‘동등성 입증’은 아니다.
이 제한 아래 전체 데이터의 검토 확대 성능을 **무작위 순서 기대값**으로 기술한다.
이는 실제 JEV/LLM 호출 결과가 아니다.

각 fault마다 `round(n_rescue * E)`개 검토 및 `검토량 × rescue 양성률`의 기대 TP를 계산한다.
fault1+2는 fault별 결과를 합산한다. 기대 TP/FN은 분수일 수 있다.
LSTM 단독과 rescue 전부 검토는 동일 경보 집합이다.

**CI 설계 (탐색):** 2,000회, seed=20260927. fault별 sequence 수를 유지하는 층화 시퀀스 bootstrap.
같은 시퀀스의 모든 endpoint를 함께 재표집하고, 중복된 시퀀스는 서로 다른 복제본으로 포함한다.
각 replicate/fault에서 rescue endpoint 복제본을 실제 무작위 순열로 섞고 `round(n×E)`개를 선택한다.
0~100% (5% 간격)의 각 E는 이 순열의 prefix를 사용하므로 E별 선택은 균등 비복원이며
각각 시퀀스 재표집과 random 선택 불확실성을 반영한다. paired baseline 대비 recall 변화도 같은 replicate에서 계산한다.
CI는 2.5/97.5 percentile이며 동시 신뢰대역은 아니다. endpoint 독립 bootstrap은 사용하지 않는다.

## T6. test sequence 단위 고장 사건 탐지 — 탐색

마지막 5,000초 내 경보가 한 번이라도 있으면 탐지한다.
‘첫 경보의 고장 전 시간’은 **해당 마지막 5,000초 내 첫 유효 경보**의 TTF(최댓값)이며 탐지된 sequence만 요약한다.
5k 밖 첫 경보를 섞지 않는다. 원본 test sequence 전부가 분모이고, 저장 endpoint에 5k 양성이 없는
sequence는 별도 열에 표시한다. 마지막 열은 구조 레인이 이론적으로 회수 가능한 사건 수이며 실제 회수 보장은 아니다.''')
code('''T5, T6 = analysis.full_endpoint_tables(repetitions=2000)
display(T5)
display(T6)''')
md('''## 8. T5b. 04b 원래 저장 지표 — 기존 평가 기술통계 (재계산 없음)

출처: `artifacts/huang2018/metrics.json`의 `rf`, `lstm/*/test_intrinsic`, `combined`만 복사한다.
**target 차이:** intrinsic RUL은 failure-anchored sample-count RUL이며, RF precision/F1은
저장된 inverse-probability weighted test 지표이다. combined의 과거 recall은 RF gate가 수치를
출력했는지에 대한 지표로서 현재 T5의 `LSTM<=5k & RF>=0.5` 정의와 다를 수 있다.
이를 현재 5k alarm 성능으로 재명명하지 않는다. `old_04`나 논문 수치는 04b 저장 지표로 대체하지 않는다.
04b R²는 저장된 값이 없어 ‘저장된 지표 없음’으로 기록한다.''')
code('''T5b = analysis.original_metrics()
display(T5b)''')
md('''## 9. F7. 전체 데이터: 추가 검토량 대비 recall — fault1+2 (탐색)

Panel A는 전체 endpoint의 무작위 기대값과 pointwise 95% bootstrap band,
Panel B는 07b 실제 관측 표본의 E10/E20/E30이다. 서로 다른 x/y축으로 분리하여 표시한다.
두 panel을 직접 겹쳐 ‘LLM ordering 효과’처럼 해석하지 않는다.''')
code('''fig = analysis.full_figure()
display(fig)
plt.close(fig)''')
md('''## 10. 최종 consistency check, 한국어 핵심 5줄 요약 및 manifest

07b 저장값, rule, 기존 80행, 전체 endpoint 완전성, recall/검토량 논리, 300 dpi,
파일 존재, 원본 SHA-256 불변을 assert한다. 마지막에 모든 CSV/PNG의 상대 경로·SHA-256을 출력하고
`artifacts/08_final_comparison/manifest.csv`에 저장한다.
manifest 자신은 자기참조를 피하기 위해 manifest 행에 넣지 않는다.''')
code('''summary, consistency, manifest = analysis.finish()
print(summary)
display(consistency)
print('생성한 모든 tables/*.csv 및 figures/*.png (relative path, SHA-256):')
print(manifest.to_string(index=False))
assert len(manifest) == len(list(analysis.tables.glob('*.csv'))) + len(list(analysis.figures.glob('*.png')))
print('PASS: manifest 저장 및 모든 SHA-256 재검증 완료')''')

nb = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3 (FabJudge)', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.12'},
})
path = ROOT/'notebooks/08_final_comparison.ipynb'
nbf.write(nb, path)
print(path)
