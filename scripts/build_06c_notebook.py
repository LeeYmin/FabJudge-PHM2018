"""Build the reviewable 06c experiment notebook; execution is separate."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = nb.cells

def md(value):
    cells.append(nbf.v4.new_markdown_cell(value.strip()))

def code(value):
    cells.append(nbf.v4.new_code_cell(value.strip()))

md(r"""
# 06c · LSTM–JEV semantic gate

**질문:** fault3의 LSTM 5,000초 경보에 센서 의미와 dev에서 정한 연속 임계값을 더하면 거짓 경보가 줄어드는가? JEV를 단순 규칙 및 로지스틱 회귀와 비교한다. 06b의 test 관찰은 가설이며 feature, 방향, 프롬프트 수치, 임계값을 결정하는 데 사용하지 않는다.

실험 순서는 dev 분리 → causal 추론 재현 → feature/질문 선택 → dev JEV/기준선 → 설정 고정 → test 한 번 평가다. `data/raw/`는 사용하지 않는다. NPZ에서 현재와 과거의 `x`, `time`, `raw_row_index`만 읽으며 `samples_to_fault`는 열지 않는다. JEV 판단은 측정값의 정답을 대체하지 않는다.
""")

md(r"""
## 0–1. 저장된 모델과 dev 분리

**가정:** 04b의 causal endpoint는 원본 시퀀스의 15행 간격 endpoint와 마지막 행이다. 06a와 동일하게 각 endpoint 목록의 5, 15, …, 95 백분위 위치에서 `floor(p × (n−1))`를 쓴다. validation 경보가 20건 미만일 때만 train 시퀀스로 전환한다. 체크포인트는 추론에만 쓰며, 저장된 test 예측 첫 5행을 절대오차 1e-6 이내로 재현한 뒤 dev를 계산한다. test의 라벨·경보 성능은 이 단계에서 보지 않는다.
""")

code(r"""
import sys, json, math, hashlib, difflib
from pathlib import Path
import joblib, numpy as np, pandas as pd, torch
from IPython.display import display
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT = Path.cwd() if (Path.cwd() / 'src/fabjudge/paths.py').exists() else Path.cwd().parent
sys.path.insert(0, str(PROJECT / 'src'))
from fabjudge.paths import PROJECT_ROOT as ROOT
from fabjudge.huang2018_models import SequenceRUL
from fabjudge.jev_gate import (CANDIDATES, JEVClient, causal_sensor_features,
    effect_size, infer_causal_sampled_rows, metrics)
assert PROJECT.resolve() == ROOT.resolve()
torch.set_num_threads(2)
ART = ROOT / 'artifacts/huang2018'
OUT = ROOT / 'artifacts/06c_lstm_jev_semantic_gate'
OUT.mkdir(parents=True, exist_ok=True)
SPLIT = json.loads((ART / 'split_metadata.json').read_text(encoding='utf-8'))['splits']['fault3']
SEQUENCES = pd.read_csv(ART / 'sequence_metadata.csv')
INPUT_COLUMNS = json.loads((ART / 'reproduction_config.json').read_text(encoding='utf-8'))['model_input_columns']
RF = joblib.load(ROOT / 'artifacts/models/huang2018/fault3_rf.joblib')
saved_checkpoint = torch.load(ROOT / 'artifacts/models/huang2018/fault3_lstm.pt', map_location='cpu', weights_only=True)
LSTM = SequenceRUL(**saved_checkpoint['architecture'])
LSTM.load_state_dict(saved_checkpoint['state_dict'])
LSTM.eval()
split_counts = {key: len(SPLIT[f'{key}_original_sequences']) for key in ('train','validation','test')}
assert len(set(SPLIT['train_original_sequences']) & set(SPLIT['test_original_sequences'])) == 0
assert len(set(SPLIT['validation_original_sequences']) & set(SPLIT['test_original_sequences'])) == 0
print('fault3 split:', split_counts)

# A few exact saved 04b rows are a parity check, never a fitting set.
parity_id = SPLIT['test_original_sequences'][0]
_, parity_rows = infer_causal_sampled_rows(ROOT, SEQUENCES, [parity_id], RF, LSTM, INPUT_COLUMNS)
reference = pd.read_parquet(ART / 'fault3_test_predictions.parquet', columns=[
    'sequence_id','raw_row_index','time','wall_ttf_seconds','lstm_pred_seconds',
    'rf_probability','pipeline_rul_seconds'])
assert set(reference.sequence_id) == set(SPLIT['test_original_sequences'])
parity = parity_rows.merge(reference[['sequence_id','raw_row_index','lstm_pred_seconds','rf_probability']],
    on=['sequence_id','raw_row_index'], suffixes=('_new','_saved'), validate='one_to_one').head(5)
parity_errors = {field: float(np.max(np.abs(parity[f'{field}_new'] - parity[f'{field}_saved'])))
                 for field in ('lstm_pred_seconds','rf_probability')}
print('04b parity, first 5 rows, max absolute error:', parity_errors)
if any(error > 1e-6 for error in parity_errors.values()):
    raise RuntimeError('Stop: checkpoint inference does not reproduce 04b test predictions')

validation, _ = infer_causal_sampled_rows(ROOT, SEQUENCES, SPLIT['validation_original_sequences'], RF, LSTM, INPUT_COLUMNS)
validation_alarms = int(validation.lstm_pred_seconds.le(5000).sum())
DEV_SPLIT = 'validation' if validation_alarms >= 20 else 'train'
DEV_REASON = ('validation has at least 20 LSTM alarms' if DEV_SPLIT == 'validation'
              else f'validation has only {validation_alarms} LSTM alarms (<20)')
REPLAY = (OUT / 'frozen_config.json').exists() and (OUT / 'dev_rows.csv').exists()
if REPLAY:
    dev = pd.read_csv(OUT / 'dev_rows.csv', float_precision='round_trip')
    old_frozen = json.loads((OUT / 'frozen_config.json').read_text(encoding='utf-8'))
    assert old_frozen['dev_split'] == DEV_SPLIT
    assert set(dev.sequence_id) == set(SPLIT[f'{DEV_SPLIT}_original_sequences'])
    print('Replaying the already frozen experiment from exact saved dev states and JEV cache.')
else:
    dev = validation if DEV_SPLIT == 'validation' else infer_causal_sampled_rows(
        ROOT, SEQUENCES, SPLIT['train_original_sequences'], RF, LSTM, INPUT_COLUMNS)[0]
dev['actual_fail_5000'] = dev.wall_ttf_seconds.le(5000)
dev['lstm_alarm'] = dev.lstm_pred_seconds.le(5000)
dev['pipeline_alarm'] = dev.rf_probability.ge(.5) & dev.lstm_alarm
dev_alarms = dev.loc[dev.lstm_alarm].copy()
dev_tp, dev_fp = int(dev_alarms.actual_fail_5000.sum()), int((~dev_alarms.actual_fail_5000).sum())
print({'dev_split':DEV_SPLIT, 'reason':DEV_REASON, 'rows':len(dev),
       'sequences':dev.sequence_id.nunique(), 'alarms':len(dev_alarms), 'TP':dev_tp, 'FP':dev_fp})
if len(dev_alarms) < 20:
    raise RuntimeError('Stop: dev LSTM alarms below 20')
HISTORY_IN_STATE = dev_alarms.history_count.nunique() > 1
print('history_count in state:', HISTORY_IN_STATE, '; unique dev alarm values:', dev_alarms.history_count.unique().tolist())
if not REPLAY:
    dev.to_csv(OUT / 'dev_rows.csv', index=False)
""")

md(r"""
## 2. dev feature 선택 및 의미

**가정:** 06b의 12개 후보에 대해 효과 크기 `(TP median − FP median) / sqrt((TP IQR² + FP IQR²)/2)`를 계산한다. 절댓값 순으로 후보를 보며 상관 절댓값이 0.9 미만인 최상위 쌍을 고른다. 두 효과가 모두 0.4 미만이면 P2의 센서 입력을 생략한다. 각 경계는 dev TP/FP 중앙값의 중간점이고, 그 중 FP 중앙값 쪽을 보류 방향으로 정의한다. 현재값·RMS·peak에는 보편적인 ‘1.0 안정’ 의미가 없으므로 수치 1.0은 shape factor에만 설명할 수 있다. 시퀀스 NPZ에 RECIPE/RECIPE_STEP이 없으면 운전 상태 전환과의 관계는 ‘미확인’으로 기록한다.
""")

code(r"""
rows = []
for feature in CANDIDATES:
    effect, tp_med, fp_med, tp_iqr, fp_iqr = effect_size(
        dev_alarms.loc[dev_alarms.actual_fail_5000, feature],
        dev_alarms.loc[~dev_alarms.actual_fail_5000, feature])
    rows.append({'feature':feature,'effect_size':effect,'tp_median':tp_med,'fp_median':fp_med,
                 'tp_iqr':tp_iqr,'fp_iqr':fp_iqr})
feature_table = pd.DataFrame(rows)
ranking = feature_table.loc[feature_table.effect_size.notna()].copy()
ranking['abs_effect'] = ranking.effect_size.abs()
ranking = ranking.sort_values(['abs_effect','feature'],ascending=[False,True],kind='stable')
SELECTED, selected_r = [], None
for feature in ranking.feature:
    if not SELECTED:
        SELECTED.append(feature)
    elif abs(float(dev_alarms[SELECTED[0]].corr(dev_alarms[feature]))) < .9:
        selected_r = float(dev_alarms[SELECTED[0]].corr(dev_alarms[feature]))
        SELECTED.append(feature)
        break
SENSOR_ENABLED = len(SELECTED) == 2 and not all(
    abs(float(feature_table.set_index('feature').loc[name,'effect_size'])) < .4 for name in SELECTED)
if len(SELECTED) < 2:
    SENSOR_ENABLED = False
SENSOR_RULES = {}
for name in SELECTED:
    row = feature_table.set_index('feature').loc[name]
    boundary = float((row.tp_median + row.fp_median) / 2)
    SENSOR_RULES[name] = {'boundary':boundary,
        'fp_side':'above' if row.fp_median > row.tp_median else 'below',
        'tp_median':float(row.tp_median), 'fp_median':float(row.fp_median),
        'effect_size':float(row.effect_size)}
feature_table['selected'] = feature_table.feature.isin(SELECTED)
feature_table['boundary'] = feature_table.feature.map(lambda x: SENSOR_RULES.get(x,{}).get('boundary'))
feature_table['fp_side'] = feature_table.feature.map(lambda x: SENSOR_RULES.get(x,{}).get('fp_side'))
feature_table.to_csv(OUT / 'feature_selection_dev.csv', index=False)
with np.load(ROOT / str(SEQUENCES.set_index('sequence_id').loc[dev.sequence_id.iloc[0], 'sequence_path'])) as sample_npz:
    recipe_status = '미확인' if not {'RECIPE','RECIPE_STEP'}.issubset(sample_npz.files) else 'available'
print({'selected':SELECTED,'selected_correlation':selected_r,'sensor_P2_enabled':SENSOR_ENABLED,
       'recipe_transition':recipe_status})
display(feature_table.sort_values('effect_size',key=lambda s:s.abs(),ascending=False).head(12))
""")

md(r"""
## 3. 질문 세트

**가정:** P0는 06a `prompt_tune_01` 질문 원문과 06b 형식의 상태(핵심 신호와 dev에서 선택한 센서 쌍)를 결합한다. 이전 test에서 고른 shape factor 이름은 재사용하지 않는다. `history_count`가 dev 경보에서 상수이면 모든 조건의 상태에서 제외한다. P1/P2는 하나의 공통 질문에서 센서 문장과 criteria의 센서 구절만 삽입해 만들고, 그 diff를 출력한다. 따라서 P0→P1은 질문과 센서 상태가 함께 달라지는 비교이며 순수 프롬프트 효과로 해석하지 않는다.
""")

code(r"""
P0 = json.loads((ROOT / 'artifacts/06a_lstm_jev/prompt_tune_01/decision_question.json').read_text(encoding='utf-8'))
core = ('Using only state.features, evaluate the evidence for keeping this LSTM alarm for failure within 5000 seconds. '
        'Do not estimate a new time to failure. Weigh supporting and contradicting evidence in a balanced way. '
        'lstm_pred_seconds: smaller values indicate a more imminent fault. '
        'rf_probability: estimated probability of a fault within 5000 seconds. '
        'recent_lstm_delta: current minus oldest of the last five LSTM predictions; negative means the predicted failure time is falling. '
        'recent_lstm_std: variability across those recent predictions. '
        'Assess the signals together and choose Weak, Uncertain, or Strong based on the evidence.')
base_criteria = [
    'Weak: the LSTM trend and RF signal give weak support for keeping the alarm',
    'Uncertain: the LSTM trend and RF signal give mixed or insufficient support for keeping the alarm',
    'Strong: the LSTM trend and RF signal give strong support for keeping the alarm']
sensor_lines = []
if SENSOR_ENABLED:
    for name in SELECTED:
        rule = SENSOR_RULES[name]
        kind = ('shape factor of the current and preceding 99 readings (1.0 means stable magnitude)'
                if name.endswith('__shape_factor') else
                'RMS magnitude of the current and preceding 99 readings' if name.endswith('__rms') else
                'peak absolute magnitude of the current and preceding 99 readings' if name.endswith('__peak_abs') else
                'current sensor reading')
        sensor_lines.append(f'{name}: {kind}. In dev alarms, TP median={rule["tp_median"]:.6g}, '
            f'FP median={rule["fp_median"]:.6g}; midpoint={rule["boundary"]:.6g}, '
            f'with the FP-associated side {rule["fp_side"]} the midpoint. '
            'This is an association in dev, not proof of a physical fault mechanism.')
P1 = {'type':'score','instructions':core,'criteria':base_criteria}
P2 = {'type':'score','instructions':core + ((' ' + ' '.join(sensor_lines)) if SENSOR_ENABLED else ''),
      'criteria':[line.replace('LSTM trend and RF signal',
          'LSTM trend, RF signal, and sensor evidence') for line in base_criteria] if SENSOR_ENABLED else base_criteria}
QUESTIONS = {'P0':P0,'P1':P1,'P2':P2}
qdir = OUT / 'questions'
qdir.mkdir(exist_ok=True)
for name, question in QUESTIONS.items():
    assert question['type'] == 'score' and len(question['criteria']) == 3
    (qdir / f'{name}.json').write_text(json.dumps(question,ensure_ascii=False,indent=2),encoding='utf-8')
print('P1 → P2 diff:')
print(''.join(difflib.unified_diff(json.dumps(P1,ensure_ascii=False,indent=2).splitlines(True),
                                   json.dumps(P2,ensure_ascii=False,indent=2).splitlines(True),
                                   fromfile='P1',tofile='P2')))
""")

md(r"""
## 4. dev JEV와 기준선

**가정:** JEV의 native score와 `probabilities["2"]`는 값이 클수록 경보 유지 근거가 강하다. 임계값은 `값 ≥ t`이면 유지한다. dev에서 TP 손실이 `max(1, TP × 0.1)` 이하인 후보 중 FP 제거가 최대인 임계값을 선택하고 동률에서는 더 낮은 임계값을 쓴다. P0/P1/P2마다 두 점수를 모두 평가하고 dev FP 제거가 큰 점수를 선택한다(동률은 TP 손실이 적은 쪽, 다시 동률이면 native score). B1도 같은 규칙이다. B2는 dev 중앙값 중간점에서 FP 쪽으로 넘어간 feature가 하나라도 있으면 보류하며, TP 손실 제약을 어기면 no-op으로 둔다. B3는 dev 경보에만 표준화+기본 `C=1` 로지스틱 회귀를 맞춘 뒤 임계값을 선택한다. B3 dev 성능은 학습 집합의 성능이다.

호출 상한은 dev 경보 수×3 + 안정성 5×3 + test 23×3이다. 반복 요청은 API body를 그대로 두고 cache key에만 `repeat_index`를 넣는다. 첫 JEV 응답이 실패하거나 schema 검증에 실패하면 후속 호출 없이 중단한다. 각 dev 조건에 실패가 남으면 임계값을 결정하지 않는다.
""")

code(r"""
MAX_NEW_CALLS = len(dev_alarms)*3 + 5*3 + 23*3
print({'dev_alarm_rows':len(dev_alarms),'dev_main_calls':len(dev_alarms)*3,
       'stability_calls':15,'test_calls':69,'MAX_NEW_CALLS':MAX_NEW_CALLS})
CLIENT = JEVClient(ROOT, OUT / 'cache', 0 if REPLAY else MAX_NEW_CALLS)
response_records = []
for name in ('P0','P1','P2'):
    chosen_features = SELECTED if name == 'P0' or (name == 'P2' and SENSOR_ENABLED) else []
    for ordinal, (idx, row) in enumerate(dev_alarms.iterrows()):
        result = CLIENT.call_jev(row, QUESTIONS[name], chosen_features,
                                 include_history=HISTORY_IN_STATE)
        response_records.append({'split':'dev','condition':name,'row_index':int(idx),
                                 'repeat_index':None,'result':result})
        dev.at[idx,f'{name}_score'] = result['score']
        dev.at[idx,f'{name}_probability_2'] = result['probability_2']
        dev.at[idx,f'{name}_status'] = result['status']
        dev.at[idx,f'{name}_error'] = result['error']
        if ordinal == 0:
            response = result['response'] or {}
            print(name,'first response:',{'model':response.get('model'),'provider':response.get('provider'),
                  'schema_valid':result['score'] is not None,'error':result['error']})
            if result['score'] is None:
                raise RuntimeError('Stop: first JEV response invalid or unavailable')
    print(name,'complete; new API calls:',CLIENT.new_calls,'cache hits:',CLIENT.cache_hits)
if dev.loc[dev.lstm_alarm,[f'{name}_{field}' for name in QUESTIONS for field in ('score','probability_2')]].isna().any().any():
    dev.to_csv(OUT / 'dev_rows.csv',index=False)
    raise RuntimeError('Stop: dev JEV responses unresolved, thresholds cannot be fitted')

stability_records = []
for idx, row in dev_alarms.head(5).iterrows():
    for repeat_index in range(3):
        result = CLIENT.call_jev(row, QUESTIONS['P2'], SELECTED if SENSOR_ENABLED else [],
                                 include_history=HISTORY_IN_STATE, repeat_index=repeat_index)
        response_records.append({'split':'stability','condition':'P2','row_index':int(idx),
                                 'repeat_index':repeat_index,'result':result})
        stability_records.append({'row_index':int(idx),'sequence_id':row.sequence_id,
                                  'raw_row_index':int(row.raw_row_index),'repeat_index':repeat_index,
                                  'score':result['score'],'probability_2':result['probability_2'],
                                  'status':result['status'],'error':result['error']})
stability = pd.DataFrame(stability_records)
if not REPLAY:
    stability.to_csv(OUT / 'stability.csv',index=False)
if stability.score.isna().any():
    raise RuntimeError('Stop: stability measurements unresolved')
stability_sd = stability.groupby('row_index').score.std(ddof=1)
print('P2 repeat score SD:',stability_sd.to_dict())

def fit_threshold(values, labels):
    values = np.asarray(values,dtype=float)
    labels = np.asarray(labels,dtype=bool)
    assert np.isfinite(values).all() and labels.any() and (~labels).any()
    allowed = max(1.0, .1*int(labels.sum()))
    candidates = np.r_[np.nextafter(values.min(),-np.inf),np.unique(values)]
    options = []
    for threshold in candidates:
        keep = values >= threshold
        lost_tp = int((labels & ~keep).sum())
        if lost_tp <= allowed:
            options.append({'threshold':float(threshold),'removed_fp':int((~labels & ~keep).sum()),
                            'lost_tp':lost_tp})
    return sorted(options,key=lambda x:(-x['removed_fp'],x['threshold']))[0]

labels = dev_alarms.actual_fail_5000.to_numpy(bool)
dev_gate_rows = []
GATES = {}
for name in QUESTIONS:
    choices = {}
    for field in ('score','probability_2'):
        values = dev.loc[dev.lstm_alarm,f'{name}_{field}'].to_numpy(float)
        choice = fit_threshold(values,labels)
        choice.update({'auc':float(roc_auc_score(labels,values)),'field':field})
        choices[field] = choice
        dev_gate_rows.append({'condition':name,'score_field':field,**choice,
                              'dev_TP_kept':dev_tp-choice['lost_tp'],
                              'dev_FP_kept':dev_fp-choice['removed_fp']})
    winner = sorted(choices.values(),key=lambda x:(-x['removed_fp'],x['lost_tp'],
                                                    0 if x['field']=='score' else 1))[0]
    GATES[name] = {'selected':winner,'candidates':choices}

B1 = fit_threshold(dev_alarms.rf_probability,labels)
B1['auc'] = float(roc_auc_score(labels,dev_alarms.rf_probability))
def sensor_fp_side(frame, name):
    rule = SENSOR_RULES[name]
    return frame[name].gt(rule['boundary']) if rule['fp_side']=='above' else frame[name].lt(rule['boundary'])
B2_hold = pd.Series(False,index=dev_alarms.index)
if SENSOR_ENABLED:
    for name in SELECTED:
        B2_hold |= sensor_fp_side(dev_alarms,name)
B2_lost = int((B2_hold & dev_alarms.actual_fail_5000).sum())
B2_active = bool(SENSOR_ENABLED and B2_lost <= max(1,.1*dev_tp))
if not B2_active:
    B2_hold[:] = False
B2 = {'active':B2_active,'lost_tp':int((B2_hold & dev_alarms.actual_fail_5000).sum()),
      'removed_fp':int((B2_hold & ~dev_alarms.actual_fail_5000).sum()),
      'boundary_rule':'hold when any selected feature is on dev FP side of median midpoint'}
BASE_FEATURES = ['rf_probability'] + (SELECTED if SENSOR_ENABLED else [])
scaler = StandardScaler().fit(dev_alarms[BASE_FEATURES])
model = LogisticRegression(C=1.0,max_iter=1000,random_state=42).fit(
    scaler.transform(dev_alarms[BASE_FEATURES]),labels.astype(int))
B3_values = model.predict_proba(scaler.transform(dev_alarms[BASE_FEATURES]))[:,1]
B3 = fit_threshold(B3_values,labels)
B3['auc'] = float(roc_auc_score(labels,B3_values))
B3['features'] = BASE_FEATURES
B3['scaler_mean'] = scaler.mean_.tolist()
B3['scaler_scale'] = scaler.scale_.tolist()
B3['coef'] = model.coef_[0].tolist()
B3['intercept'] = float(model.intercept_[0])
dev_gate_rows += [
    {'condition':'B1','score_field':'rf_probability',**B1,'dev_TP_kept':dev_tp-B1['lost_tp'],'dev_FP_kept':dev_fp-B1['removed_fp']},
    {'condition':'B2','score_field':'sensor_rule','threshold':None,'auc':float(roc_auc_score(labels,(~B2_hold).astype(int))),
     'removed_fp':B2['removed_fp'],'lost_tp':B2['lost_tp'],'dev_TP_kept':dev_tp-B2['lost_tp'],'dev_FP_kept':dev_fp-B2['removed_fp']},
    {'condition':'B3','score_field':'logistic_probability','threshold':B3['threshold'],'auc':B3['auc'],
     'removed_fp':B3['removed_fp'],'lost_tp':B3['lost_tp'],'dev_TP_kept':dev_tp-B3['lost_tp'],'dev_FP_kept':dev_fp-B3['removed_fp']}]
dev_gate_table = pd.DataFrame(dev_gate_rows)
display(dev_gate_table)
if not REPLAY:
    dev.to_csv(OUT / 'dev_rows.csv',index=False)
""")

md(r"""
## 5. 설정 고정

**가정:** 아래 JSON에는 dev에서 결정한 값만 쓴다. 파일 바이트의 SHA-256을 기록하고 다음 셀에서 다시 확인한다. 이후 feature, 질문, 임계값, 기준선 계수는 변경하지 않는다. 변경이 필요하면 06d에서 새 실험으로 수행한다.
""")

code(r"""
FROZEN = {'design':'fault3 causal 06c, dev-tuned before one test evaluation',
    'dev_split':DEV_SPLIT,'dev_reason':DEV_REASON,'dev_sequences':sorted(dev.sequence_id.unique().tolist()),
    'dev_rows':len(dev),'dev_alarm_rows':len(dev_alarms),'dev_TP':dev_tp,'dev_FP':dev_fp,
    'feature_selection_rule':'highest |effect| pair with |Pearson r|<0.9; skip P2 sensors if both |effect|<0.4',
    'selected_features':SELECTED,'selected_correlation':selected_r,
    'sensor_enabled':SENSOR_ENABLED,'sensor_rules':SENSOR_RULES,
    'history_count_in_state':HISTORY_IN_STATE,'recipe_transition':recipe_status,
    'questions':QUESTIONS,'gates':GATES,'B1':B1,'B2':B2,'B3':B3,
    'threshold_rule':'keep when score >= t; TP loss <= max(1,0.1*dev TP); maximize FP removal; tie lower t',
    'parity_max_abs_error':parity_errors}
FROZEN_PATH = OUT / 'frozen_config.json'
payload = json.dumps(FROZEN,ensure_ascii=False,indent=2,allow_nan=False)
if REPLAY:
    assert json.loads(FROZEN_PATH.read_text(encoding='utf-8')) == FROZEN, 'Frozen choices changed on replay'
else:
    FROZEN_PATH.write_text(payload,encoding='utf-8')
FROZEN_SHA256 = hashlib.sha256(FROZEN_PATH.read_bytes()).hexdigest()
print('frozen_config.json sha256:',FROZEN_SHA256)
""")

md(r"""
## 6. 고정된 설정으로 test 한 번 평가

**가정:** 06a `larger_test_01`의 80개 행 키를 사용하고 저장된 04b 예측과 exact join한다. 원본 06b cell 1의 baseline 혼동행렬 TP 15, FP 8, FN 1, TN 56을 확인한 다음에만 P0/P1/P2를 호출한다. 센서 후보는 현재 행과 직전 99행의 저장된 `x`로 계산한다. JEV 실패/미시도 경보는 unresolved로 남기고 그 조건의 전체 혼동행렬과 bootstrap CI를 계산하지 않는다. 시퀀스 bootstrap은 8개 시퀀스를 복원 추출해 1,000회 반복한다. 같은 fault3 분포에서 고른 dev 설정을 독립적인 다른 공정으로 일반화하지 않는다.
""")

code(r"""
if hashlib.sha256(FROZEN_PATH.read_bytes()).hexdigest() != FROZEN_SHA256:
    raise RuntimeError('Frozen configuration changed before test')
pilot = pd.read_csv(ROOT / 'artifacts/06a_lstm_jev/larger_test_01/pilot_rows.csv')
pilot_summary = json.loads((ROOT / 'artifacts/06a_lstm_jev/larger_test_01/summary.json').read_text(encoding='utf-8'))
assert len(pilot)==80 and pilot.sequence_id.nunique()==8 and not pilot.duplicated(['sequence_id','raw_row_index']).any()
assert set(pilot.sequence_id)==set(SPLIT['test_original_sequences'])
assert set(reference.sequence_id)==set(SPLIT['test_original_sequences'])
test = pilot[['sequence_id','raw_row_index','sample_position','recent_lstm_delta',
              'recent_lstm_std','history_count']].merge(reference,on=['sequence_id','raw_row_index'],
                                                        how='left',validate='one_to_one',sort=False)
assert len(test)==80 and not test[['lstm_pred_seconds','rf_probability','wall_ttf_seconds']].isna().any().any()
for field in ('lstm_pred_seconds','rf_probability'):
    assert np.allclose(test[field],pilot[field],rtol=0,atol=1e-9)
test['actual_fail_5000']=test.wall_ttf_seconds.le(5000)
test['lstm_alarm']=test.lstm_pred_seconds.le(5000)
test['pipeline_alarm']=test.pipeline_rul_seconds.le(5000)
assert np.array_equal(test.actual_fail_5000.to_numpy(),pilot.actual_fail_5000.to_numpy())
baseline=metrics('LSTM only',test.lstm_alarm,test.actual_fail_5000)
expected={'TP':15,'FP':8,'FN':1,'TN':56}
if any(baseline[k]!=v for k,v in expected.items()) or int(test.lstm_alarm.sum())!=23 or any(
        pilot_summary['metrics'][0][k]!=v for k,v in expected.items()):
    raise RuntimeError(f'Stop: 06b cell 1 test integrity failed: {baseline}')
print('06b test integrity passed:',baseline)

seq_paths=SEQUENCES.set_index('sequence_id')['sequence_path']
for sid, group in test.groupby('sequence_id',sort=False):
    path=(ROOT / str(seq_paths.loc[sid]).replace('\\','/')).resolve()
    if not path.is_relative_to((ROOT / 'artifacts/huang2018/sequences').resolve()):
        raise RuntimeError('Sequence artifact outside expected directory')
    with np.load(path) as data:
        source_rows=data['raw_row_index']
        x=data['x']
    positions=np.searchsorted(source_rows,group.raw_row_index.to_numpy())
    if np.any(positions>=len(source_rows)) or not np.array_equal(source_rows[positions],group.raw_row_index.to_numpy()):
        raise RuntimeError('Test row missing exact saved sensor measurement')
    for index,current in zip(group.index,positions):
        for name,value in causal_sensor_features(x,int(current),INPUT_COLUMNS).items():
            test.at[index,name]=value

for name in ('P0','P1','P2'):
    test[f'{name}_score']=np.nan
    test[f'{name}_probability_2']=np.nan
    test[f'{name}_status']=np.where(test.lstm_alarm,'not_attempted','not_called')
    test[f'{name}_error']=None
    features=FROZEN['selected_features'] if name=='P0' or (name=='P2' and FROZEN['sensor_enabled']) else []
    for idx,row in test.loc[test.lstm_alarm].iterrows():
        result=CLIENT.call_jev(row,FROZEN['questions'][name],features,
                               include_history=FROZEN['history_count_in_state'])
        response_records.append({'split':'test','condition':name,'row_index':int(idx),
                                 'repeat_index':None,'result':result})
        for field in ('score','probability_2','status','error'):
            test.at[idx,f'{name}_{field}']=result[field]
    print(name,'test JEV complete; successful:',int(test.loc[test.lstm_alarm,f'{name}_score'].notna().sum()),
          'new API calls:',CLIENT.new_calls,'cache hits:',CLIENT.cache_hits)

def logistic_probability(frame, frozen):
    x=frame[frozen['features']].to_numpy(float)
    z=(x-np.asarray(frozen['scaler_mean']))/np.asarray(frozen['scaler_scale'])
    logits=z@np.asarray(frozen['coef'])+frozen['intercept']
    return 1/(1+np.exp(-logits))

for name in ('P0','P1','P2'):
    gate=FROZEN['gates'][name]['selected']
    test[f'{name}_alarm']=pd.Series(False,index=test.index,dtype='boolean')
    test.loc[test.lstm_alarm,f'{name}_alarm']=pd.NA
    resolved=test.lstm_alarm & test[f'{name}_{gate["field"]}'].notna()
    test.loc[resolved,f'{name}_alarm']=test.loc[resolved,f'{name}_{gate["field"]}'].ge(gate['threshold']).to_numpy()
test['B1_alarm']=test.lstm_alarm & test.rf_probability.ge(FROZEN['B1']['threshold'])
B2_hold_test=pd.Series(False,index=test.index)
if FROZEN['B2']['active']:
    for feature in FROZEN['selected_features']:
        rule=FROZEN['sensor_rules'][feature]
        B2_hold_test |= (test[feature].gt(rule['boundary']) if rule['fp_side']=='above'
                         else test[feature].lt(rule['boundary']))
test['B2_alarm']=test.lstm_alarm & ~B2_hold_test
test['B3_probability']=logistic_probability(test,FROZEN['B3'])
test['B3_alarm']=test.lstm_alarm & test.B3_probability.ge(FROZEN['B3']['threshold'])

systems={'LSTM only':'lstm_alarm','RF to LSTM pipeline':'pipeline_alarm',
         **{name:f'{name}_alarm' for name in ('P0','P1','P2')},
         'B1':'B1_alarm','B2':'B2_alarm','B3':'B3_alarm'}
comparison=[]
sequence_results=[]
rng=np.random.default_rng(42)
ids=np.asarray(sorted(test.sequence_id.unique()))
samples=rng.choice(ids,size=(1000,len(ids)),replace=True)
for name,column in systems.items():
    entry=metrics(name,test[column],test.actual_fail_5000)
    if name in ('P0','P1','P2'):
        values=test.loc[test.lstm_alarm,f'{name}_{FROZEN["gates"][name]["selected"]["field"]}']
        entry['auc_TP_vs_FP']=float(roc_auc_score(test.loc[test.lstm_alarm,'actual_fail_5000'],values)) if values.notna().all() else None
        for field in ('score','probability_2'):
            vals=test.loc[test.lstm_alarm,f'{name}_{field}']
            entry[f'auc_{field}']=float(roc_auc_score(test.loc[test.lstm_alarm,'actual_fail_5000'],vals)) if vals.notna().all() else None
    elif name=='B1':
        entry['auc_TP_vs_FP']=float(roc_auc_score(test.loc[test.lstm_alarm,'actual_fail_5000'],test.loc[test.lstm_alarm,'rf_probability']))
    elif name=='B3':
        entry['auc_TP_vs_FP']=float(roc_auc_score(test.loc[test.lstm_alarm,'actual_fail_5000'],test.loc[test.lstm_alarm,'B3_probability']))
    elif name=='B2':
        entry['auc_TP_vs_FP']=float(roc_auc_score(test.loc[test.lstm_alarm,'actual_fail_5000'],
                                               test.loc[test.lstm_alarm,'B2_alarm'].astype(int)))
    else:
        entry['auc_TP_vs_FP']=None
    if entry['unresolved_rows']==0:
        pred=test[column].astype(bool)
        entry['removed_FP']=int((test.lstm_alarm & ~pred & ~test.actual_fail_5000).sum())
        entry['lost_TP']=int((test.lstm_alarm & ~pred & test.actual_fail_5000).sum())
        boots={'precision':[],'removed_FP':[],'lost_TP':[]}
        for chosen in samples:
            frame=pd.concat([test.loc[test.sequence_id.eq(sid)] for sid in chosen],ignore_index=True)
            pred_b=frame[column].astype(bool)
            tp=int((pred_b & frame.actual_fail_5000).sum())
            fp=int((pred_b & ~frame.actual_fail_5000).sum())
            boots['precision'].append(tp/(tp+fp) if tp+fp else np.nan)
            boots['removed_FP'].append(int((frame.lstm_alarm & ~pred_b & ~frame.actual_fail_5000).sum()))
            boots['lost_TP'].append(int((frame.lstm_alarm & ~pred_b & frame.actual_fail_5000).sum()))
        for field,values in boots.items():
            arr=np.asarray(values,dtype=float)
            entry[f'{field}_ci_low']=float(np.nanquantile(arr,.025)) if np.isfinite(arr).any() else None
            entry[f'{field}_ci_high']=float(np.nanquantile(arr,.975)) if np.isfinite(arr).any() else None
    else:
        entry['removed_FP']=entry['lost_TP']=None
        for field in ('precision','removed_FP','lost_TP'):
            entry[f'{field}_ci_low']=entry[f'{field}_ci_high']=None
    comparison.append(entry)
    for sid,group in test.groupby('sequence_id',sort=True):
        if group[column].isna().any():
            first=None; first_time=None; has_fp=None
        else:
            fired=group.loc[group[column].astype(bool)].sort_values('raw_row_index')
            first=int(fired.raw_row_index.iloc[0]) if len(fired) else None
            first_time=int(fired.time.iloc[0]) if len(fired) else None
            has_fp=bool((~fired.actual_fail_5000).any())
        sequence_results.append({'system':name,'sequence_id':sid,'first_alarm_raw_row_index':first,
                                 'first_alarm_time':first_time,
                                 'has_false_positive':has_fp})
comparison=pd.DataFrame(comparison)
if REPLAY:
    saved_test = pd.read_csv(OUT / 'test_results.csv')
    for name in ('P0','P1','P2','B1','B2','B3'):
        assert np.array_equal(test[f'{name}_alarm'].to_numpy(bool), saved_test[f'{name}_alarm'].to_numpy(bool))
else:
    test.to_csv(OUT / 'test_results.csv',index=False)
    comparison.to_csv(OUT / 'metric_comparison.csv',index=False)
print('Sequence results:')
display(pd.DataFrame(sequence_results))
display(comparison)
""")

md(r"""
## 결과 저장 및 해석 기준

**가정:** 사용량 비용은 API가 `usage.cost`로 숫자를 반환한 성공 응답에서만 합산한다. 캐시 응답의 과거 비용과 이번 실행의 새 호출 비용을 구분한다. PHM2018 비대칭 score 함수가 `fabjudge.huang2018`에 없으면 새로 만들지 않고 미적용이라고 기록한다. JEV와 기준선의 bootstrap 95% 구간이 겹치면 우열을 ‘구분 불가’로 쓴다.
""")

code(r"""
import fabjudge.huang2018 as huang2018
def usage_total(records, field):
    values=[]
    for record in records:
        usage=((record['result'].get('response') or {}).get('usage') or {})
        value=usage.get(field)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            return 'unavailable'
        values.append(float(value))
    return float(sum(values)) if values else 0.0
successful=[r for r in response_records if r['result']['response'] is not None]
new_successful=[r for r in successful if not r['result']['cache_hit']]
asymmetric=[name for name in dir(huang2018) if 'asym' in name.lower() and 'score' in name.lower()]
unique_cached_responses=len(list((OUT / 'cache').glob('*.json')))
summary={'design':'dev-tuned fault3 gate, one frozen test evaluation',
         'split_counts':split_counts,'dev_split':DEV_SPLIT,'dev_reason':DEV_REASON,
         'dev_alarm_rows':len(dev_alarms),'dev_TP':dev_tp,'dev_FP':dev_fp,
         'selected_features':SELECTED,'selected_correlation':selected_r,
         'sensor_enabled':SENSOR_ENABLED,'recipe_transition':recipe_status,
         'frozen_config_sha256':FROZEN_SHA256,'test_rows':len(test),'test_sequences':test.sequence_id.nunique(),
         'test_alarm_rows':int(test.lstm_alarm.sum()),'stability_score_sd_by_row':stability_sd.to_dict(),
         'stability_score_sd_mean':float(stability_sd.mean()),
         'new_api_calls':CLIENT.new_calls,'cache_hits':CLIENT.cache_hits,
         'unique_cached_api_responses':unique_cached_responses,
         'max_new_calls':MAX_NEW_CALLS,
         'api_usage':{'input_tokens':usage_total(successful,'input_tokens'),
                      'output_tokens':usage_total(successful,'output_tokens'),
                      'reported_cost_usd':usage_total(successful,'cost'),
                      'new_calls_reported_cost_usd':usage_total(new_successful,'cost')},
         'dev_gate_table':dev_gate_table.to_dict(orient='records'),
         'metrics':comparison.to_dict(orient='records'),'sequence_results':sequence_results,
         'phm2018_asymmetric_score':'미적용' if not asymmetric else 'function available: '+','.join(asymmetric),
         'limits':['test has only 8 sequences','dev and test are both from fault3 distribution',
                   'B3 dev fit and threshold use the same dev alarms']}
def json_safe(value):
    if isinstance(value,dict):
        return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value,np.generic):
        value=value.item()
    if isinstance(value,float) and not math.isfinite(value):
        return None
    return value
summary=json_safe(summary)
if REPLAY:
    summary=json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    assert summary['frozen_config_sha256']==FROZEN_SHA256
else:
    (OUT / 'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print({'new_api_calls':CLIENT.new_calls,'cache_hits':CLIENT.cache_hits,
       'api_usage':summary['api_usage'],
       'PHM2018_asymmetric_score':summary['phm2018_asymmetric_score']})
""")

md(r"""
## 결론

실행 결과를 확인한 뒤 이 셀에 수치 결론을 기록한다. P1→P2 센서 입력, P0→P1 질문 변경(상태 구성도 변경됨), P2→B1–B3 비교와 CI, 반복 score 표준편차의 임계값 근처 영향 순서로 해석한다. test는 8개 시퀀스이며 dev/test가 같은 fault3 분포라는 한계가 있다.
""")

if (ROOT / 'artifacts/06c_lstm_jev_semantic_gate/summary.json').exists():
    cells[-1].source = r"""## 결론

1. **P1 대비 P2:** 센서 의미를 더하자 23개 test 경보 중 판정 2건이 바뀌었다. P2는 TP 1건과 FP 1건을 더 유지했다. P1은 TP 7·FP 1(precision 0.875, recall 0.4375), P2는 TP 8·FP 2(precision 0.800, recall 0.500)였다. 센서 입력이 판단을 바꿨지만 FP를 더 줄이지는 못했다.
2. **P0 대비 P1:** 판정 2건이 바뀌었으나 둘 다 TP 7·FP 1로 집계됐다. 선택한 게이트 점수의 TP/FP AUC는 0.654→0.717(+0.063)이었다. P0와 P1은 질문과 센서 상태가 함께 다르므로 이 수치를 순수 프롬프트 효과로 분리할 수 없다.
3. **P2 대비 기준선:** P2는 FP 6건 제거·TP 7건 손실(precision 0.800, 95% 시퀀스 bootstrap CI 0.400–1.000). B1은 7·9(0.857, 0.333–1.000), B2는 0·0(0.652, 0.552–0.786), B3는 8·8(1.000, 1.000–1.000)이다. 각 precision CI가 P2와 겹치므로 우열은 **구분 불가**다. B2 중간점 규칙은 dev TP 27건을 잃어 제약을 넘었으므로 사전 규칙대로 no-op이었다.
4. **반복 안정성:** dev 5행의 P2 score 표준편차는 0.020–0.049(평균 0.033)였다. P2 임계값 1.27에서 test 경보 2건의 score가 최대 표준편차 0.049 안에 있고 가장 가까운 거리는 0.010이다. 이 경계 근처 판정은 반복 호출 변동에 민감할 수 있다. 이 반복 측정은 dev 행에서만 수행했다.

test는 8개 시퀀스뿐이고 dev/test 모두 fault3 분포다. JEV score와 센서 연관성은 판단 보조 근거이며 고장의 물리적 원인이나 독립적 재현성을 입증하지 않는다. 최초 고정 설정 SHA-256은 `b513f932be5ad82c1fa352468c469986ac0281b8153b74bcfcb5920d93878162`이다. 최초 실험은 API 465회·$0.013034028, 저장 오류 복구 시 부동소수점 캐시 키 차이로 추가 102회·$0.002912826가 발생했다. 총 567회·$0.015946854다."""

nb.metadata.kernelspec = {'display_name':'Python (.venv)','language':'python','name':'python3'}
nb.metadata.language_info = {'name':'python','version':'3.12'}
target=ROOT / 'notebooks/06c_lstm_jev_semantic_gate.ipynb'
nbf.write(nb,target)
print(target)
