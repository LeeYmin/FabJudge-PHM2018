"""Restore the first frozen configuration from cached responses and saved results.

This one-time recovery never calls the API or refits from held-out labels.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fabjudge.jev_gate import canonical, request_spec, infer_causal_sampled_rows
from fabjudge.huang2018_models import SequenceRUL

OUT=ROOT/'artifacts/06c_lstm_jev_semantic_gate'
old=set(p.stem for p in sorted((OUT/'cache').glob('*.json'),key=lambda p:p.stat().st_mtime)[:465])
all_cache=list((OUT/'cache').glob('*.json'))
print('first run cache records',len(old))
frozen=json.loads((OUT/'frozen_config.json').read_text(encoding='utf-8'))
torch.set_num_threads(2)
art=ROOT/'artifacts/huang2018'
seq=pd.read_csv(art/'sequence_metadata.csv')
inputs=json.loads((art/'reproduction_config.json').read_text(encoding='utf-8'))['model_input_columns']
rf=joblib.load(ROOT/'artifacts/models/huang2018/fault3_rf.joblib')
saved=torch.load(ROOT/'artifacts/models/huang2018/fault3_lstm.pt',map_location='cpu',weights_only=True)
lstm=SequenceRUL(**saved['architecture'])
lstm.load_state_dict(saved['state_dict'])
original_order=json.loads((art/'split_metadata.json').read_text(encoding='utf-8'))['splits']['fault3']['train_original_sequences']
dev,_=infer_causal_sampled_rows(ROOT,seq,original_order,rf,lstm,inputs)
dev['lstm_alarm']=dev.lstm_pred_seconds.le(5000)
question=json.loads((OUT/'questions/P0.json').read_text(encoding='utf-8'))
features=frozen['selected_features']
include_history=frozen['history_count_in_state']

def hash_for(row):
    spec=request_spec(row,question,features,include_history=include_history)
    return hashlib.sha256(canonical(spec).encode('utf-8')).hexdigest()

recovered={}
missing=[]
for index,row in dev.loc[dev.lstm_alarm].iterrows():
    p=float(row.rf_probability)
    if hash_for(row) in old:
        recovered[index]=p
        continue
    hits=[]
    for direction in (-math.inf,math.inf):
        candidate=p
        for distance in range(1,101):
            candidate=float(np.nextafter(candidate,direction))
            trial=row.copy()
            trial['rf_probability']=candidate
            if hash_for(trial) in old:
                hits.append((distance,candidate))
                break
    if hits:
        recovered[index]=sorted(hits)[0][1]
    else:
        missing.append(index)
print('recovered',len(recovered),'missing',missing)
print('changed RF probabilities',sum(recovered[i]!=float(dev.at[i,'rf_probability']) for i in recovered))
if missing or len(recovered)!=127:
    raise RuntimeError('Cannot reconstruct first dev requests')
for i,p in recovered.items():
    dev.at[i,'rf_probability']=p
dev['actual_fail_5000']=dev.wall_ttf_seconds.le(5000)
alarms=dev.loc[dev.lstm_alarm]
labels=alarms.actual_fail_5000.to_numpy(bool)

def fit_threshold(values, labels):
    values=np.asarray(values,dtype=float)
    labels=np.asarray(labels,dtype=bool)
    allowed=max(1.0,.1*int(labels.sum()))
    candidates=np.r_[np.nextafter(values.min(),-np.inf),np.unique(values)]
    options=[]
    for threshold in candidates:
        keep=values>=threshold
        lost_tp=int((labels & ~keep).sum())
        if lost_tp<=allowed:
            options.append({'threshold':float(threshold),'removed_fp':int((~labels & ~keep).sum()),
                            'lost_tp':lost_tp})
    return sorted(options,key=lambda x:(-x['removed_fp'],x['threshold']))[0]

gates={}
original_scores={}
for name in ('P0','P1','P2'):
    q=json.loads((OUT/f'questions/{name}.json').read_text(encoding='utf-8'))
    chosen=features if name=='P0' or name=='P2' else []
    columns={'score':[],'probability_2':[]}
    for index,row in alarms.iterrows():
        stem=hashlib.sha256(canonical(request_spec(row,q,chosen,include_history=include_history)).encode('utf-8')).hexdigest()
        if stem not in old:
            raise RuntimeError(f'First-run cache entry absent for {name}')
        answer=json.loads((OUT/'cache'/f'{stem}.json').read_text(encoding='utf-8'))['response']['answers']['alarm_evidence']
        columns['score'].append(answer['score'])
        columns['probability_2'].append(answer['probabilities']['2'])
        dev.at[index,f'{name}_score']=float(answer['score'])
        dev.at[index,f'{name}_probability_2']=float(answer['probabilities']['2'])
        dev.at[index,f'{name}_status']='success'
        dev.at[index,f'{name}_error']=None
    original_scores[name]=columns
    choices={}
    for field,values in columns.items():
        choice=fit_threshold(values,labels)
        choice.update({'auc':float(roc_auc_score(labels,values)),'field':field})
        choices[field]=choice
    winner=sorted(choices.values(),key=lambda x:(-x['removed_fp'],x['lost_tp'],
                                                0 if x['field']=='score' else 1))[0]
    gates[name]={'selected':winner,'candidates':choices}

b1=fit_threshold(alarms.rf_probability,labels)
b1['auc']=float(roc_auc_score(labels,alarms.rf_probability))
base_features=['rf_probability']+(features if frozen['sensor_enabled'] else [])
scaler=StandardScaler().fit(alarms[base_features])
model=LogisticRegression(C=1.0,max_iter=1000,random_state=42).fit(
    scaler.transform(alarms[base_features]),labels.astype(int))
b3_values=model.predict_proba(scaler.transform(alarms[base_features]))[:,1]
b3=fit_threshold(b3_values,labels)
b3['auc']=float(roc_auc_score(labels,b3_values))
b3['features']=base_features
b3['scaler_mean']=scaler.mean_.tolist()
b3['scaler_scale']=scaler.scale_.tolist()
b3['coef']=model.coef_[0].tolist()
b3['intercept']=float(model.intercept_[0])
frozen['gates']=gates
frozen['B1']=b1
frozen['B3']=b3
print('first gates',{k:v['selected'] for k,v in gates.items()})
print('first B1',b1,'first B3',{'threshold':b3['threshold'],'auc':b3['auc']})
payload=json.dumps(frozen,ensure_ascii=False,indent=2,allow_nan=False)
candidate=OUT/'frozen_config.recovered.json'
candidate.write_text(payload,encoding='utf-8')
digest=hashlib.sha256(candidate.read_bytes()).hexdigest()
print('candidate SHA256',digest)
print('original SHA256','b513f932be5ad82c1fa352468c469986ac0281b8153b74bcfcb5920d93878162')
if digest!='b513f932be5ad82c1fa352468c469986ac0281b8153b74bcfcb5920d93878162':
    raise RuntimeError('Original frozen config hash not recovered; no outputs overwritten')

# Restore exactly the configuration used before the held-out evaluation.
(OUT/'frozen_config.json').write_bytes(candidate.read_bytes())
dev['pipeline_alarm']=dev.rf_probability.ge(.5) & dev.lstm_alarm
dev.to_csv(OUT/'dev_rows.csv',index=False)

stability=[]
for index,row in alarms.head(5).iterrows():
    for repeat_index in range(3):
        stem=hashlib.sha256(canonical(request_spec(row,
            json.loads((OUT/'questions/P2.json').read_text(encoding='utf-8')),
            features,include_history=include_history,repeat_index=repeat_index)).encode('utf-8')).hexdigest()
        if stem not in old:
            raise RuntimeError('Original stability response absent')
        answer=json.loads((OUT/'cache'/f'{stem}.json').read_text(encoding='utf-8'))['response']['answers']['alarm_evidence']
        stability.append({'row_index':int(index),'sequence_id':row.sequence_id,
                          'raw_row_index':int(row.raw_row_index),'repeat_index':repeat_index,
                          'score':float(answer['score']),
                          'probability_2':float(answer['probabilities']['2']),
                          'status':'success','error':None})
stability=pd.DataFrame(stability)
stability.to_csv(OUT/'stability.csv',index=False)
print('restored original dev rows and stability responses')

test=pd.read_csv(OUT/'test_results.csv')
comparison=pd.read_csv(OUT/'metric_comparison.csv')
assert len(test)==80 and test.sequence_id.nunique()==8 and len(comparison)==8
assert hashlib.sha256((OUT/'frozen_config.json').read_bytes()).hexdigest()==digest
for name in ('P0','P1','P2'):
    gate=gates[name]['selected']
    p=np.where(test.lstm_alarm,test[f'{name}_{gate["field"]}']>=gate['threshold'],False)
    if not np.array_equal(p,test[f'{name}_alarm'].to_numpy(bool)):
        raise RuntimeError(f'Original {name} test result does not match recovered frozen config')
assert np.array_equal(np.where(test.lstm_alarm,test.rf_probability>=b1['threshold'],False),
                      test.B1_alarm.to_numpy(bool))
assert np.array_equal(np.where(test.lstm_alarm,test.B3_probability>=b3['threshold'],False),
                      test.B3_alarm.to_numpy(bool))

dev_rows=[]
for name in ('P0','P1','P2'):
    for field,choice in gates[name]['candidates'].items():
        dev_rows.append({'condition':name,'score_field':field,**choice,
                         'dev_TP_kept':79-choice['lost_tp'],'dev_FP_kept':48-choice['removed_fp']})
dev_rows += [
    {'condition':'B1','score_field':'rf_probability',**b1,'dev_TP_kept':79-b1['lost_tp'],'dev_FP_kept':48-b1['removed_fp']},
    {'condition':'B2','score_field':'sensor_rule','threshold':None,'auc':.5,
     'removed_fp':0,'lost_tp':0,'dev_TP_kept':79,'dev_FP_kept':48},
    {'condition':'B3','score_field':'logistic_probability','threshold':b3['threshold'],'auc':b3['auc'],
     'removed_fp':b3['removed_fp'],'lost_tp':b3['lost_tp'],'dev_TP_kept':79-b3['lost_tp'],
     'dev_FP_kept':48-b3['removed_fp']}]

sequence_results=[]
systems={'LSTM only':'lstm_alarm','RF to LSTM pipeline':'pipeline_alarm',
         **{name:f'{name}_alarm' for name in ('P0','P1','P2')},
         'B1':'B1_alarm','B2':'B2_alarm','B3':'B3_alarm'}
for name,column in systems.items():
    for sid,group in test.groupby('sequence_id',sort=True):
        fired=group.loc[group[column].astype(bool)].sort_values('raw_row_index')
        sequence_results.append({'system':name,'sequence_id':sid,
            'first_alarm_raw_row_index':int(fired.raw_row_index.iloc[0]) if len(fired) else None,
            'first_alarm_time':int(fired.time.iloc[0]) if len(fired) else None,
            'has_false_positive':bool((~fired.actual_fail_5000).any())})

first_records=[json.loads((OUT/'cache'/f'{stem}.json').read_text(encoding='utf-8'))['response']['usage'] for stem in old]
replay_records=[json.loads(p.read_text(encoding='utf-8'))['response']['usage'] for p in all_cache if p.stem not in old]
def totals(records):
    return {'responses':len(records),'input_tokens':sum(int(r['input_tokens']) for r in records),
            'output_tokens':sum(int(r['output_tokens']) for r in records),
            'cost_usd':float(sum(float(r['cost']) for r in records))}
sd=stability.groupby('row_index').score.std(ddof=1)
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
summary=json_safe({
    'design':'first frozen dev configuration; one held-out test evaluation',
    'split_counts':{'train':36,'validation':4,'test':8},
    'dev_split':'train','dev_reason':'validation has only 18 LSTM alarms (<20)',
    'dev_alarm_rows':127,'dev_TP':79,'dev_FP':48,'selected_features':features,
    'selected_correlation':frozen['selected_correlation'],
    'sensor_enabled':frozen['sensor_enabled'],'recipe_transition':'미확인',
    'frozen_config_sha256':digest,'test_rows':80,'test_sequences':8,'test_alarm_rows':23,
    'stability_score_sd_by_row':sd.to_dict(),'stability_score_sd_mean':float(sd.mean()),
    'experiment_new_api_calls':465,'replay_new_api_calls':len(replay_records),
    'total_new_api_calls':len(first_records)+len(replay_records),'max_experiment_new_calls':465,
    'api_usage':{'experiment':totals(first_records),'replay':totals(replay_records),
                 'total':totals(first_records+replay_records)},
    'dev_gate_table':dev_rows,'metrics':comparison.to_dict(orient='records'),
    'sequence_results':sequence_results,'phm2018_asymmetric_score':'미적용',
    'limits':['test has only 8 sequences','dev and test are both from fault3 distribution',
              'B3 dev fit and threshold use the same dev alarms',
              'RF inference differs in the last floating-point digit on some repeat runs; first-run state was restored from cache hashes']})
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print('test predictions match original frozen thresholds; summary saved')
print('usage',summary['api_usage'])
