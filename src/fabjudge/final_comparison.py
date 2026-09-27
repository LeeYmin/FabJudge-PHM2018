"""Read-only analysis of frozen 04b/06/07 artifacts; no inference or training.

07b uses its preregistered half-up rounding. Full-endpoint scenarios use
Python round as specified for notebook 08. These are deliberately separate.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

FAULTS = ('fault1', 'fault2', 'fault3')
SCOPES = {**{f: [f] for f in FAULTS}, 'fault1+fault2': ['fault1', 'fault2']}
RANDOM_SEED = 20260927
COLORS = {'04b': '#333333', 'LSTM': '#0072B2', 'JEV': '#009E73',
          'LLM': '#CC79A7', 'Random': '#999999', 'Logistic': '#E69F00',
          'RF': '#D55E00', 'E10': '#56B4E9', 'E20': '#009E73', 'E30': '#0072B2'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def confusion(y, alarm):
    y, alarm = np.asarray(y, bool), np.asarray(alarm, bool)
    tp, fp, fn, tn = (int(np.sum(v)) for v in
                      (y & alarm, ~y & alarm, y & ~alarm, ~y & ~alarm))
    return dict(TP=tp, FP=fp, FN=fn, TN=tn,
                recall=tp / (tp + fn) if tp + fn else np.nan,
                precision=tp / (tp + fp) if tp + fp else np.nan)


class FinalComparison:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.art = self.root / 'artifacts'
        self.b = self.art / '07b_rescue_lane_confirmatory'
        self.out = self.art / '08_final_comparison'
        self.tables, self.figures = self.out / 'tables', self.out / 'figures'
        self.tables.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)
        self.sources, self.checks, self.created = {}, [], []
        self.integrity_ok = self.full_ok = False
        self.unavailable = []
        plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
                             'font.size': 10, 'axes.titlesize': 12,
                             'axes.spines.top': False, 'axes.spines.right': False,
                             'figure.dpi': 110, 'savefig.dpi': 300})

    def source(self, path):
        p = Path(path)
        if not p.is_absolute():
            p = self.root / p
        assert p.is_file(), f'계산 불가: artifact 누락 {p}'
        assert p.is_relative_to(self.art) and not p.is_relative_to(self.out)
        self.sources[p] = sha256(p)
        return p

    def csv(self, path):
        return pd.read_csv(self.source(path))

    def js(self, path):
        return json.loads(self.source(path).read_text(encoding='utf-8'))

    def jsonl(self, path):
        return [json.loads(x) for x in self.source(path).read_text(encoding='utf-8').splitlines() if x.strip()]

    def check(self, name, saved, recomputed):
        assert np.isclose(saved, recomputed, atol=1e-9, rtol=0, equal_nan=True), (
            f'INTEGRITY FAILED: {name}: saved={saved}, recomputed={recomputed}')
        self.checks.append({'check': name, 'saved': saved, 'recomputed': recomputed, 'status': 'PASS'})

    def table(self, name, frame):
        p = self.tables / name
        frame.to_csv(p, index=False, encoding='utf-8-sig')
        if p not in self.created:
            self.created.append(p)
        return frame

    def figure(self, name, fig, caption):
        fig.text(.02, .015, caption, fontsize=9, va='bottom')
        fig.tight_layout(rect=(0, .12, 1, .94))
        p = self.figures / name
        fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
        self.created.append(p)
        return fig

    def inventory(self):
        required = [self.b / n for n in ['effect_vs_04b.csv', 'summary.json', 'hypothesis_tests.json',
                    'prereg_07_v2.json', 'prereg_07_v2_1.json', 'exploratory.csv', 'cost_curve.csv',
                    'tier_split.csv', 'eval_labels.csv', 'eval_sample.csv', 'rankings.csv',
                    'jev_R2_scores.csv', 'llm_noRF_results.jsonl', 'api_attempt_log.jsonl']]
        required += [self.art / 'huang2018' / f'{f}_test_predictions.parquet' for f in FAULTS]
        required += [self.art / 'huang2018' / n for n in ['split_metadata.json', 'sequence_metadata.csv', 'metrics.json']]
        required += [self.art / '06a_lstm_jev/larger_test_01' / n for n in ['pilot_rows.csv', 'sample_plan.csv', 'summary.json']]
        discovered = [p for folder in self.art.iterdir() if folder.name.startswith(('06', '07')) or folder.name == 'huang2018'
                      for p in folder.rglob('*') if p.is_file() and len(p.relative_to(folder).parts) <= 2
                      and not any('cache' in x or x in {'failed_raw','interrupted_v2'} for x in p.parts)
                      and p.suffix in {'.csv', '.json', '.jsonl', '.parquet', '.md'}]
        frame = pd.DataFrame([{'source_path': p.relative_to(self.root).as_posix(),
                               'required': p in required, 'exists': p.exists(),
                               'bytes': p.stat().st_size if p.exists() else None}
                              for p in sorted(set(required + discovered))])
        self.table('artifact_inventory.csv', frame)
        assert frame.loc[frame.required, 'exists'].all(), '필수 artifact 누락: inventory 확인'
        return frame

    def load_and_verify_07b(self):
        self.summary = self.js(self.b / 'summary.json')
        self.hyp = self.js(self.b / 'hypothesis_tests.json')
        self.prereg = self.js(self.b / 'prereg_07_v2_1.json')
        for name, field in [('prereg_07_v2.json', 'prereg_v2_bytes_sha256'),
                            ('prereg_07_v2_1.json', 'prereg_v2_1_bytes_sha256'),
                            ('rankings.csv', 'rankings_sha256')]:
            assert sha256(self.source(self.b / name)) == self.summary[field], f'Frozen hash mismatch: {name}'
        assert self.hyp == self.summary['hypothesis_tests']
        assert self.prereg['hypotheses']['order'] == ['H1', 'H2', 'H3']
        self.effect = self.csv(self.b / 'effect_vs_04b.csv')
        self.cost = self.csv(self.b / 'cost_curve.csv')
        self.expl = self.csv(self.b / 'exploratory.csv')
        tier, labels = self.csv(self.b / 'tier_split.csv'), self.csv(self.b / 'eval_labels.csv')
        sample = self.csv(self.b / 'eval_sample.csv')
        keys = ['row_key', 'fault', 'sequence_id', 'original_row_position']
        assert tier.row_key.is_unique and labels.row_key.is_unique and sample.row_key.is_unique
        assert set(tier.row_key) == set(labels.row_key) == set(sample.row_key)
        self.ev = tier.merge(labels, on=keys, validate='one_to_one')
        self.ev = self.ev.merge(sample[['row_key', 'sequence_group']], on='row_key', validate='one_to_one')
        assert len(self.ev) == 720
        assert np.array_equal(self.ev.label_near_5k, self.ev.wall_ttf_seconds.le(5000))
        rule = self.ev.lstm_pred_seconds.le(5000) & self.ev.rf_probability.ge(.5)
        assert np.array_equal(rule, self.ev['04b_decision'].eq('pass'))
        assert np.array_equal(rule, self.ev.tier.eq('tier1'))
        assert np.array_equal(self.ev.lstm_alarm, self.ev.lstm_pred_seconds.le(5000))
        assert self.ev.lstm_alarm.all()
        self.check('07b 720행 04b rule 일치', 720, int((rule == self.ev['04b_decision'].eq('pass')).sum()))
        self.rank = self.csv(self.b / 'rankings.csv')
        self.rescue = self.ev.loc[self.ev.tier.eq('rescue')].copy()
        self.orders = {(method, fault): sub.sort_values('rank').row_key.tolist()
                       for (method, fault), sub in self.rank.groupby(['method', 'fault'], sort=True)}
        for (method, fault), order in self.orders.items():
            assert len(order) == len(set(order))
            assert set(order) == set(self.rescue.loc[self.rescue.fault.eq(fault), 'row_key'])
        self.y = self.ev.set_index('row_key').label_near_5k.to_dict()
        for fault, sub in self.ev.groupby('fault'):
            for saved_name, value in [('sampled_rows', len(sub)), ('tier1_rows', sub.tier.eq('tier1').sum()),
                    ('rescue_rows', sub.tier.eq('rescue').sum()), ('positive_rows', sub.label_near_5k.sum())]:
                self.check(f'{fault}/{saved_name}', self.summary['sample']['per_fault'][fault][saved_name], value)
        for row in self.effect.to_dict('records'):
            es = self.ev[self.ev.fault.isin(SCOPES[row['scope']])]
            selected = self.top('JEV_routing', SCOPES[row['scope']], row['E'] / 100)
            tp = sum(self.y[k] for k in selected)
            base_tp = es.loc[es.tier.eq('tier1'), 'label_near_5k'].sum()
            pos = es.label_near_5k.sum()
            rescue_pos = es.loc[es.tier.eq('rescue'), 'label_near_5k'].sum()
            values = dict(additional_review_rows=len(selected), rescue_positive_rows=rescue_pos,
                          imminent_failures_recovered=tp, rescue_recovery_rate=tp/rescue_pos,
                          rescue_precision=tp/len(selected), tier1_positive_rows=base_tp,
                          total_sampled_positive_rows=pos, **{'04b_recall': base_tp/pos},
                          structural_recall=(base_tp+tp)/pos, recall_change_vs_04b=tp/pos)
            for key, value in values.items():
                self.check(f"effect/{row['scope']}/E{row['E']}/{key}", row[key], value)
        for saved in self.summary['effect_vs_04b']:
            row = self.effect.loc[(self.effect.scope == saved['scope']) & (self.effect.E == saved['E'])].iloc[0]
            for key in saved.keys() - {'scope'}:
                self.check(f'summary effect/{saved["E"]}/{key}', saved[key], row[key])
        # Reconstruct confusion from frozen selected keys; cost counts independently cross-check it.
        conf_rows = []
        for scope, faults in SCOPES.items():
            es = self.ev[self.ev.fault.isin(faults)]
            base_alarm = es.tier.eq('tier1')
            for e in [0, 10, 20, 30]:
                selected = self.top('JEV_routing', faults, e/100) if e else []
                alarm = base_alarm | es.row_key.isin(selected)
                cm = confusion(es.label_near_5k, alarm)
                strategy = f'04b_plus_structural_E{e}' if e else '04b_only'
                conf_rows.append(dict(scope=scope, method=strategy, **cm))
                costs = self.cost[(self.cost.scope == scope) & (self.cost.strategy == strategy)]
                assert len(costs) == 20
                for r in costs.to_dict('records'):
                    for field, calc in [('review_count', int(alarm.sum())), ('missed_positive_count', cm['FN']),
                         ('review_cost', int(alarm.sum())), ('missed_failure_cost', cm['FN']*r['missed_failure_cost_r']),
                         ('total_cost', int(alarm.sum())+cm['FN']*r['missed_failure_cost_r']),
                         ('incremental_reviews_vs_04b', len(selected)),
                         ('incremental_recovered_positives_vs_04b', sum(self.y[k] for k in selected))]:
                        self.check(f'cost/{scope}/{strategy}/r{r["missed_failure_cost_r"]}/{field}', r[field], calc)
                    if e:
                        found = sum(self.y[k] for k in selected)
                        self.check(f'break_even/{scope}/{e}', r['break_even_r_vs_04b'], len(selected)/found if found else np.nan)
        self.table('07b_confusion_verified.csv', pd.DataFrame(conf_rows))
        # All exploratory precision values are checked against saved rank keys.
        for row in self.expl[self.expl.E.notna()].to_dict('records'):
            fs, frac = SCOPES[row['scope']], row['E']/100
            for method, field in [(row['method'], 'precision'), (row['comparator'], 'comparator_precision')]:
                self.check(f'expl/{row["scope"]}/{method}/{row["E"]}', row[field], self.precision(method, fs, frac))
            self.check('expl difference', row['difference'], row['precision']-row['comparator_precision'])
        primary = self.rescue[self.rescue.fault.isin(FAULTS[:2])]
        estimates = [primary.label_near_5k.mean(), self.precision('full_LLM', FAULTS[:2], .2),
                     np.mean([self.precision(f'random_{i:04d}', FAULTS[:2], .2) for i in range(1, 1001)])]
        jev_p = self.precision('JEV_routing', FAULTS[:2], .2)
        for i, h in enumerate(['H1', 'H2', 'H3']):
            self.check(f'{h}/estimate', self.hyp[h]['estimate'], jev_p)
            self.check(f'{h}/comparator', self.hyp[h]['comparator_estimate'], estimates[i])
            self.check(f'{h}/difference', self.hyp[h]['difference'], jev_p-estimates[i])
            assert self.hyp[h]['testing_order'] == i+1
        # Reproduce H1's original 10,000 sequence draws exactly; H2/H3 remain untested.
        rng = np.random.default_rng(self.hyp['bootstrap_seed'])
        prepared = {}
        for fault in FAULTS[:2]:
            sub = primary[primary.fault.eq(fault)].set_index('row_key')
            groups = sorted(sub.sequence_group.unique())
            order = self.orders['JEV_routing', fault]
            prepared[fault] = (groups, sub.loc[order, 'sequence_group'].tolist(), sub.loc[order, 'label_near_5k'].to_numpy())
        draws = []
        for _ in range(self.hyp['bootstrap_repetitions']):
            total_y = total_n = found = reviewed = 0
            for fault in FAULTS[:2]:
                groups, group_order, y = prepared[fault]
                mult = Counter(rng.choice(np.asarray(groups, dtype=object), size=len(groups), replace=True).tolist())
                expanded = np.repeat(y, [mult.get(g, 0) for g in group_order])
                k = math.floor(len(expanded)*.2+.5)
                total_y += expanded.sum(); total_n += len(expanded)
                found += expanded[:k].sum(); reviewed += k
            draws.append(found/reviewed-total_y/total_n)
        lower = np.quantile(draws, .05)
        self.check('H1 one-sided 95% lower', self.hyp['H1']['one_sided_lower_bound'], lower)
        assert self.hyp['H1']['decision'] == ('PASS' if lower > 0 else 'FAIL_TO_REJECT')
        assert self.hyp['H1']['decision'] == 'FAIL_TO_REJECT'
        for h in ['H2', 'H3']:
            assert self.hyp[h]['decision'] == 'NOT_TESTED_DUE_TO_GATEKEEPING'
            assert self.hyp[h]['one_sided_lower_bound'] is None and self.hyp[h]['bootstrap_repetitions'] == 0
        self.jev = self.csv(self.b / 'jev_R2_scores.csv')
        llm = self.jsonl(self.b / 'llm_noRF_results.jsonl')
        self.llm = pd.DataFrame([dict(row_key=r['row_key'], fault=r['fault'], status=r['status'],
                       priority_score=(r.get('output') or {}).get('priority_score'),
                       cost_usd_reported=r.get('cost_usd_reported')) for r in llm])
        assert self.jev.row_key.is_unique and self.llm.row_key.is_unique
        assert set(self.jev.row_key) == set(self.llm.row_key) == set(self.rescue.row_key)
        self.check('MODEL_UNAVAILABLE', self.summary['main_MODEL_UNAVAILABLE'], self.llm.status.eq('MODEL_UNAVAILABLE').sum())
        calls = self.summary['cost_and_calls']
        self.attempt = pd.DataFrame(self.jsonl(self.b / 'api_attempt_log.jsonl'))
        api = self.attempt.loc[~self.attempt.cache_hit]
        for stage, field in [('jev', 'JEV'), ('llm_main', 'main_LLM'), ('llm_stability', 'stability_LLM')]:
            a = api[api.call_stage.eq(stage)]
            self.check(f'actual API calls/{stage}', calls['actual_API_requests'][field], len(a))
            self.check(f'actual API cost/{stage}', calls['07b_API_cost_by_stage_reported_usd'][stage], a.cost_usd.sum())
        self.check('actual total calls', calls['actual_API_requests']['total'], len(api))
        self.check('actual total cost', calls['07b_API_cost_reported_usd'], api.cost_usd.sum())
        self.check('retries', calls['retry_count'], (api.groupby(['row_key','call_stage','repeat_tag'], dropna=False).size()-1).clip(lower=0).sum())
        self.check('full LLM equivalent calls', calls['projected_full_LLM_equivalent_calls'], len(self.rescue))
        mid_n = len(self.rank[(self.rank.method == 'JEV_routing') & (self.rank.queue_tier == 'MID')])
        self.check('routed equivalent calls', calls['JEV_routing_LLM_equivalent_calls'], mid_n)
        self.check('LLM reduction percent (equivalent)', calls['LLM_call_reduction_pct'], 100*(1-mid_n/len(self.rescue)))
        self.integrity_ok = True
        return self.table('07b_integrity_checks.csv', pd.DataFrame(self.checks))

    def top(self, method, faults, fraction):
        return [k for f in faults for k in self.orders[method, f][:math.floor(len(self.orders[method, f])*fraction+.5)]]

    def precision(self, method, faults, fraction):
        selected = self.top(method, faults, fraction)
        return sum(self.y[k] for k in selected)/len(selected)

    def comparison_tables(self):
        assert self.integrity_ok
        rows = []
        for scope in ['fault1+fault2', 'fault3']:
            es = self.ev[self.ev.fault.isin(SCOPES[scope])]
            effects = self.effect[self.effect.scope.eq(scope)].set_index('E')
            pos = int(es.label_near_5k.sum()); base_tp = int(es.loc[es.tier.eq('tier1'), 'label_near_5k'].sum())
            # Per-fault actual experiment costs are attributable by row_key, not by E scenario.
            actual = self.attempt.loc[~self.attempt.cache_hit & self.attempt.row_key.isin(es.row_key)]
            for e in [0, 10, 20, 30]:
                r = effects.loc[e] if e else None
                rows.append({'fault_group': scope.replace('+fault2', '+2'), 'method': f'04b + 구조 레인 E{e}' if e else '04b',
                  '전체 임박 고장 수': pos, '회수 TP': base_tp+int(r.imminent_failures_recovered) if e else base_tp,
                  '놓친 고장 FN': pos-base_tp-int(r.imminent_failures_recovered) if e else pos-base_tp,
                  'recall': r.structural_recall if e else base_tp/pos,
                  '추가 검토 건수': int(r.additional_review_rows) if e else 0,
                  'rescue precision': r.rescue_precision if e else np.nan,
                  'LLM 호출 수': int(actual.call_stage.isin(['llm_main','llm_stability']).sum()) if e else 0,
                  '실제 API 비용': float(actual.cost_usd.sum()) if e else 0.,
                  '구분': '확증(H1) 표본 기술통계' if scope=='fault1+fault2' and e==20 else '탐색/기술통계',
                  '호출·비용 범위': '07b fault group 실제 실험 총량(재시도·안정성 포함); E별 공통, 합산 금지' if e else '추가 API 없음',
                  'source_path': 'artifacts/07b_rescue_lane_confirmatory/effect_vs_04b.csv; summary.json; api_attempt_log.jsonl'})
        self.t1 = self.table('T1_04b_vs_final_pipeline.csv', pd.DataFrame(rows))
        rows = []
        for h in ['H1','H2','H3']:
            v = self.hyp[h]
            rows.append({'가설': h, '추정치': v['estimate'], '비교값': v['comparator_estimate'], '차이': v['difference'],
                         '단측 95% 하한': v['one_sided_lower_bound'], '판정': v['decision'], '검정 순서': v['testing_order']})
        self.t2 = self.table('T2_confirmatory_hypothesis_tests.csv', pd.DataFrame(rows))
        return self.t1, self.t2

    def verify_full_endpoints(self):
        assert self.integrity_ok
        meta = self.csv(self.art/'huang2018/sequence_metadata.csv').set_index('sequence_id')
        splits = self.js(self.art/'huang2018/split_metadata.json')['splits']
        self.full, coverage, schemas = {}, [], []
        for fault in FAULTS:
            path = self.source(self.art/'huang2018'/f'{fault}_test_predictions.parquet')
            d = pd.read_parquet(path)
            schemas.append({'fault': fault, 'endpoint 수':len(d), 'sequence 수':d.sequence_id.nunique(), '열 목록':json.dumps(d.columns.tolist())})
            assert set(d.sequence_id) == set(splits[fault]['test_original_sequences']), f'{fault}: test sequence 누락'
            assert not d.duplicated(['sequence_id','raw_row_index']).any()
            for sid, sub in d.groupby('sequence_id'):
                m = meta.loc[sid]
                with np.load(self.source(self.root/str(m.sequence_path))) as z:
                    raw_index, raw_time = z['raw_row_index'], z['time']
                assert len(raw_index) == int(m.final_length)
                positions = np.unique(np.r_[np.arange(0, len(raw_index), 15), len(raw_index)-1]).astype(int)
                sub = sub.sort_values('raw_row_index')
                assert np.array_equal(sub.raw_row_index, raw_index[positions]), f'{fault}/{sid}: causal endpoint 누락/subset'
                assert np.array_equal(sub.time, raw_time[positions])
                assert np.array_equal(sub.wall_ttf_seconds, int(m.fault_time)-raw_time[positions])
                coverage.append({'fault':fault, 'sequence_id':sid, 'source_rows':len(raw_index),
                                 'expected_endpoints':len(positions),'stored_endpoints':len(sub), 'status':'PASS'})
            assert np.isfinite(d[['wall_ttf_seconds','lstm_pred_seconds','rf_probability']].to_numpy()).all()
            d['fault'] = fault
            d['y'] = d.wall_ttf_seconds.le(5000)
            d['lstm_alarm'] = d.lstm_pred_seconds.le(5000)
            d['alarm_04b'] = d.lstm_alarm & d.rf_probability.ge(.5)
            d['rescue'] = d.lstm_alarm & ~d.alarm_04b
            self.full[fault] = d
            req = self.ev[self.ev.fault.eq(fault)]
            joined = req.merge(d, left_on=['sequence_id','original_row_position'], right_on=['sequence_id','raw_row_index'],
                               how='left', validate='one_to_one', suffixes=('_07b','_04b'))
            assert len(joined) == len(req) and joined.raw_row_index.notna().all()
            for field in ['wall_ttf_seconds','lstm_pred_seconds','rf_probability']:
                assert np.allclose(joined[field+'_07b'], joined[field+'_04b'], atol=1e-9, rtol=0)
            assert np.array_equal(joined['04b_decision'].eq('pass'), joined.alarm_04b)
        self.schema = self.table('full_endpoint_schema.csv', pd.DataFrame(schemas))
        self.coverage = self.table('full_endpoint_coverage.csv', pd.DataFrame(coverage))
        plan = self.csv(self.art/'06a_lstm_jev/larger_test_01/sample_plan.csv')
        pilot = self.csv(self.art/'06a_lstm_jev/larger_test_01/pilot_rows.csv')
        keys = ['sequence_id','raw_row_index']
        assert len(plan) == len(pilot) == 80
        assert set(map(tuple, plan[keys].to_numpy())) == set(map(tuple, pilot[keys].to_numpy()))
        exact = plan.merge(self.full['fault3'], on=keys, how='left', validate='one_to_one')
        assert len(exact) == 80 and exact.y.notna().all()
        cm = confusion(exact.y, exact.alarm_04b)
        saved = self.js(self.art/'06a_lstm_jev/larger_test_01/summary.json')
        saved_cm = next(x for x in saved['metrics'] if x['system']=='RF to LSTM pipeline')
        for k,v in {'TP':9,'FP':4,'FN':7,'TN':60}.items():
            assert cm[k] == v
            self.check(f'06a exact80/{k}', saved_cm[k], cm[k])
            self.check(f'07b saved exact80/{k}', self.summary['06a_exact_80_row_confusion'][k], cm[k])
        for k in ['recall','precision']:
            self.check(f'06a exact80/{k}', saved_cm[k], cm[k])
        self.exact80 = exact
        self.full_ok = True
        return self.schema, self.coverage, cm

    def history_tables(self):
        folders = {'06a':'06a_lstm_jev/larger_test_01', '06b':'06b_lstm_jev_sensor_skip',
                   '06c':'06c_lstm_jev_semantic_gate', '06d':'06d_lstm_jev_oof_edge',
                   '06e':'06e_fix_and_weakness_diagnosis', '07a':'07a_pipeline_skeleton'}
        paths = {k:self.art/v for k,v in folders.items()}
        a, b, c = (self.js(paths[k]/'summary.json') for k in ['06a','06b','06c'])
        fix = self.csv(paths['06e']/'dev_metric_comparison.csv').set_index('candidate')
        stats = self.js(paths['07a']/'dev_sample_statistics.json')
        prereg_e = self.js(paths['06e']/'prereg_06e.json')
        erratum = self.source(paths['06d']/'ERRATUM.md').read_text(encoding='utf-8')
        conclusion = self.source(paths['06e']/'conclusion.md').read_text(encoding='utf-8')
        assert a['candidates']['conservative']['removed_false_positives'] == 0
        assert b['jev_removed_alarm_rows'] == 0
        assert stats['1']['near_failure_fraction'] == stats['2']['near_failure_fraction'] == 0
        assert 'in-sample' in erratum and 'sequence first 20000' in prereg_e['payload']['history']
        def rel(p): return p.relative_to(self.root).as_posix()
        p2 = next(v for v in c['metrics'] if v['system']=='P2')
        rows = [
          ['06a','초기 hybrid의 보수적 경보 제거 가능성','제거 경보 0건','보수적 JEV routing에서 경보 제거 없음','확인',rel(paths['06a']/'summary.json')],
          ['06b','후속 prompt/sensor routing으로 FP 제거 가능한가','제거 경보 0건','탐색: 경보 제거 없음','확인',rel(paths['06b']/'summary.json')],
          ['06c','dev threshold가 test에서도 유지되는가',f"P2 test recall={p2['recall']:.4f}; LSTM=0.9375",
           '탐색: train dev의 in-sample 성격, test recall 급락으로 일반화 실패','확인',rel(paths['06c']/'summary.json')+'; notebooks/06c_lstm_jev_semantic_gate.ipynb'],
          ['06d','OOF/edge 결합에서 독립 신호가 있는가',f"J2 dev AUC={fix.loc['J2','original_dev_auc']:.6f} (무효)",
           'dev P2 요청에 in-sample RF 사용: 영향받은 J2/S2 dev 결과와 적합 설정 무효, 06e 정정','정정됨',rel(paths['06d']/'ERRATUM.md')],
          ['06e','leakage 정정 및 긴 window 검증',f"정정 J2 dev AUC={fix.loc['J2','corrected_dev_auc']:.6f}",
           '탐색: RF 누수 정정; 긴 window는 클래스별 시퀀스 근거 부족으로 검증 불가','확인',rel(paths['06e']/'dev_metric_comparison.csv')+'; '+rel(paths['06e']/'conclusion.md')],
          ['07a','infrastructure / pipeline sanity check','fault1/2 positive=0건; 표본 상한 600, 시퀀스당 1행',
           '성능 확증 아님; 표본 상한 문제에 따라 prereg v2에서 percentile 표본 설계 수정','확인',rel(paths['07a']/'dev_sample_statistics.json')+'; artifacts/07b_rescue_lane_confirmatory/prereg_07_v2.json'],
          ['07b','최종 사전등록 확증',f"H1={self.hyp['H1']['decision']}; H2/H3={self.hyp['H2']['decision']}",
           'H1 실패; H2/H3 gatekeeping 미검정. JEV/LLM superiority 근거 없음','확인','artifacts/07b_rescue_lane_confirmatory/hypothesis_tests.json']]
        self.t3 = self.table('T3_experiment_history.csv', pd.DataFrame(rows, columns=['노트북','질문','핵심 수치','결론','상태','근거 파일 경로']))
        # Early warning is a label diagnostic, never a TP relabeling.
        test = self.csv(paths['06c']/'test_results.csv')
        early = test.loc[test.lstm_alarm & ~test.actual_fail_5000, 'wall_ttf_seconds']
        early_n = int(early.between(5500,8300).sum())
        early_text = f'기존 공통 80행의 LSTM FP {len(early)}건 중 {early_n}건이 TTF 5,500~8,300초; 범위 {early.min():.0f}~{early.max():.0f}초'
        rows = [
          ['06d','dev P2 요청의 in-sample RF leakage','J2/S2 dev 구분력과 적합 설정 오염','ERRATUM 및 06e 저장 비교','영향받은 결과 무효; 06e 정정',rel(paths['06d']/'ERRATUM.md')],
          ['06c','in-sample 성격 dev에서 고른 threshold의 test 붕괴','낙관적 dev 성능은 일반화 증거가 아님','P2 test recall 0.5000, LSTM 0.9375','규칙 변경 없이 일반화 실패 기록',rel(paths['06c']/'summary.json')],
          ['06a~06d','5,000초 label과 early-warning의 차이','5k 밖 경보는 공식 FP이나 곧 고장날 수 있음',early_text,'공식 TP로 재분류하지 않음; label 유지',rel(paths['06c']/'test_results.csv')],
          ['07a','표본 상한으로 fault1/2 positive 0건','판별력을 label 기준으로 확증할 수 없음','dev_sample_statistics의 두 fault 양성률 0','prereg v2에서 표본 설계 수정',rel(paths['07a']/'dev_sample_statistics.json')],
          ['07b','고정 길이·failure-anchored sequence의 sample position 누수','미래 fault와 기계적으로 연동된 위치를 배포 feature로 오인 가능','F5 위치별 양성률 및 길이 분포','사용 불가 / leakage diagnostic only','artifacts/07b_rescue_lane_confirmatory/tier_split.csv; artifacts/huang2018/sequence_metadata.csv'],
          ['06e 설계','시퀀스 시작 기반 baseline의 위치 누수 가능성','failure-anchored 시작점은 온라인의 외생적 시작점이 아님','prereg: first 20000 seconds baseline, elapsed >=30000','가능성 진단; 성능 feature로 제안하지 않음',rel(paths['06e']/'prereg_06e.json')]]
        self.t4 = self.table('T4_pitfalls.csv', pd.DataFrame(rows, columns=['단계','함정','왜 문제인가','어떻게 확인했는가','후속 처리','근거']))
        # Fixed historical candidates, not retrospective best-of selection.
        specs = [('06b test',paths['06b']/'jev_results.csv','jev_score','rf_probability'),
                 ('06c dev',paths['06c']/'dev_rows.csv','P2_score','rf_probability'),
                 ('06c test',paths['06c']/'test_results.csv','P2_score','rf_probability'),
                 ('06d dev',paths['06d']/'dev_rows.csv','P2_score','rf_probability_oof'),
                 ('06e dev',paths['06e']/'dev_rows_06d_fix.csv','P2_score','rf_probability_oof')]
        aucs = []
        for stage,path,jcol,rcol in specs:
            df = self.csv(path)
            sub = df.loc[df.lstm_alarm & df[jcol].notna() & df[rcol].notna()]
            assert sub.actual_fail_5000.nunique() == 2
            for method,col in [('JEV',jcol),('RF',rcol)]:
                value = roc_auc_score(sub.actual_fail_5000, sub[col])
                if stage in ['06d dev','06e dev'] and method=='JEV':
                    key = 'original_dev_auc' if stage=='06d dev' else 'corrected_dev_auc'
                    self.check(stage+' J2 AUC', fix.loc['J2',key], value)
                aucs.append({'stage':stage,'method':method,'auc':value,'n':len(sub),'positive':int(sub.actual_fail_5000.sum()),
                             'signal':col,'status':'버그로 무효' if stage=='06d dev' and method=='JEV' else '탐색', 'source_path':rel(path)})
        self.history_auc = self.table('F6_auc_source_data.csv', pd.DataFrame(aucs))
        return self.t3, self.t4

    def sample_figures(self):
        assert self.integrity_ok and self.full_ok
        figs = []
        ef = self.effect[self.effect.scope.eq('fault1+fault2')].sort_values('E')
        base = float(ef['04b_recall'].iloc[0]); pos = int(ef.total_sampled_positive_rows.iloc[0])
        fig,ax = plt.subplots(figsize=(8.5,5.5))
        ax.scatter([0],[base],color=COLORS['04b'],s=70,label='04b',zorder=3)
        ax.plot(ef.additional_review_rows,ef.structural_recall,color=COLORS['JEV'],label='04b + 구조 레인')
        for r in ef.itertuples():
            ax.scatter(r.additional_review_rows,r.structural_recall,color=COLORS[f'E{r.E}'],marker='o' if r.E==20 else 'D',s=65)
            ax.annotate(f'E{r.E}',(r.additional_review_rows,r.structural_recall),xytext=(4,8),textcoords='offset points')
        grid = np.arange(0, .351, .01)
        expected = []
        for e in grid:
            k = found = 0
            for f in FAULTS[:2]:
                sub = self.rescue[self.rescue.fault.eq(f)]; n = math.floor(len(sub)*e+.5)
                k += n; found += n*sub.label_near_5k.mean()
            expected.append((k,base+found/pos))
        ax.plot(*np.array(expected).T,'--',color=COLORS['Random'],label='무작위 순서 기대값 (탐색)')
        ax.set(xlabel='04b 대비 추가 검토 건수',ylabel='Recall',ylim=(.67,.85))
        ax.legend(loc='upper left'); ax.grid(alpha=.2)
        fig.suptitle('F1. 추가 검토량 대비 recall — fault1+2, 07b 확증 표본')
        figs.append(self.figure('F1_review_volume_vs_recall.png',fig,
             '확증 H1: E20 선택 효율. E10/E30·recall 변화·무작위 기대선은 탐색/기술통계.\n추가 검토에 따라 recall 증가; 정렬의 무작위 대비 우월성은 확보하지 못함.'))
        fig,ax = plt.subplots(figsize=(9,5.5))
        x = self.expl[(self.expl.scope=='fault1+fault2') & (self.expl.E==20)]
        logistic = x[(x.method=='full_LLM') & (x.comparator=='logistic')].iloc[0].comparator_precision
        lstm = x[(x.method=='full_LLM') & (x.comparator=='LSTM_only')].iloc[0].comparator_precision
        vals = [self.hyp['H1']['estimate'],self.hyp['H2']['comparator_estimate'],self.hyp['H3']['comparator_estimate'],lstm,logistic]
        names = ['JEV 라우팅\n확증(H1)','전체 LLM\n탐색','무작위 라우팅 평균\n탐색','LSTM 순\n탐색','로지스틱\n탐색']
        bars = ax.bar(names, vals, color=[COLORS[k] for k in ['JEV','LLM','Random','LSTM','Logistic']],edgecolor='black',width=.62)
        for bar in bars[1:]: bar.set_hatch('//')
        ax.bar_label(bars,labels=[f'{v:.3f}' for v in vals],padding=3)
        ax.axhline(self.hyp['H1']['comparator_estimate'],color=COLORS['04b'],ls='--',label=f"rescue 기저율 {self.hyp['H1']['comparator_estimate']:.3f}")
        ax.set(ylabel='rescue precision@E20',ylim=(0,max(vals)*1.3)); ax.legend()
        fig.suptitle('F2. rescue precision@E20 비교 — 확증 H1 및 탐색 분석')
        figs.append(self.figure('F2_rescue_precision_E20.png',fig,'fault1+2; H1 실패. H2/H3는 gatekeeping 미검정. 빗금 막대는 탐색 비교.\nLSTM 순 = LSTM_only (LLM 미사용); 무작위 라우팅 = 저장된 1,000회 평균.'))
        fig,ax = plt.subplots(figsize=(8.5,5.5))
        for strategy, label, color in [('04b_only','04b',COLORS['04b']),('04b_plus_structural_E20','04b + 구조 레인 E20',COLORS['E20'])]:
            sub = self.cost[(self.cost.scope=='fault1+fault2') & (self.cost.strategy==strategy)].sort_values('missed_failure_cost_r')
            ax.plot(sub.missed_failure_cost_r,sub.total_cost,label=label,color=color,lw=2)
        be = sub.break_even_r_vs_04b.iloc[0]
        if np.isfinite(be) and 1<=be<=20:
            ax.axvline(be,color=COLORS['Random'],ls=':'); ax.annotate(f'손익분기 r={be:.2f}',(be,sub.total_cost.min()),xytext=(6,16),textcoords='offset points')
        ax.set(xlabel='놓친 고장 비용 r',ylabel='총비용 (검토 1건 비용 = 1)',xlim=(1,20));ax.legend();ax.grid(alpha=.2)
        fig.suptitle('F3. 놓친 고장 비용에 따른 총비용 비교 — 탐색')
        figs.append(self.figure('F3_cost_curve.png',fig,'fault1+2 07b 표본; 총비용 = 검토 건수 + FN × r. 실제 API 비용은 포함하지 않음.\n출처: 07b cost_curve.csv. 손익분기는 이 비용 가정에 한정.'))
        # Explicit identity joins. No positional assignment of predictions to labels.
        rs = self.rescue[self.rescue.fault.isin(FAULTS[:2])]
        joined = rs.merge(self.jev[['row_key','fault','score']],on=['row_key','fault'],how='left',validate='one_to_one')
        joined = joined.merge(self.llm[['row_key','fault','status','priority_score']],on=['row_key','fault'],how='left',validate='one_to_one')
        assert len(joined)==len(rs) and joined.status.notna().all()
        excluded = int(joined.status.eq('MODEL_UNAVAILABLE').sum())
        joined['negative_lstm'] = -joined.lstm_pred_seconds
        signal_rows = []
        for label,col in [('JEV R2 score','score'),('LLM priority_score','priority_score'),('-LSTM predicted RUL','negative_lstm'),('RF probability','rf_probability')]:
            sub = joined[joined.status.ne('MODEL_UNAVAILABLE')] if col=='priority_score' else joined
            assert sub[col].notna().all() and sub.label_near_5k.nunique()==2
            signal_rows.append({'signal':label,'auc':roc_auc_score(sub.label_near_5k,sub[col]),'n':len(sub),
                                'excluded_MODEL_UNAVAILABLE':excluded if col=='priority_score' else 0})
        self.signal_auc = self.table('F4_auc_source_data.csv',pd.DataFrame(signal_rows))
        fig,ax = plt.subplots(figsize=(9,5.5))
        bars = ax.bar(self.signal_auc.signal,self.signal_auc.auc,color=[COLORS[k] for k in ['JEV','LLM','LSTM','RF']],hatch='//',edgecolor='white')
        ax.bar_label(bars,fmt='%.3f',padding=3);ax.axhline(.5,color=COLORS['04b'],ls='--');ax.set(ylabel='ROC AUC',ylim=(0,1))
        fig.suptitle('F4. rescue 내부 각 신호의 구분력 — 탐색')
        figs.append(self.figure('F4_rescue_signal_auc.png',fig,f'fault1+2 rescue; endpoint key로 일대일 join. LLM MODEL_UNAVAILABLE {excluded}건 제외.\nLLM n={len(joined)-excluded}, 다른 신호 n={len(joined)}; 표본 차이 주의.'))
        fig,axes = plt.subplots(1,2,figsize=(12,5.7))
        position_rows, length_rows = [], []
        for i,f in enumerate(FAULTS):
            es = self.ev[self.ev.fault.eq(f)]
            agg = es.groupby('sampled_percentile').label_near_5k.agg(['mean','size']).reset_index()
            axes[0].plot(agg.sampled_percentile,agg['mean'],marker=['o','s','^'][i],label=f)
            position_rows.extend(agg.assign(fault=f).to_dict('records'))
            d = self.full[f]
            lengths = d.groupby('sequence_id').lstm_alarm.sum()
            length_rows.extend({'fault':f,'sequence_id':sid,'alarm_endpoints':n,'all_endpoints':int((d.sequence_id==sid).sum())} for sid,n in lengths.items())
            axes[1].scatter(np.full(len(lengths),i)+np.linspace(-.17,.17,len(lengths)),lengths,alpha=.65,s=23)
        self.table('F5_position_rates.csv',pd.DataFrame(position_rows))
        self.table('F5_alarm_lengths.csv',pd.DataFrame(length_rows))
        axes[0].set(xlabel='경보 구간 내 sample percentile',ylabel='임박 고장 비율',xlim=(5,95),ylim=(-.03,1.03),title='A. 07b percentile 표본');axes[0].legend()
        axes[1].set(xticks=range(3),xticklabels=FAULTS,ylabel='경보 구간 길이 (causal endpoint 행 수)',title='B. 전체 test sequence (무경보 0행 포함)')
        fig.suptitle('F5. 사용 불가 신호: 고정 길이 시퀀스 구조로 인한 위치 누수 — 탐색',fontsize=13)
        figs.append(self.figure('F5_position_leakage.png',fig,'사용 불가 / leakage diagnostic only. 실제 길이는 절단·gap·경보 유무에 따라 다름.\nfailure-anchored 표본 위치의 label 연동을 진단하며, 위치를 성능 향상 feature로 제안하지 않음.'))
        fig,ax = plt.subplots(figsize=(10,5.7));stages=self.history_auc.stage.drop_duplicates().tolist()
        for j,method in enumerate(['JEV','RF']):
            sub = self.history_auc[self.history_auc.method.eq(method)].set_index('stage').loc[stages]
            bars=ax.bar(np.arange(len(stages))+(j-.5)*.34,sub.auc,width=.34,color=COLORS[method],label=method)
            ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
            if method=='JEV':bars[3].set_hatch('xxx');bars[3].set_edgecolor('black');bars[3].set_linewidth(2)
        ax.axhline(.5,ls='--',color=COLORS['Random']);ax.set(xticks=range(len(stages)),xticklabels=stages,ylabel='ROC AUC',ylim=(0,1.08))
        ax.annotate('버그로 무효\n(J2)',xy=(3-.17,.85),xytext=(2.8,1.0),ha='center',arrowprops={'arrowstyle':'->'});ax.legend(loc='upper left')
        fig.suptitle('F6. JEV 및 RF 판별력 변화 — 06 시리즈 (탐색)')
        figs.append(self.figure('F6_jev_auc_history.png',fig,'고정 후보: 06b sensor prompt; 06c P2 score; 06d/e J2. LSTM 경보 내 동일 유효행에서 비교.\n06d/e RF는 OOF; 06c dev는 in-sample 성격. 표본·prompt 차이로 단계 간 성능 추세로 해석하지 않음.'))
        return figs

    def full_endpoint_tables(self, repetitions=2000):
        assert self.full_ok and repetitions == 2000
        # Fault-stratified sequence bootstrap; counts of each fault's sequences are fixed.
        # A random permutation of the full replicate rescue pool is uniform sampling
        # without replacement at each E and couples E points without changing marginals.
        grid = np.arange(0, 101, 5)/100
        scopes = list(SCOPES)
        point, clusters = {}, {}
        for f,d in self.full.items():
            pos=int(d.y.sum()); tp=int((d.y & d.alarm_04b).sum()); n=int(d.alarm_04b.sum())
            ry=d.loc[d.rescue,'y'].to_numpy(dtype=int)
            reviews=np.array([round(len(ry)*float(e)) for e in grid])
            found=reviews*ry.mean() if len(ry) else np.zeros(len(grid))
            point[f]={'positive':pos,'tp':tp+found,'review':n+reviews,'additional':reviews,'base_tp':tp,
                      'lstm_tp':int((d.y & d.lstm_alarm).sum()),'lstm_reviews':int(d.lstm_alarm.sum())}
            clusters[f]=[(int(s.y.sum()),int((s.y & s.alarm_04b).sum()),
                          s.loc[s.rescue,'y'].to_numpy(dtype=int)) for _,s in d.groupby('sequence_id',sort=True)]
        point['fault1+fault2']={k:point['fault1'][k]+point['fault2'][k] for k in point['fault1']}
        recalls=np.empty((repetitions,len(scopes),len(grid)))
        deltas=np.empty_like(recalls)
        rng=np.random.default_rng(RANDOM_SEED)
        zero_denominators=Counter()
        for rep in range(repetitions):
            draws={}
            for f in FAULTS:
                seqs=clusters[f]
                chosen=rng.choice(len(seqs),size=len(seqs),replace=True)
                pos=sum(seqs[i][0] for i in chosen); tp=sum(seqs[i][1] for i in chosen)
                rescue=np.concatenate([seqs[i][2] for i in chosen])
                # This actually selects endpoint copies, including repeated copies of a sampled sequence.
                randomized=rescue[rng.permutation(len(rescue))]
                cumulative=np.r_[0,np.cumsum(randomized)]
                ks=np.array([round(len(rescue)*float(e)) for e in grid],dtype=int)
                draws[f]=(pos,tp+cumulative[ks],tp)
            draws['fault1+fault2']=tuple(draws['fault1'][i]+draws['fault2'][i] for i in range(3))
            for j,s in enumerate(scopes):
                pos,tp,base=draws[s]
                if pos==0:
                    zero_denominators[s]+=1
                    recalls[rep,j]=np.nan; deltas[rep,j]=np.nan
                else:
                    recalls[rep,j]=tp/pos;deltas[rep,j]=(tp-base)/pos
        assert not zero_denominators, f'양성 0 bootstrap replicate 발생: {zero_denominators}; CI 계산 중단'
        ci=np.quantile(recalls,[.025,.975],axis=0)
        dci=np.quantile(deltas,[.025,.975],axis=0)
        rows,curve=[] ,[]
        for j,s in enumerate(scopes):
            p=point[s]
            assert p['positive']>0
            assert np.isclose(p['tp'][-1],p['lstm_tp'],atol=1e-9,rtol=0)
            assert p['review'][-1]==p['lstm_reviews']
            assert np.all(np.diff(p['additional'])>=0) and np.all(np.diff(p['tp'])>=0)
            for idx,e in enumerate(grid):
                curve.append({'fault_group':s.replace('+fault2','+2'),'E':int(round(e*100)),
                              'additional_reviews':int(p['additional'][idx]),'recall':p['tp'][idx]/p['positive'],
                              'ci_low':ci[0,j,idx],'ci_high':ci[1,j,idx]})
            for method,idx in [('LSTM 단독',20),('04b',0),('04b + rescue 무작위 순서 E10',2),
                               ('04b + rescue 무작위 순서 E20',4),('04b + rescue 무작위 순서 E30',6),
                               ('04b + rescue 전부 검토',20)]:
                tp=p['tp'][idx];n=p['review'][idx];pos=p['positive']
                rows.append({'fault_group':s.replace('+fault2','+2'),'method':method,'전체 임박 고장 수':pos,
                             'TP':tp,'FN':pos-tp,'recall':tp/pos,
                             'recall 95% CI low':ci[0,j,idx],'recall 95% CI high':ci[1,j,idx],
                             '경보(검토) 건수':int(n),'precision':tp/n if n else np.nan,
                             '04b 대비 추가 검토 건수':int(p['additional'][idx]),'04b 대비 recall 변화':(tp-p['base_tp'])/pos,
                             'recall 변화 95% CI low':dci[0,j,idx],'recall 변화 95% CI high':dci[1,j,idx],
                             '구분':'탐색: 실제 JEV/LLM 결과 아님' if '무작위' in method else '탐색: 저장 endpoint 기술통계'})
        self.t5=self.table('T5_full_endpoint_comparison.csv',pd.DataFrame(rows))
        self.full_curve=self.table('F7_full_endpoint_curve.csv',pd.DataFrame(curve))
        events=[]
        for f,d in self.full.items():
            for sid,s in d.groupby('sequence_id',sort=True):
                event={'fault':f,'sequence_id':sid,'positive_endpoints':int(s.y.sum())}
                for method,col in [('LSTM','lstm_alarm'),('04b','alarm_04b')]:
                    hit=s.loc[s.y & s[col], 'wall_ttf_seconds']
                    event[method+'_detected']=len(hit)>0
                    event[method+'_first_lead_seconds']=float(hit.max()) if len(hit) else np.nan
                event['04b_missed_LSTM_detected']=event['LSTM_detected'] and not event['04b_detected']
                events.append(event)
        self.events=self.table('T6_sequence_event_details.csv',pd.DataFrame(events))
        rows=[]
        for scope,fs in SCOPES.items():
            es=self.events[self.events.fault.isin(fs)]
            for method,label in [('LSTM','LSTM 단독'),('04b','04b')]:
                times=es.loc[es[method+'_detected'], method+'_first_lead_seconds']
                rows.append({'fault_group':scope.replace('+fault2','+2'),'method':label,'시퀀스 수':len(es),
                  '탐지 시퀀스 수':int(es[method+'_detected'].sum()),'놓친 시퀀스 수':int((~es[method+'_detected']).sum()),
                  '첫 경보의 고장 전 시간 중앙값':times.median(),'첫 경보의 고장 전 시간 Q1':times.quantile(.25),
                  '첫 경보의 고장 전 시간 Q3':times.quantile(.75),
                  '04b가 놓치고 LSTM은 잡은 시퀀스 수':int(es['04b_missed_LSTM_detected'].sum()),
                  '5k 양성 endpoint 없는 시퀀스 수':int(es.positive_endpoints.eq(0).sum()),'구분':'탐색'})
        self.t6=self.table('T6_sequence_event_detection.csv',pd.DataFrame(rows))
        assert (self.events['LSTM_detected'] | ~self.events['04b_detected']).all()
        return self.t5,self.t6

    def original_metrics(self):
        path=self.art/'huang2018/metrics.json'
        data=self.js(path);rows=[]
        for f in FAULTS:
            branches=[('rf',data['rf'][f], '5k fault vs >50k normal; IPW precision/F1; boundary 제외'),
                      ('lstm/test_intrinsic',data['lstm'][f]['test_intrinsic'],'failure-anchored sample-count RUL seconds'),
                      ('combined',data['combined'][f],'causal RF gate 출력; alarm<=5k 추가 조건 없는 기존 지표')]
            for prefix,metrics,target in branches:
                for metric,value in metrics.items():
                    if isinstance(value,(int,float)) and not isinstance(value,bool) and any(x in metric for x in ['rmse','mae','smape','r2','precision','recall','f1','auc','fraction','lead_seconds','false_alarm']):
                        pointer=f'{prefix.split("/")[0]}/{f}/'+('test_intrinsic/' if prefix.startswith('lstm') else '')+metric
                        rows.append({'fault':f,'metric':prefix+'/'+metric,'value':value,'split':'test','target':target,
                                     'source_path':path.relative_to(self.root).as_posix()+'#/'+pointer,'구분':'기존 저장값 (재계산 없음)'})
        rows.append({'fault':'all','metric':'04b R²','value':np.nan,'split':'test','target':'04b RUL',
                     'source_path':path.relative_to(self.root).as_posix(),'구분':'저장된 지표 없음 (old_04 R²는 04b가 아님)'})
        self.t5b=self.table('T5b_04b_original_metrics.csv',pd.DataFrame(rows))
        return self.t5b

    def full_figure(self):
        assert self.full_ok
        fig,axes=plt.subplots(1,2,figsize=(12,5.8))
        a=self.full_curve[self.full_curve.fault_group.eq('fault1+2')]
        axes[0].plot(a.additional_reviews,a.recall,color=COLORS['Random'],label='무작위 rescue 선택 기대값')
        axes[0].fill_between(a.additional_reviews,a.ci_low,a.ci_high,color=COLORS['Random'],alpha=.23,label='95% bootstrap band')
        for idx,method,label in [(0,'04b','04b'),(-1,'LSTM','LSTM / rescue 전부')]:
            r=a.iloc[idx];axes[0].scatter(r.additional_reviews,r.recall,color=COLORS[method],s=65,label=label,zorder=3)
        axes[0].set(title='A. 전체 endpoint — 탐색',xlabel='04b 대비 추가 검토 건수',ylabel='recall',ylim=(.5,1.03));axes[0].legend(fontsize=9)
        b=self.effect[self.effect.scope.eq('fault1+fault2')].sort_values('E')
        axes[1].plot(b.additional_review_rows,b.structural_recall,color=COLORS['JEV'],alpha=.7)
        for r in b.itertuples():
            axes[1].scatter(r.additional_review_rows,r.structural_recall,color=COLORS[f'E{r.E}'],marker='o' if r.E==20 else 'D',s=65)
            axes[1].annotate(f'JEV E{r.E}',(r.additional_review_rows,r.structural_recall),xytext=(0,9),textcoords='offset points',ha='center')
        axes[1].set(title='B. 07b 표본 결과 — 확증 표본의 기술통계',xlabel='04b 대비 추가 검토 건수',ylabel='recall',xlim=(10,90),ylim=(.68,.85))
        for ax in axes:ax.grid(alpha=.2)
        fig.suptitle('F7. 전체 데이터: 추가 검토량 대비 recall — fault1+2 (탐색)',fontsize=13)
        return self.figure('F7_full_data_review_vs_recall.png',fig,
          'A: fault별 기대값 합산; 2,000회 fault 내 sequence 재표집 + 실제 rescue 무작위 선택.\nB: 07b 저장 관측값; E10/E30은 탐색, H1 E20 선택 효율 확증 실패. 두 모집단을 같은 축에 합치지 않음.')

    def finish(self):
        from PIL import Image
        assert self.integrity_ok and self.full_ok
        for f,sub in self.t5.groupby('fault_group'):
            t=sub.set_index('method')
            assert t.loc['LSTM 단독','recall'] >= t.loc['04b','recall']
            assert t.loc['04b + rescue 전부 검토','recall'] == t.loc['LSTM 단독','recall']
            rnd=t.loc[[f'04b + rescue 무작위 순서 E{e}' for e in [10,20,30]]]
            assert np.all(np.diff(rnd['04b 대비 추가 검토 건수'])>=0)
            assert np.all(np.diff(rnd.recall)>=0)
        expected_tables=['T1_04b_vs_final_pipeline.csv','T2_confirmatory_hypothesis_tests.csv','T3_experiment_history.csv',
                         'T4_pitfalls.csv','T5_full_endpoint_comparison.csv','T5b_04b_original_metrics.csv','T6_sequence_event_detection.csv']
        expected_figures=['F1_review_volume_vs_recall.png','F2_rescue_precision_E20.png','F3_cost_curve.png',
                          'F4_rescue_signal_auc.png','F5_position_leakage.png','F6_jev_auc_history.png','F7_full_data_review_vs_recall.png']
        for n in expected_tables:assert self.tables.joinpath(n).is_file()
        dimensions=[]
        for n in expected_figures:
            p=self.figures/n
            with Image.open(p) as im:
                assert all(abs(x-300)<.1 for x in im.info['dpi'])
                dimensions.append({'file':n,'width_px':im.width,'height_px':im.height,'dpi':im.info['dpi'][0]})
        self.table('figure_validation.csv',pd.DataFrame(dimensions))
        self.table('07b_integrity_checks.csv',pd.DataFrame(self.checks))
        for p,before in self.sources.items():
            assert sha256(p)==before, f'원본 artifact 변경됨: {p}'
        self.table('source_manifest.csv',pd.DataFrame([{'source_path':p.relative_to(self.root).as_posix(),'sha256':h}
                                                      for p,h in sorted(self.sources.items())]))
        validation=pd.DataFrame({'check':['07b 저장값 재계산 일치 (atol=1e-9, rtol=0)',
          '07b 720행 04b rule 100% 일치','06a fault3 기존 80행 TP9 FP4 FN7 TN60',
          '전체 causal test endpoint key 완전성','LSTM recall >= 04b recall',
          'rescue 전부 검토 recall == LSTM recall','E10/E20/E30 검토량 및 기대 recall 단조성',
          '모든 figure 300 dpi','T1~T6 및 T5b, F1~F7 생성','입력 artifact SHA-256 불변'], 'status':'PASS'})
        self.table('final_consistency_checks.csv',validation)
        all_outputs=sorted(list(self.tables.glob('*.csv'))+list(self.figures.glob('*.png')))
        manifest=pd.DataFrame([{'file':p.relative_to(self.root).as_posix(),'sha256':sha256(p)} for p in all_outputs])
        manifest.to_csv(self.out/'manifest.csv',index=False,encoding='utf-8-sig')
        assert all(sha256(self.root/r.file)==r.sha256 for r in manifest.itertuples())
        t=self.t5[self.t5.fault_group.eq('fault1+2')].set_index('method')
        event_n=int(self.t6[(self.t6.fault_group=='fault1+2') & (self.t6.method=='04b')]['04b가 놓치고 LSTM은 잡은 시퀀스 수'].iloc[0])
        e20=self.effect[(self.effect.scope=='fault1+fault2') & (self.effect.E==20)].iloc[0]
        calls=self.summary['cost_and_calls']
        lines=[f"1. 07b fault1+2 구조 레인 E20: recall {e20['04b_recall']:.4f} → {e20.structural_recall:.4f} (+{100*e20.recall_change_vs_04b:.2f}%p), 추가 검토 {int(e20.additional_review_rows)}건.",
               f"2. H1 확증 판정 {self.hyp['H1']['decision']}; H2/H3는 gatekeeping으로 미검정. JEV/LLM 정렬 우월성 근거를 확보하지 못함.",
               f"3. JEV 라우팅 환산 LLM 호출 {calls['projected_full_LLM_equivalent_calls']} → {calls['JEV_routing_LLM_equivalent_calls']} ({calls['LLM_call_reduction_pct']:.2f}% 감소); 실제 07b는 전체 rescue 호출, LLM 요청 {calls['actual_API_requests']['main_LLM']}+안정성 {calls['actual_API_requests']['stability_LLM']}회, 총 API 비용 ${calls['07b_API_cost_reported_usd']:.8f}.",
               '4. 주요 함정: 06d dev RF 누수(06e 정정), 06c in-sample threshold 일반화 실패, 07a 양성 0건, 고정 길이 위치 누수; 5k 밖 early-warning은 공식 FP 유지.',
               '5. 한계: 07b percentile 표본과 전체 endpoint는 별도 모집단; fault3 재사용, 소수·동일 장비 시퀀스, 선택 편향 및 모델 일반화 한계. 전체 rescue 기대값은 탐색 시나리오이며 실제 JEV/LLM 결과가 아님.']
        for name in ['04b','LSTM 단독','04b + rescue 무작위 순서 E20']:
            r=t.loc[name];lines.append(f"전체 fault1+2 {name}: recall={r.recall:.6f}, 95% CI [{r['recall 95% CI low']:.6f}, {r['recall 95% CI high']:.6f}]")
        r=t.loc['04b + rescue 무작위 순서 E20']
        lines += [f"E20 추가 검토={int(r['04b 대비 추가 검토 건수'])}건; recall 변화={r['04b 대비 recall 변화']:.6f}, 95% CI [{r['recall 변화 95% CI low']:.6f}, {r['recall 변화 95% CI high']:.6f}]",
                  f'04b가 놓치고 LSTM은 잡은 사건={event_n}개 sequence.',
                  '계산 불가: 04b R² 저장값 없음. 요구하지 않은 T7은 명세에 정의되지 않아 생성 대상에서 제외.']
        summary='\n'.join(lines)
        (self.out/'summary_ko.txt').write_text(summary,encoding='utf-8')
        return summary,validation,manifest
