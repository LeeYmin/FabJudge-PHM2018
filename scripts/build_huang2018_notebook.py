"""Build the paper-specific reproduction notebook from reviewable cells."""

from pathlib import Path
import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "04b_huang2018_reproduction.ipynb"
cells = []


def md(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source))


md("""# 04b · Huang et al. (2018) reproduction

**Purpose.** Diagnose the previous `04_ttf_baseline` and reproduce the two-stage method on raw PHM 2018 training files. This is a separate paper-specific branch; the existing FabJudge 02/03/04 notebooks and artifacts are untouched.

Primary source: [Huang et al., *Remaining Useful Life Estimation for Systems with Abrupt Failures*, PHM Society 2018](https://papers.phmsociety.org/index.php/phmconf/article/download/590/phmc_18_590), DOI [10.36001/phmconf.2018.v10i1.590](https://doi.org/10.36001/phmconf.2018.v10i1.590). Relevant paper sections: 3, 4.1–4.3, 5.2–5.5.

The raw source data are read only. Every numeric result below is computed from the saved raw audit, extracted sensor data, or fitted models. Paper-reported values are labeled separately.""")

md("""## 1. Reproduction specification

| Component | Previous 04, verified in repository | Huang et al. 2018 | This reproduction |
|---|---|---|---|
| Degradation classifier | `ExtraTreesClassifier` on compact engineered windows | Random Forest | `RandomForestClassifier` on raw 21 inputs + pressure/flow residual |
| Fault threshold | Training TTF 25th percentile, hundreds of thousands of seconds | TTF ≤ 5,000 s | TTF ≤ 5,000 s |
| Normal threshold | Complement of the 25th-percentile near-failure label | TTF > 50,000 s | TTF > 50,000 s |
| Boundary samples | Included in complement | Excluded from RF training | 5,000 < TTF ≤ 50,000 s excluded; evaluated separately |
| Imbalance | Balanced class weights | Fault ×1,000, then equal-sized random normal sample | Weighted, sampled approximation; exact count and deviation reported |
| RUL estimator | `ExtraTreesRegressor` | Two LSTM layers + two 8-node dense layers | PyTorch sequence-to-sequence LSTM, separate per fault |
| Sequence representation | 100-measurement/stride-50 FabJudge windows | Raw run-to-fault, maxL=300, N=15, SampleR=15, Mmin=3000 | Same named parameters; 1-hour gap cutoff is an assumption |
| Metric | Full-validation MAE/RMSE/R², numerical output for every row | Degrading-sequence RMSE/SMAPE; normal → no RUL | Intrinsic LSTM and combined gate evaluated separately |

**PAPER-SPECIFIED:** RF thresholds, ×1,000, Mmin=3000, maxL=300, N=15, SampleR=15, two LSTM layers, dense 8/8, MSE, RMSprop, dropout, L2, validation early stop, 80/20 original-sequence split, and 90/10 train/validation split. The first two LSTM widths are 32/64, doubled to 64/128 for fault modes 1 and 2.

**REPRODUCTION-ASSUMPTION:** Numerical large-gap cutoff is 3,600 seconds, since the paper does not give one and no authors' implementation with a cutoff was located. The residual relation estimator is a one-input RF regressor, since the paper specifies `q − f̂(p)` but no fitting algorithm. RF storage sampling and weighted class balance conserve memory and are not identical to bootstrap on literally duplicated rows. Exact original paper split seed is unavailable.

**IMPLEMENTATION-DETAIL:** Fixed seed 42, CPU PyTorch, target divided by 4,515 during optimization then multiplied back, 3 repeated trainings for each fault in this compute-limited run, and 4 seconds per raw measurement for seconds conversion. The raw TTF itself is already in seconds. Input sensor measurements are not additionally scaled.""")

md("""## 2. Why 04 was low

The old 04 notebook labels the earliest 25% of **window-level** TTF values as near failure. Its cutoffs are about 515,398 / 1,016,521 / 2,466,814 seconds for low/high/leak, far wider than the paper's 5,000-second fault region. It trains an ExtraTrees regressor only on that set, assigns a fixed threshold-valued numerical prediction to normal rows, then scores the entire validation TTF range. This creates the millions-of-seconds errors and negative full-range R². Raw TTF has been verified as **seconds**, so an erroneous time-unit conversion is not the cause. These are different estimands and evaluation domains; the RMSE values below must not be interpreted as a like-for-like leaderboard.""")

code("""import sys, json, random, platform
from pathlib import Path
ROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent
sys.path[:0] = [str(ROOT / '.venv' / 'Lib' / 'site-packages'), str(ROOT / 'src')]
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from IPython.display import display, Image, Markdown
from fabjudge.huang2018 import (audit_raw_labels, discover_schema, extract_raw_inputs, FAULT_NAMES, SENSOR_COLUMNS, OPERATING_COLUMNS, SHUTTER_CATEGORIES, INPUT_COLUMNS, sampled_raw_indices)
from fabjudge.huang2018_models import (FAULT_SLUGS, PAPER_METRICS, make_original_splits, sequence_generator_unit_test, summarize_original, load_summaries, SequenceRUL, train_evaluate_rf, train_repeated_lstm, combined_pipeline_metrics, plot_test_curves)
ART = ROOT / 'artifacts' / 'huang2018'
MODEL_DIR = ROOT / 'artifacts' / 'models' / 'huang2018'
ART.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
random.seed(42); np.random.seed(42); torch.manual_seed(42)
print('Python', platform.python_version(), 'PyTorch', torch.__version__, 'device=cpu', 'seed=42')""")

md("""## 3–4. Raw schema and RUL unit audit

The audit streams all 20 raw TTF files. Counts, limits and bands are exact; displayed quantiles use a seeded 1% row sample. For every finite label it verifies `TTF = next fault timestamp − measurement timestamp`. This establishes seconds as the unit without inferring from magnitude.""")
code("""audit_path = ART / 'raw_label_audit.json'
audit = json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.exists() else audit_raw_labels(ROOT, ART)
schema = audit['schema']
print('Tools:', schema['sensor_file_count'], 'raw rows:', sum(f['rows'] for f in audit['file_inventory']))
print('Sensor column count:', len(schema['sensor_columns']))
print('Sensor columns:', schema['sensor_columns'])
print('Dtypes from first 1,000 rows:', schema['sensor_dtypes_sample'])
print('Time/equipment/run columns:', [c for c in schema['sensor_columns'] if c in ('time','Tool','Lot','runnum','recipe','recipe_step','stage')])
print('Raw TTF unit:', audit['raw_ttf_unit'], '; seconds conversion factor:', audit['seconds_conversion_factor'])
print('Fault → TTF columns:', schema['fault_to_ttf'])
display(pd.DataFrame(audit['target_stats']).T[['count','min','p01','p05','p25','median','p75','p95','p99','max','le_5000','between_5000_50000','gt_50000','exact_timestamp_matches','timestamp_mismatches']])""")

md("""## 5–8. Failure events, gap truncation, samples to fault, Mmin

Failure events are deduplicated by source file, fault name and timestamp. A run-to-fault interval starts after the previous same-mode event. A gap over one hour truncates history at the most recent gap. `Mmin > 3000` is then applied. Exact timestamps are often absent from measurements; the last observed measurement before the fault is the sequence endpoint, and its sample-based RUL is zero. This proxy and the event-to-last-observation lag are recorded.""")
code("""print('Timestamp gap distribution:', audit['gap_summary'])
extract_path = ART / 'raw_input_extraction.json'
extraction = json.loads(extract_path.read_text(encoding='utf-8')) if extract_path.exists() else extract_raw_inputs(ROOT, ART)
events = pd.read_csv(ART / 'raw_failure_events.csv').drop_duplicates('sequence_id')
sequences = pd.read_csv(ART / 'sequence_metadata.csv')
summary_rows = []
for name, slug in zip(FAULT_NAMES, FAULT_SLUGS):
    e = events.loc[events.fault_name.eq(name)]
    s = sequences.loc[sequences.fault_name.eq(name)]
    valid = int(s.status.eq('valid').sum())
    summary_rows.append({'fault':slug, 'raw_event_rows':int(audit['schema']['fault_name_counts'][name]), 'unique_events':len(e), 'gap_truncated_below_Mmin':int(((e.base_length > 3000) & (e.length_after_gap_3600 <= 3000)).sum()), 'removed_Mmin_before_gap':int((e.base_length <= 3000).sum()), 'candidate_after_gap_Mmin':int((e.length_after_gap_3600 > 3000).sum()), 'removed_after_missing':int(s.status.eq('removed_after_missing_or_gap').sum()), 'valid_sequences':valid, 'paper_sequences':PAPER_METRICS[slug]['paper_train_sequences']+PAPER_METRICS[slug]['paper_test_sequences'], 'difference':valid-(PAPER_METRICS[slug]['paper_train_sequences']+PAPER_METRICS[slug]['paper_test_sequences'])})
sequence_count_table = pd.DataFrame(summary_rows)
display(sequence_count_table)
print('Missing removal:', extraction['missing'])
print('Column missing counts:', extraction['column_missing'])
print('Observed shutter values:', extraction['shutter_counts'])
print('Any candidate retruncated after missing deletion:', extraction['missing'].get('sequences_retruncated_after_missing',0))""")

md("""## 9. Paper input reconstruction and raw distribution

The 21 inputs are exactly 12 sensor readings, 4 operating-time variables, and five fixture-shutter indicators (0, 1, 2, 3, 255). There is no FabJudge tool-wise normalization or compact-feature selection.""")
code("""print('12 sensor columns:', SENSOR_COLUMNS)
print('4 operating-time columns:', OPERATING_COLUMNS)
print('5 shutter one-hot categories:', SHUTTER_CATEGORIES)
print('Final input dimension:', len(INPUT_COLUMNS))
assert len(INPUT_COLUMNS) == 21
display(pd.DataFrame(extraction['input_distribution_0p02pct_sample']).T)""")

md("""## 10–11. RF labels, counts and leakage-safe original-sequence split

The exact complete-row band counts are from all measurements. The model stores a seeded small fraction per band to bound RAM. The paper's literal ×1,000 duplication and equal-size normal draw are represented by balanced sample weights on the stored rows. This matches balanced expected class risk, but RF bootstrap draws are not identical to literal duplication; it is a reproduction deviation. The pressure/flow residual uses `FLOWCOOLPRESSURE` and `FLOWCOOLFLOWRATE`, whose schema names identify the physical variables. Its regression is fitted only on training normal rows.""")
code("""display(pd.DataFrame(extraction['rf_exact_band_counts_complete_rows']).T)
display(pd.DataFrame(extraction['rf_stored_band_counts']).T)
print('RF inclusion rates:', extraction['rf_storage_rates'])
splits = make_original_splits(sequences, ART / 'split_metadata.json', seed=42)
for slug, split in splits.items():
    assert set(split['train_original_sequences']).isdisjoint(split['test_original_sequences'])
    assert set(split['train_original_sequences']).isdisjoint(split['validation_original_sequences'])
    print(slug, split['counts'])""")

md("""## 12–14. Sequence generator, unit test and sample alignment

Offset 0 ends at the last measurement; offsets 1–14 move the endpoint one source row earlier. Each summary walks backward by 15 source rows, retains up to 300 time steps, then reverses into chronological order. Short history is left padded and excluded from loss and metrics. We split original failure sequences **before** constructing the 15 summaries.""")
code("""unit_test = sequence_generator_unit_test()
print('Synthetic indexing test:', unit_test)
first_valid = sequences.loc[sequences.status.eq('valid')].iloc[0]
first_summary = summarize_original(ROOT / first_valid.sequence_path)
print('First real sequence:', first_valid.sequence_id)
print('Raw length after cleaning:', first_valid.final_length, '; generated shape:', first_summary['x'].shape)
print('First summary selected source indices, first/last 10:', first_summary['raw_index'][0][first_summary['mask'][0]][:10], first_summary['raw_index'][0][first_summary['mask'][0]][-10:])
assert first_summary['x'].shape == (15,300,21)
assert np.isfinite(first_summary['x']).all()
assert np.all(first_summary['y'][first_summary['mask']] >= 0)
fig, axes = plt.subplots(1, 2, figsize=(11,3))
for ax, fault_name in zip(axes, FAULT_NAMES[:2]):
    example = sequences.loc[sequences.fault_name.eq(fault_name) & sequences.status.eq('valid')].iloc[0]
    summary = summarize_original(ROOT / example.sequence_path)
    m = summary['mask'][0]
    ax.plot(summary['y'][0,m], summary['x'][0,m,SENSOR_COLUMNS.index('FLOWCOOLPRESSURE')])
    ax.set_title(fault_name)
    ax.set_xlabel('samples to fault')
    ax.set_ylabel('raw pressure')
fig.tight_layout(); display(fig); plt.close(fig)""")

md("""## 15. LSTM architecture and smoke test

The paper states two LSTM layers followed by 8- and 8-node fully connected layers. We interpret this as a sequence-to-sequence model whose dense head is applied at each timestep. Faults 1 and 2 use 64/128 LSTM widths; fault 3 uses 32/64. The paper describes dropout on LSTM input connections; PyTorch input/interlayer dropout approximates that behavior. L2 is optimizer weight decay. Prediction and target at timestep *t* are compared under the same mask.""")
code("""smoke = SequenceRUL(21, 32, 64)
smoke_x = torch.from_numpy(first_summary['x'][:2])
smoke_y = torch.from_numpy(first_summary['y'][:2] / 4515.0)
smoke_mask = torch.from_numpy(first_summary['mask'][:2].astype(np.float32))
smoke_optimizer = torch.optim.RMSprop(smoke.parameters(), lr=.001, weight_decay=1e-5)
smoke_pred = smoke(smoke_x)
smoke_loss = (((smoke_pred - smoke_y)**2)*smoke_mask).sum()/smoke_mask.sum()
assert smoke_pred.shape == smoke_y.shape == (2,300) and torch.isfinite(smoke_loss)
smoke_loss.backward(); smoke_optimizer.step()
print('Smoke output shape:', tuple(smoke_pred.shape), 'finite one-step loss:', float(smoke_loss.detach()))""")

md("""## 16–18. Fault-specific training

The paper repeats each LSTM 10 times and keeps the best validation model. This CPU reproduction runs three fixed-seed repeats per fault, with best validation weights restored. Model input and targets remain unscaled except that sample-count targets are divided by 4,515 during optimization and converted back for evaluation. Max epochs 40 and patience 6 are implementation limits, recorded per repeat.""")
code("""rf_bundles, rf_metrics = {}, {}
for index, slug in enumerate(FAULT_SLUGS, 1):
    print('Training RF', slug, flush=True)
    rf_bundles[slug], rf_metrics[slug] = train_evaluate_rf(slug, index, ART, MODEL_DIR, splits[slug])
    print(slug, {k:rf_metrics[slug][k] for k in ('precision','recall','f1','false_alarm_rate','roc_auc','pr_auc','stored_train_counts','virtual_fault_count_after_x1000','actual_weighted_rf_rows')}, flush=True)
display(pd.DataFrame(rf_metrics).T[['precision','recall','f1','false_alarm_rate','roc_auc','pr_auc']])""")

for slug in ("fault1", "fault2", "fault3"):
    md(f"### {slug}: three LSTM repeats and validation early stopping")
    code(f"""split = splits['{slug}']
train_data = load_summaries(ROOT, sequences, split['train_original_sequences'])
val_data = load_summaries(ROOT, sequences, split['validation_original_sequences'])
test_data = load_summaries(ROOT, sequences, split['test_original_sequences'])
print('{slug} generated summary counts:', len(train_data['x']), len(val_data['x']), len(test_data['x']))
{slug}_model, {slug}_train = train_repeated_lstm('{slug}', train_data, val_data, test_data, MODEL_DIR, repeats=3, max_epochs=40, patience=6)
print('Selected repeat:', {slug}_train['selected_repeat'], 'intrinsic test:', {slug}_train['test_intrinsic'])
print('All repeats:', {slug}_train['runs'])
del train_data, val_data, test_data""")

md("""## 19–21. Intrinsic RUL and combined RF→LSTM evaluation

Intrinsic RMSE and the paper's Eq. (2) SMAPE use every valid summarized test timestep, including the original sequence's 15 offsets. The target is sample count × 4 seconds, **not** wall-clock TTF. Combined evaluation covers the retained test-sequence tail: RF-negative positions produce `NaN`; RF-positive positions receive LSTM estimates. Detection bands use the original wall-clock TTF. These two domains are intentionally reported separately.""")
code("""lstm_models = {'fault1':fault1_model,'fault2':fault2_model,'fault3':fault3_model}
lstm_training = {'fault1':fault1_train,'fault2':fault2_train,'fault3':fault3_train}
combined_metrics = {}
curve_ids = {}
for slug in FAULT_SLUGS:
    test_data = load_summaries(ROOT, sequences, splits[slug]['test_original_sequences'])
    combined_metrics[slug], prediction_rows = combined_pipeline_metrics(rf_bundles[slug], lstm_models[slug], test_data, sequences)
    prediction_rows.to_parquet(ART / f'{slug}_test_predictions.parquet', index=False)
    curve_ids[slug] = plot_test_curves(prediction_rows, ART / f'{slug}_test_curves.png')
    print(slug, combined_metrics[slug], 'plotted:', curve_ids[slug])
    display(Image(filename=str(ART / f'{slug}_test_curves.png')))
    del test_data, prediction_rows""")

md("""## 22–23. Paper comparison and the previous 04

Paper Table 1 sequence counts and Tables 2–4 RFR/LSTM results are fixed reference values, never used as training targets. Old 04 scores forced numerical TTF predictions over its full fixed validation set. This notebook scores sample-based RUL only on retained run-to-fault test sequences, so old and new RMSE are **not directly comparable**.""")
code("""old_rmse = {'fault1':7870022.110,'fault2':8707703.688,'fault3':7872709.387}
old_r2 = {'fault1':-0.595888955,'fault2':-0.542861158,'fault3':-0.552500462}
comparison = pd.DataFrame([{'fault':slug, 'paper_train_sequences':PAPER_METRICS[slug]['paper_train_sequences'], 'paper_test_sequences':PAPER_METRICS[slug]['paper_test_sequences'], 'reproduction_valid_sequences':int(sequence_count_table.loc[sequence_count_table.fault.eq(slug),'valid_sequences'].iloc[0]), 'paper_RFR_RMSE_seconds':PAPER_METRICS[slug]['rfr_rmse_seconds'], 'paper_LSTM_RMSE_seconds':PAPER_METRICS[slug]['lstm_rmse_seconds'], 'reproduction_LSTM_RMSE_seconds':lstm_training[slug]['test_intrinsic']['rmse_seconds'], 'paper_LSTM_SMAPE_percent':PAPER_METRICS[slug]['lstm_smape_percent'], 'reproduction_LSTM_SMAPE_percent':lstm_training[slug]['test_intrinsic']['smape_percent'], 'old_04_full_range_RMSE_seconds':old_rmse[slug], 'old_04_full_range_R2':old_r2[slug]} for slug in FAULT_SLUGS])
display(comparison)
comparison.to_csv(ART / 'comparison_with_paper.csv', index=False)
display(pd.DataFrame(combined_metrics).T)""")

md("""## 24–26. Assumptions, saved artifacts and final diagnostic

**PAPER-SPECIFIED:** Raw complete-case deletion, 5,000/50,000 s bands, ×1,000 class imbalance concept, residual feature, 21 inputs, run-to-fault segmentation, Mmin/maxL/N/SampleR, RF and two-layer LSTM, RMSprop/MSE/dropout/L2/early stop, 80/20 and 90/10 splits, intrinsic RMSE/SMAPE, normal→no numerical RUL.

**REPRODUCTION-ASSUMPTION:** 3,600 s gap cutoff; previous same-mode event boundary; deduplicated repeated event rows; last observed measurement as fault endpoint; one-input RF pressure-to-flow relation; limited Bernoulli RF storage + weighted balance (different bootstrap distribution from literal upsampling); three repetitions rather than ten; random split seed not provided by paper.

**IMPLEMENTATION-DETAIL:** PyTorch CPU sequence model with input/interlayer dropout, 4 seconds per measurement as the paper's approximate conversion, target division by 4,515 only during optimization, padding mask, fixed seed 42, 40-epoch ceiling and patience 6. The paper's '1 sample / 4 minutes' phrase conflicts with its 15×4-second sampling arithmetic; this notebook uses the verified raw median of 4 seconds.

Discrepancies in sequence counts or RMSE trigger inspection of units, segmentation, gap cutoff, sample-count conversion, indexing, 21 inputs, shutter indicators, split disjointness, seconds conversion, and output-target alignment. They do not trigger post-hoc tuning.""")
code("""config = {
    'paper_source':'https://papers.phmsociety.org/index.php/phmconf/article/download/590/phmc_18_590',
    'raw_input_columns':list(extraction['raw_input_columns']), 'model_input_columns':list(INPUT_COLUMNS),
    'fault_to_ttf':schema['fault_to_ttf'], 'raw_ttf_unit':'seconds', 'seconds_conversion_factor':1.0,
    'rf_thresholds_seconds':{'fault_le':5000,'boundary_gt':5000,'boundary_le':50000,'normal_gt':50000},
    'rf_exact_band_counts_complete_rows':extraction['rf_exact_band_counts_complete_rows'],
    'rf_stored_band_counts':extraction['rf_stored_band_counts'], 'rf_storage_rates':extraction['rf_storage_rates'],
    'rf_training_counts':{slug:rf_metrics[slug]['stored_train_counts'] for slug in FAULT_SLUGS},
    'gap_seconds_reproduction_assumption':3600, 'sequence_parameters':{'Mmin_exclusive':3000,'maxL':300,'N':15,'SampleR':15},
    'seconds_per_sample_approx':4.0, 'split_seed':42, 'splits':splits,
    'lstm_architecture':{slug:{'lstm1':64 if slug!='fault3' else 32,'lstm2':128 if slug!='fault3' else 64,'dense':[8,8]} for slug in FAULT_SLUGS},
    'lstm_training':{'repeats':3,'paper_repeats':10,'max_epochs':40,'patience':6,'optimizer':'RMSprop','loss':'masked MSE','target_scale_samples':4515},
    'paper_reported_metrics':PAPER_METRICS,
    'assumptions_and_deviations':['gap cutoff 3600 s unspecified in paper','previous same-mode failure defines initial interval','last observed measurement treated as fault endpoint','RF residual relation fitted by one-input RandomForestRegressor','Bernoulli stored RF examples with balanced weights instead of literal x1000 bootstrap','three LSTM repetitions rather than ten','CPU PyTorch input/interlayer dropout interpretation'],
}
(ART / 'reproduction_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
metrics = {'sequence_count_table':sequence_count_table.to_dict(orient='records'),'rf':rf_metrics,'lstm':lstm_training,'combined':combined_metrics,'paper_comparison':comparison.to_dict(orient='records'),'sequence_generator_unit_test':unit_test,'curve_sequence_ids':curve_ids}
(ART / 'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding='utf-8')
print('Saved config, split, metrics, comparison, three RF models, three LSTM models, test predictions and six test curves.')
print('Sequence count differences:', sequence_count_table[['fault','valid_sequences','paper_sequences','difference']].to_dict(orient='records'))
print('All trained models reloaded successfully during model functions.')""")

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "FabJudge runtime", "language": "python", "name": "fabjudge-runtime"},
    "language_info": {"name": "python"},
})
nbf.write(notebook, NOTEBOOK)
print(NOTEBOOK)
