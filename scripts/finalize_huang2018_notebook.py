"""Append and execute the population-corrected RF audit after full training."""

from pathlib import Path
import json
import os
import sys


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / ".venv" / "Lib" / "site-packages"
sys.path.insert(0, str(SITE))
os.environ["JUPYTER_PATH"] = str(ROOT / ".jupyter") + os.pathsep + os.environ.get("JUPYTER_PATH", "")

import nbformat as nbf
from nbclient import NotebookClient


NOTEBOOK = ROOT / "notebooks" / "04b_huang2018_reproduction.ipynb"
notebook = nbf.read(NOTEBOOK, as_version=4)
marker = "population-corrected-rf-metrics"
if not any(marker in cell.get("metadata", {}).get("tags", []) for cell in notebook.cells):
    notebook.cells.append(nbf.v4.new_markdown_cell("""## 27. Population-corrected RF and causal pipeline verification

The RF test cache stores fault rows with probability 0.05 and normal rows with probability 0.0005. Therefore raw sampled precision, F1 and PR-AUC shown earlier overstate performance at the dataset's natural fault prevalence. This final evaluation uses inverse inclusion-probability weights. Recall, ROC-AUC and false alarm rate are unchanged by class-only sampling. The corrected figures below are the ones in the final report and `metrics.json`.

The retained-sequence median sample interval, complete-case removal percentage, and last-measurement-to-fault lag are also shown. The earlier combined calculation used summaries anchored at the known failure endpoint. That can give positional information unavailable at deployment. The final combined results below instead use causal windows ending at each decision time, with an explicit no-future-input assertion. Detection lead time is measured only once TTF is within 50,000 seconds; an earlier false alarm does not count as useful lead time. A position-only reconstruction check quantifies the limitation of the paper-style failure-anchored intrinsic test."""))
    cell = nbf.v4.new_code_cell("""import sys, json
from pathlib import Path
ROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent
sys.path[:0] = [str(ROOT / '.venv' / 'Lib' / 'site-packages'), str(ROOT / 'src')]
import joblib, numpy as np, pandas as pd, torch
from IPython.display import display, Image
from fabjudge.huang2018_models import FAULT_SLUGS, SequenceRUL, train_evaluate_rf, combined_pipeline_metrics_causal, plot_test_curves, summarize_original
ART = ROOT / 'artifacts' / 'huang2018'
MODEL_DIR = ROOT / 'artifacts' / 'models' / 'huang2018'
sequences = pd.read_csv(ART / 'sequence_metadata.csv')
splits = json.loads((ART / 'split_metadata.json').read_text(encoding='utf-8'))['splits']
metrics = json.loads((ART / 'metrics.json').read_text(encoding='utf-8'))
metrics['rf_sampled_before_population_weighting'] = metrics['rf']
metrics['combined_before_lead_filter'] = metrics['combined']
corrected_rf, corrected_combined = {}, {}
for index, slug in enumerate(FAULT_SLUGS, 1):
    rf_bundle, corrected_rf[slug] = train_evaluate_rf(slug, index, ART, MODEL_DIR, splits[slug])
    saved = torch.load(MODEL_DIR / f'{slug}_lstm.pt', map_location='cpu', weights_only=True)
    model = SequenceRUL(**saved['architecture'])
    model.load_state_dict(saved['state_dict'])
    corrected_combined[slug], rows = combined_pipeline_metrics_causal(ROOT, rf_bundle, model, sequences, splits[slug]['test_original_sequences'])
    rows.to_parquet(ART / f'{slug}_test_predictions.parquet', index=False)
    plot_test_curves(rows, ART / f'{slug}_causal_test_curves.png')
    del rows
metrics['rf'] = corrected_rf
metrics['combined'] = corrected_combined
interval = json.loads((ART / 'sequence_interval_audit.json').read_text(encoding='utf-8'))
extraction = json.loads((ART / 'raw_input_extraction.json').read_text(encoding='utf-8'))
removed_percent = 100 * extraction['missing']['rows_removed_missing_or_invalid'] / extraction['missing']['rows_before']
metrics['sample_interval_audit'] = interval
metrics['missing_removed_percent'] = removed_percent
(ART / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
config = json.loads((ART / 'reproduction_config.json').read_text(encoding='utf-8'))
config['rf_evaluation_inclusion_probability'] = {'fault':0.05,'normal':0.0005}
config['reported_rf_precision_f1_pr_auc'] = 'inverse-probability-weighted test estimates'
config['reported_combined_pipeline_evaluation'] = 'causal rolling windows ending at each decision time, endpoint stride 15'
(ART / 'reproduction_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
print('Missing rows removed: %.6f%%' % removed_percent)
print('Sequence interval audit:', interval)
example = sequences.loc[sequences.status.eq('valid')].iloc[0]
anchor = summarize_original(ROOT / example.sequence_path)
position_only = np.zeros_like(anchor['y'])
for offset in range(15):
    length = int(anchor['mask'][offset].sum())
    position_only[offset,-length:] = offset + 15*np.arange(length-1,-1,-1)
position_rmse = float(np.sqrt(np.mean((position_only[anchor['mask']] - anchor['y'][anchor['mask']])**2)) * 4)
assert position_rmse == 0.0
metrics['failure_anchored_position_only_rmse_seconds'] = position_rmse
(ART / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
print('Failure-anchored position-only target reconstruction RMSE:', position_rmse, 'seconds; intrinsic LSTM metrics are an offline paper-style comparison, not a deployable online score.')
display(Image(filename=str(ART / 'pretraining_raw_sequences.png')))
display(pd.DataFrame(corrected_rf).T[['precision','recall','f1','false_alarm_rate','roc_auc','pr_auc','sampled_test_precision','sampled_test_f1']])
display(pd.DataFrame(corrected_combined).T)
for slug in FAULT_SLUGS:
    display(Image(filename=str(ART / f'{slug}_causal_test_curves.png')))
print('Corrected metrics and reloaded model evaluations saved.')""")
    cell.metadata["tags"] = [marker]
    notebook.cells.append(cell)
    nbf.write(notebook, NOTEBOOK)

index = next(i for i, cell in enumerate(notebook.cells) if marker in cell.get("metadata", {}).get("tags", []))
client = NotebookClient(notebook, timeout=None, kernel_name="fabjudge-runtime", resources={"metadata": {"path": str(ROOT)}})
try:
    with client.setup_kernel():
        client.execute_cell(notebook.cells[index], index)
finally:
    nbf.write(notebook, NOTEBOOK)
for output in notebook.cells[index].get("outputs", []):
    if output.get("output_type") == "stream":
        print(output.get("text", ""))
print(f"Finalized cell {index+1}/{len(notebook.cells)}", flush=True)
