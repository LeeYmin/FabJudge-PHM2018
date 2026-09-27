"""Add observed outcomes to the fully executed 04b notebook."""

from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".venv" / "Lib" / "site-packages"))
import nbformat as nbf

notebook_path = ROOT / "notebooks" / "04b_huang2018_reproduction.ipynb"
artifact_dir = ROOT / "artifacts" / "huang2018"
metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
position = json.loads((artifact_dir / "position_only_audit.json").read_text(encoding="utf-8"))
names = {"fault1": "Pressure low", "fault2": "Pressure high", "fault3": "Leak"}
lines = [
    "## 28. Observed outcome and interpretation",
    "",
    f"Complete-case removal: **{metrics['missing_removed_percent']:.6f}%** of 82,189,440 raw measurement rows. Raw TTF is seconds, with conversion factor 1.0. All 391 retained sequences have a 4-second median measurement interval.",
    "",
    "| Fault | Valid sequences / paper | RF precision¹ | RF recall | RF F1¹ | RF false alarm | Offline LSTM RMSE / paper | Offline SMAPE / paper | Causal post-detection RMSE |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
counts = {r["fault"]: r for r in metrics["sequence_count_table"]}
for slug in ("fault1", "fault2", "fault3"):
    rf = metrics["rf"][slug]
    lstm = metrics["lstm"][slug]["test_intrinsic"]
    combined = metrics["combined"][slug]
    paper = next(r for r in metrics["paper_comparison"] if r["fault"] == slug)
    lines.append(
        f"| {names[slug]} | {counts[slug]['valid_sequences']} / {counts[slug]['paper_sequences']} | "
        f"{rf['precision']:.3f} | {rf['recall']:.3f} | {rf['f1']:.3f} | {rf['false_alarm_rate']:.3f} | "
        f"{lstm['rmse_seconds']:,.0f} / {paper['paper_LSTM_RMSE_seconds']:,.0f} s | "
        f"{lstm['smape_percent']:.2f} / {paper['paper_LSTM_SMAPE_percent']:.2f}% | "
        f"{combined['rul_rmse_after_detection_seconds']:,.0f} s |"
    )
lines += [
    "",
    "¹ RF precision and F1 use inverse inclusion-probability weights; the initially displayed sampled test metrics were biased by fault oversampling. The paper did not report directly comparable RF metrics.",
    "",
    f"**Target alignment diagnostic:** On all {position['checked_sequences']} retained sequences and {position['checked_augmented_points']:,} augmented time points, the failure-anchored offset plus time-step position reconstructs the sample-count target with maximum error **{position['maximum_position_only_target_error_samples']:.0f} samples**. The offline intrinsic RMSE values reproduce the paper-style evaluation format but cannot establish an online RUL forecast. The causal rolling-window plots show nearly flat low/high estimates and only a partial decline toward failure for leak; no monotonic smoothing was applied.",
    "",
    "**Main reasons the old 04 scores were poor:** It used a 25th-percentile near-failure target (515,398 / 1,016,521 / 2,466,814 seconds), different tree models and engineered windows, and forced numerical predictions across the full validation horizon. Its raw unit was already seconds. The old 04 full-range RMSE and this notebook's degradation-sequence RMSE have different evaluation domains.",
    "",
    "**Remaining reproduction differences:** The paper's numerical gap cutoff, original split seed, residual regression algorithm, and detailed LSTM implementation were unavailable. This run used a 3,600-second gap assumption, sample-weighted RF storage in place of literal ×1,000 duplicate bootstrap, and three LSTM repeats instead of ten. Fault 3's chosen repeat reached its 40-epoch ceiling. The last observed measurement can precede the fault event substantially (median 398 seconds; maximum 481,904 seconds across retained sequences).",
    "",
    "**FabJudge next step:** Keep the raw-unit audit, fault-mode-specific labels, pressure/flow residual, original-event grouping, and normal→no-RUL gate idea. Redesign RUL training and validation around causal decision-time windows before adopting numerical RUL outputs; calibrate the RF gate for the natural fault prevalence.",
]
notebook = nbf.read(notebook_path, as_version=4)
notebook.cells = [c for c in notebook.cells if "observed-outcome-summary" not in c.get("metadata", {}).get("tags", [])]
cell = nbf.v4.new_markdown_cell("\n".join(lines))
cell.metadata["tags"] = ["observed-outcome-summary"]
notebook.cells.append(cell)
nbf.write(notebook, notebook_path)
print(notebook_path)
