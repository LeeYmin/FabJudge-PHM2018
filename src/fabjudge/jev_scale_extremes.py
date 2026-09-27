"""Extend the saved 05a one-feature scale study with 0.1x and 10x arms.

The 45 validation windows, 20 train-only Robust features, Jev score question,
and one perturbed feature are inherited from the completed 05a study. A fresh
1x control measures response drift during this additional batch.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


MODEL = "typesafe/jev-1.13"
ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
FEATURE = "FLOWCOOLPRESSURE__shape_factor"
FACTORS = {"down_0p1": .1, "fresh_control_1p0": 1.0, "up_10p0": 10.0}
REGIONS = ("normal", "boundary", "near_failure")
MAX_NEW_ATTEMPTS = 135
COST_CAP_USD = .01
PROMPT_VERSION = "jev_tut_v4_scale_extremes_v1"
MODEL_PATTERN = re.compile(r"^typesafe/jev-1\.13(?:-\d{8})?$")


def _required(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required saved 05a artifact is missing: {path}")
    return path


def _json(path: Path):
    return json.loads(_required(path).read_text(encoding="utf-8"))


def _write_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _digest(obj) -> str:
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


def _secret(root: Path) -> str:
    key = os.environ.get("JEV_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    path = root / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                name, value = line.split("=", 1)
                if name.strip() in ("JEV_API_KEY", "OPENROUTER_API_KEY"):
                    value = value.strip().strip('"').strip("'")
                    if value:
                        return value
    raise RuntimeError("Set JEV_API_KEY or OPENROUTER_API_KEY in the environment or project .env.")


def _validate_reply(raw: dict) -> dict:
    model = raw.get("model")
    provider = raw.get("provider")
    if not isinstance(model, str) or not MODEL_PATTERN.fullmatch(model) or provider != "TypeSafe":
        raise ValueError(f"Wrong Jev response model/provider: {model!r}/{provider!r}")
    answer = raw.get("answers", {}).get("equipment_state", {})
    if answer.get("type") != "score":
        raise ValueError("Unexpected Jev answer type.")
    score = float(answer["score"])
    confidence = float(answer["confidence"])
    probabilities = {str(i): float(answer["probabilities"][str(i)]) for i in range(3)}
    if not (math.isfinite(score) and 0 <= score <= 2 and math.isfinite(confidence) and 0 <= confidence <= 1):
        raise ValueError("Jev score or confidence is outside its expected range.")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()) or not .95 <= sum(probabilities.values()) <= 1.05:
        raise ValueError("Invalid Jev three-level probabilities.")
    usage = raw.get("usage") or {}
    inp = usage.get("input_tokens")
    out = usage.get("output_tokens")
    cost = usage.get("cost")
    if not (isinstance(inp, int) and inp > 0 and isinstance(out, int) and out > 0 and
            isinstance(cost, (int, float)) and math.isfinite(float(cost)) and 0 <= cost <= COST_CAP_USD):
        raise ValueError("Missing or invalid OpenRouter usage input/output/cost fields.")
    return {"risk_score": score / 2, "native_score": score,
            "confidence": confidence, "probabilities": probabilities,
            "predicted_state": ("normal", "uncertain", "degrading")[
                max(range(3), key=lambda i: probabilities[str(i)])],
            "usage_input_tokens": inp, "usage_output_tokens": out,
            "usage_cost_usd": float(cost), "model": model, "provider": provider}


def _load_project(root: Path):
    base = root / "artifacts" / "jev_tut"
    previous = root / "artifacts" / "jev_scale_sensitivity"
    base_config = _json(base / "config.json")
    previous_config = _json(previous / "config.json")
    stats = _json(base / "train_scaler_stats.json")
    question = json.loads(_required(base / "prompt_v4.txt").read_text(encoding="utf-8").split("question=\n", 1)[1])
    previous_question = _json(previous / "decision_question.json")
    if question != previous_question or question["type"] != "score" or len(question["criteria"]) != 3:
        raise ValueError("Saved v4 Jev question differs from the earlier scale study.")
    if (base_config["model_id"] != MODEL or previous_config["model_id"] != MODEL or
            previous_config["feature"] != FEATURE or stats["source"] != "03 train groups only"):
        raise ValueError("Saved model, scaled feature, or Robust training source differs.")
    features = base_config["feature_columns"]
    targets = base_config["target_columns"]
    if len(features) != 20 or FEATURE not in features:
        raise ValueError("Expected the exact 20 selected numeric features and pressure shape factor.")
    samples = pd.read_csv(_required(base / "selected_samples.csv"))
    previous_results = pd.read_csv(_required(previous / "all_results.csv"),
                                   usecols=["sample_id", "representation", "target", "region", "risk_score",
                                            "confidence", "parse_success", "model", "provider"])
    if len(samples) != 45 or samples.sample_id.nunique() != 45:
        raise ValueError("Expected the fixed 45-row 05a sample.")
    counts = samples.groupby(["target", "region"]).size().reindex(
        pd.MultiIndex.from_product([targets, REGIONS]), fill_value=0)
    if not (counts == 5).all():
        raise ValueError("The original sample is not balanced 5 per target and TTF band.")
    split = pd.read_csv(_required(root / "artifacts" / "random_forest" / "ttf_baseline_group_split.csv"),
                        usecols=["group_id", "split"])
    validation = set(split.loc[split.split == "validation", "group_id"].astype(str))
    groups = samples.source_file.astype(str) + "::" + samples.sequence_index.astype(str)
    if not set(groups).issubset(validation):
        raise ValueError("A saved 05a sample is outside the fixed group-validation split.")
    if len(previous_results) != 135 or not previous_results.parse_success.all():
        raise ValueError("Prior 0.5x/1x/2x comparison is incomplete.")
    if set(previous_results.model) != {"typesafe/jev-1.13-20260917"} or set(previous_results.provider) != {"TypeSafe"}:
        raise ValueError("Prior returned model/provider differs from Jev 1.13/TypeSafe.")
    if set(previous_results.sample_id) != set(samples.sample_id):
        raise ValueError("Prior 0.5x/1x/2x sample set differs.")
    return base, previous, base_config, stats, question, features, targets, samples, previous_results


def _payload(sample, factor: float, question, features, stats):
    values = {}
    for feature in features:
        spec = stats["targets"][str(sample.target)][feature]
        raw_value = sample[feature] if isinstance(sample, pd.Series) else getattr(sample, feature)
        value = (float(raw_value) - float(spec["median_approx"])) / float(spec["safe_iqr"])
        if not math.isfinite(value):
            raise ValueError(f"Invalid Robust input for {feature}.")
        values[feature] = value * factor if feature == FEATURE else value
    return {"model": MODEL, "state": {"features": values},
            "questions": {"equipment_state": question}}


class _Caller:
    def __init__(self, root: Path, out: Path, allow_network: bool):
        self.root, self.out, self.allow_network = root, out, allow_network
        self.cache = out / "cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.ledger = out / "api_attempts.jsonl"
        self.attempts = ([json.loads(x) for x in self.ledger.read_text(encoding="utf-8").splitlines()]
                         if self.ledger.is_file() else [])
        self.new_calls = 0
        self.cache_hits = 0
        self.key = None

    def _cache_path(self, sample_id: str, variant: str, request_hash: str, repeat_tag: str = ""):
        identity = {"sample_id": sample_id, "variant": variant, "model": MODEL,
                    "request_hash": request_hash, "prompt_version": PROMPT_VERSION,
                    "repeat_tag": repeat_tag}
        return self.cache / f"{_digest(identity)}.json"

    def call(self, sample, variant: str, payload: dict) -> dict:
        request_hash = _digest(payload)
        cache_path = self._cache_path(str(sample.sample_id), variant, request_hash)
        if cache_path.is_file():
            rec = _json(cache_path)
            if (rec.get("request_hash") != request_hash or
                    rec.get("sample_id") != str(sample.sample_id) or
                    rec.get("variant") != variant):
                raise ValueError("Saved scale cache identity does not match this request.")
            _validate_reply(rec["raw_response"])
            self.cache_hits += 1
            return rec
        if not self.allow_network:
            raise FileNotFoundError(f"Missing successful extreme-scale cache: {cache_path}. Run explicitly with allow_network=True.")
        if len(self.attempts) >= MAX_NEW_ATTEMPTS:
            raise RuntimeError("The 135-attempt hard cap for this extension has been reached.")
        observed_costs = [float((a.get("usage") or {}).get("cost") or 0)
                          for a in self.attempts if a.get("usage")]
        observed_total = float(sum(observed_costs))
        estimate = max(float(np.quantile(observed_costs, .95)), .00005) if observed_costs else .00005
        if observed_total + estimate > COST_CAP_USD:
            raise RuntimeError("Projected next call would exceed the USD 0.01 extension cost cap.")
        if self.key is None:
            self.key = _secret(self.root)
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        req = urllib.request.Request(ENDPOINT, data=body,
                                     headers={"Authorization": f"Bearer {self.key}",
                                              "Content-Type": "application/json"}, method="POST")
        start = time.perf_counter()
        attempt = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                   "sample_id": str(sample.sample_id), "variant": variant,
                   "request_hash": request_hash, "requested_model": MODEL,
                   "http_status": None, "raw_response": None, "usage": None,
                   "parse_success": False, "error": None}
        self.new_calls += 1
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                attempt["http_status"] = response.status
                raw = json.loads(response.read().decode("utf-8"))
            attempt["raw_response"] = raw
            attempt["usage"] = raw.get("usage")
            parsed = _validate_reply(raw)
            attempt["parse_success"] = True
            attempt["parsed"] = parsed
        except urllib.error.HTTPError as exc:
            attempt["http_status"] = exc.code
            attempt["error"] = f"HTTP {exc.code}"
        except Exception as exc:
            attempt["error"] = f"{type(exc).__name__}: {str(exc).replace(self.key, '[REDACTED]')}"
        attempt["latency_seconds"] = time.perf_counter() - start
        self.attempts.append(attempt)
        with self.ledger.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(attempt, ensure_ascii=False, allow_nan=False) + "\n")
        if not attempt["parse_success"]:
            raise RuntimeError(f"Jev scale extension stopped on {variant}: {attempt['error']}")
        record = {"sample_id": str(sample.sample_id), "variant": variant,
                  "request_hash": request_hash, "raw_response": attempt["raw_response"],
                  "parsed": attempt["parsed"], "timestamp_utc": attempt["timestamp_utc"],
                  "latency_seconds": attempt["latency_seconds"]}
        temporary = cache_path.with_suffix(".tmp")
        _write_json(temporary, record)
        temporary.replace(cache_path)
        return record


def _arm_metrics(frame: pd.DataFrame, targets: list[str]):
    rows = []
    for (target, arm), d in frame.groupby(["target", "arm"], sort=True):
        means = d.groupby("region").risk_score.mean()
        binary = d.loc[d.region.isin(("normal", "near_failure"))]
        labels = (binary.region == "near_failure").astype(int)
        rows.append({"target": target, "arm": arm, "n": len(d),
                     "normal_mean": float(means["normal"]),
                     "boundary_mean": float(means["boundary"]),
                     "near_failure_mean": float(means["near_failure"]),
                     "separation": float(means["near_failure"] - means["normal"]),
                     "ordered": bool(means["normal"] < means["boundary"] < means["near_failure"]),
                     "ROC_AUC": float(roc_auc_score(labels, binary.risk_score)),
                     "PR_AUC": float(average_precision_score(labels, binary.risk_score)),
                     "confidence_mean": float(d.confidence.mean())})
    target_table = pd.DataFrame(rows)
    if set(target_table.target) != set(targets) or not (target_table.n == 15).all():
        raise ValueError("All scale arms must contain 15 responses per target.")
    macro = target_table.groupby("arm", as_index=False).agg(
        normal_mean=("normal_mean", "mean"), boundary_mean=("boundary_mean", "mean"),
        near_failure_mean=("near_failure_mean", "mean"), separation=("separation", "mean"),
        ordered_targets=("ordered", "sum"), ROC_AUC=("ROC_AUC", "mean"),
        PR_AUC=("PR_AUC", "mean"), confidence_mean=("confidence_mean", "mean"))
    return target_table, macro


def _paired_deltas(frame: pd.DataFrame):
    old = frame.loc[frame.batch == "historical"]
    new = frame.loc[frame.batch == "fresh"]
    old_p = old.pivot(index=["sample_id", "target", "region", "group_id"],
                      columns="arm", values="risk_score").reset_index()
    new_p = new.pivot(index=["sample_id", "target", "region", "group_id"],
                      columns="arm", values="risk_score").reset_index()
    merged = old_p.merge(new_p, on=["sample_id", "target", "region", "group_id"], validate="one_to_one")
    if len(merged) != 45 or merged.isna().any().any():
        raise ValueError("All historical and fresh scale responses must pair by sample.")
    comparisons = {
        "0.1x vs fresh 1x": ("0.1x new", "1x fresh"),
        "10x vs fresh 1x": ("10x new", "1x fresh"),
        "0.5x vs historical 1x": ("0.5x old", "1x old"),
        "2x vs historical 1x": ("2x old", "1x old"),
        "fresh 1x vs historical 1x": ("1x fresh", "1x old"),
    }
    paired = []
    for label, (candidate, reference) in comparisons.items():
        d = merged[["sample_id", "target", "region", "group_id"]].copy()
        d["comparison"] = label
        d["candidate_risk"] = merged[candidate]
        d["reference_risk"] = merged[reference]
        d["delta_risk"] = d.candidate_risk - d.reference_risk
        paired.append(d)
    paired = pd.concat(paired, ignore_index=True)
    summary = paired.groupby(["comparison", "target", "region"], as_index=False).agg(
        n=("sample_id", "size"), mean_delta=("delta_risk", "mean"),
        mean_abs_delta=("delta_risk", lambda x: float(x.abs().mean())),
        min_delta=("delta_risk", "min"), max_delta=("delta_risk", "max"))
    return paired, summary


def _paired_group_ci(paired: pd.DataFrame, n_boot: int = 1000):
    rng = np.random.default_rng(42)
    rows = []
    for comparison in ("0.1x vs fresh 1x", "10x vs fresh 1x"):
        d = paired.loc[paired.comparison == comparison].reset_index(drop=True)
        groups = np.asarray(sorted(d.group_id.unique()))
        codes = pd.Categorical(d.group_id, categories=groups).codes
        def estimate(weights):
            values = []
            for target in sorted(d.target.unique()):
                for region in ("normal", "near_failure"):
                    mask = ((d.target == target) & (d.region == region)).to_numpy()
                    if weights[mask].sum() == 0:
                        raise ValueError("Missing target/region group in bootstrap draw.")
                    values.append((target, region, float(np.average(d.delta_risk.to_numpy()[mask], weights=weights[mask]))))
            a = pd.DataFrame(values, columns=["target", "region", "delta"])
            p = a.pivot(index="target", columns="region", values="delta")
            return float((p.near_failure - p.normal).mean())
        observed = estimate(np.ones(len(d)))
        draws = []
        for _ in range(n_boot):
            multiplicity = np.bincount(rng.integers(len(groups), size=len(groups)), minlength=len(groups))
            try:
                draws.append(estimate(multiplicity[codes].astype(float)))
            except ValueError:
                pass
        rows.append({"comparison": comparison, "observed_separation_change": observed,
                     "ci_2p5": float(np.quantile(draws, .025)) if draws else np.nan,
                     "ci_97p5": float(np.quantile(draws, .975)) if draws else np.nan,
                     "valid_replicates": len(draws), "invalid_replicates": n_boot - len(draws),
                     "bootstrap_unit": "source_file::sequence_index"})
    return pd.DataFrame(rows)


def _usage_summary(results: pd.DataFrame):
    rows = []
    for arm, d in list(results.groupby("variant")) + [("all_new", results)]:
        rows.append({"arm": arm, "requests": len(d),
                     "input_total": int(d.usage_input_tokens.sum()),
                     "input_p50": float(d.usage_input_tokens.quantile(.5)),
                     "input_p95": float(d.usage_input_tokens.quantile(.95)),
                     "input_max": int(d.usage_input_tokens.max()),
                     "output_total": int(d.usage_output_tokens.sum()),
                     "output_p50": float(d.usage_output_tokens.quantile(.5)),
                     "output_p95": float(d.usage_output_tokens.quantile(.95)),
                     "output_max": int(d.usage_output_tokens.max()),
                     "cost_total_usd": float(d.usage_cost_usd.sum()),
                     "cost_p50_usd": float(d.usage_cost_usd.quantile(.5)),
                     "cost_p95_usd": float(d.usage_cost_usd.quantile(.95)),
                     "cost_max_usd": float(d.usage_cost_usd.max()),
                     "latency_p50_seconds": float(d.latency_seconds.quantile(.5)),
                     "latency_p95_seconds": float(d.latency_seconds.quantile(.95)),
                     "latency_max_seconds": float(d.latency_seconds.max())})
    return pd.DataFrame(rows)


def _plot(out: Path, frame: pd.DataFrame, targets: list[str], paired_summary: pd.DataFrame):
    arms = ("0.1x new", "0.5x old", "1x old", "1x fresh", "2x old", "10x new")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True, constrained_layout=True)
    for ax, target in zip(axes, targets):
        data = frame.loc[frame.target == target]
        for i, arm in enumerate(arms):
            rows = data.loc[data.arm == arm]
            x = rows.region.map({r: k for k, r in enumerate(REGIONS)}).to_numpy(dtype=float)
            ax.scatter(x + (i - 2.5) * .055, rows.risk_score, s=19, alpha=.75, label=arm)
        ax.set_xticks(range(3), REGIONS, rotation=18)
        ax.set_title(target.removeprefix("ttf_flowcool_").removesuffix("_seconds"))
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Jev 1.13 risk score")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False)
    fig.savefig(out / "risk_by_region_0p1_to_10x.png", dpi=155, bbox_inches="tight")
    plt.close(fig)

    subset = paired_summary.loc[paired_summary.comparison.isin(("0.1x vs fresh 1x", "10x vs fresh 1x"))]
    macro = subset.groupby(["comparison", "region"]).mean_delta.mean().unstack().reindex(columns=REGIONS)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for comparison, row in macro.iterrows():
        ax.plot(REGIONS, row.to_numpy(), marker="o", label=comparison)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Mean paired risk change")
    ax.legend(frameon=False)
    ax.grid(alpha=.25)
    fig.savefig(out / "paired_extreme_scale_change.png", dpi=155)
    plt.close(fig)


def _write_summary_markdown(out: Path, summary: dict):
    macro = pd.read_csv(_required(out / "macro_comparison.csv")).set_index("arm")
    ci = pd.read_csv(_required(out / "paired_group_bootstrap_ci.csv")).set_index("comparison")
    arm_order = ("0.1x new", "0.5x old", "1x old", "1x fresh", "2x old", "10x new")
    lines = ["# 05a pressure-shape scale extension", "",
             "The same 45 fixed validation windows and 20 train-only Robust features were used. "
             "Only FLOWCOOLPRESSURE__shape_factor changed. New calls included 0.1x, 10x, "
             "and a fresh 1x control; 0.5x/1x/2x rows come from the previous batch.", "",
             "| Factor and batch | Near − normal risk | ROC-AUC | PR-AUC | Ordered targets |",
             "|---|---:|---:|---:|---:|"]
    for arm in arm_order:
        r = macro.loc[arm]
        lines.append(f"| {arm} | {r.separation:.3f} | {r.ROC_AUC:.3f} | {r.PR_AUC:.3f} | {int(r.ordered_targets)}/3 |")
    lines.extend(["",
                  f"0.1x versus fresh 1x paired gap change: {summary['new_0p1_vs_fresh_1_separation_delta']:+.3f}; "
                  f"group-bootstrap 95% interval [{ci.loc['0.1x vs fresh 1x', 'ci_2p5']:.3f}, "
                  f"{ci.loc['0.1x vs fresh 1x', 'ci_97p5']:.3f}].",
                  f"10x versus fresh 1x paired gap change: {summary['new_10_vs_fresh_1_separation_delta']:+.3f}; "
                  f"group-bootstrap 95% interval [{ci.loc['10x vs fresh 1x', 'ci_2p5']:.3f}, "
                  f"{ci.loc['10x vs fresh 1x', 'ci_97p5']:.3f}].",
                  f"Fresh 1x versus historical 1x mean absolute score change: "
                  f"{summary['fresh_1_vs_historical_1_mean_abs_risk_change']:.4f}.", "",
                  f"New API responses: {summary['new_request_count']}; errors: {summary['parse_errors']}; "
                  f"input/output tokens: {summary['input_tokens_new_requests']}/"
                  f"{summary['output_tokens_new_requests']}; reported cost USD "
                  f"{summary['reported_cost_usd_new_requests']:.9f}. "
                  "All new responses were Jev 1.13 from TypeSafe.", "",
                  "Neither extreme factor improves the fresh 1x macro comparison. "
                  "Keep 1x as the exploratory baseline. The 45-window cohort is small, "
                  "the 10x feature reaches |50.3| in Robust units, and these JEV outputs "
                  "are decision support rather than measured fault truth."])
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_scale_extension(root: Path, allow_network: bool = False) -> dict:
    """Complete or read the extension. Offline is the default for notebook reruns."""
    root = Path(root).resolve()
    base, previous, base_config, stats, question, features, targets, samples, old_results = _load_project(root)
    out = previous / "extreme_0p1_10x"
    summary_path = out / "summary.json"
    if summary_path.is_file():
        saved = _json(summary_path)
        for name in ("new_results.csv", "macro_comparison.csv", "target_comparison.csv",
                     "paired_group_bootstrap_ci.csv", "usage_summary.csv", "api_attempts.jsonl"):
            _required(out / name)
        if len(list((out / "cache").glob("*.json"))) != 135:
            raise ValueError("The completed extension is missing successful response cache entries.")
        if not (out / "summary.md").is_file():
            _write_summary_markdown(out, saved)
        return saved
    out.mkdir(parents=True, exist_ok=True)
    caller = _Caller(root, out, allow_network)
    payloads = {}
    changed_values = []
    for sample in samples.itertuples(index=False):
        baseline = _payload(sample, 1.0, question, features, stats)
        for variant, factor in FACTORS.items():
            payload = _payload(sample, factor, question, features, stats)
            if set(payload) != {"model", "state", "questions"} or set(payload["state"]) != {"features"}:
                raise ValueError("Unexpected Jev request fields.")
            if payload["questions"] != baseline["questions"] or payload["model"] != baseline["model"]:
                raise ValueError("Scale arms changed the model or question.")
            changed = {f for f in features if payload["state"]["features"][f] != baseline["state"]["features"][f]}
            if changed - {FEATURE}:
                raise ValueError("A scale arm changed another feature.")
            if variant != "fresh_control_1p0":
                changed_values.append(payload["state"]["features"][FEATURE])
            payloads[(str(sample.sample_id), variant)] = payload
    _write_json(out / "request_audit.json", {
        "model": MODEL, "endpoint": ENDPOINT, "sample_count": 45,
        "planned_requests": 135, "scaled_feature": FEATURE,
        "factors": FACTORS, "feature_count": len(features),
        "only_selected_feature_changes": True,
        "same_v4_question_all_arms": True,
        "identifiers_labels_ttf_region_in_request": False,
        "scaled_feature_min": min(changed_values),
        "scaled_feature_max": max(changed_values),
        "sample_request_hash": _digest(payloads[(str(samples.iloc[0].sample_id), "down_0p1")]),
    })

    pilot_rows = []
    first_target = targets[0]
    pilot_samples = [samples.loc[(samples.target == first_target) & (samples.region == region)].iloc[0]
                     for region in ("normal", "near_failure")]
    try:
        for sample in pilot_samples:
            for variant in FACTORS:
                rec = caller.call(sample, variant, payloads[(str(sample.sample_id), variant)])
                pilot_rows.append({"sample_id": str(sample.sample_id), "region": sample.region,
                                   "variant": variant, "risk_score": rec["parsed"]["risk_score"],
                                   "model": rec["parsed"]["model"], "provider": rec["parsed"]["provider"],
                                   "input_tokens": rec["parsed"]["usage_input_tokens"],
                                   "output_tokens": rec["parsed"]["usage_output_tokens"],
                                   "cost_usd": rec["parsed"]["usage_cost_usd"]})
        if len(pilot_rows) != 6:
            raise RuntimeError("The six-request extreme-scale pilot is incomplete.")
        pilot = pd.DataFrame(pilot_rows)
        pilot.to_csv(out / "pilot.csv", index=False)
        projected = float(pilot.cost_usd.mean() * MAX_NEW_ATTEMPTS)
        if projected > COST_CAP_USD:
            raise RuntimeError("Pilot usage projects above the USD 0.01 hard cap.")
        _write_json(out / "pilot_summary.json", {"pilot_responses": 6,
                    "all_jev_typesafe": True, "all_parsed_usage_valid": True,
                    "projected_135_request_cost_usd": projected, "pass": True})

        rows = []
        for sample in samples.itertuples(index=False):
            for variant, factor in FACTORS.items():
                before_hits = caller.cache_hits
                rec = caller.call(sample, variant, payloads[(str(sample.sample_id), variant)])
                rows.append({"sample_id": str(sample.sample_id), "target": sample.target,
                             "region": sample.region, "ttf_seconds": float(sample.ttf_seconds),
                             "source_file": sample.source_file, "sequence_index": int(sample.sequence_index),
                             "window_index": int(sample.window_index),
                             "group_id": f"{sample.source_file}::{sample.sequence_index}",
                             "variant": variant, "factor": factor,
                             "request_hash": rec["request_hash"],
                             "timestamp_utc": rec["timestamp_utc"],
                             "cache_hit": caller.cache_hits > before_hits,
                             "latency_seconds": rec["latency_seconds"],
                             **rec["parsed"]})
    except Exception as exc:
        _write_json(out / "run_status.json", {"status": "stopped",
                    "reason": str(exc), "attempts_recorded": len(caller.attempts),
                    "new_calls_this_execution": caller.new_calls,
                    "successful_cache_entries": len(list(caller.cache.glob("*.json")))})
        raise

    current = pd.DataFrame(rows)
    if len(current) != 135 or current.duplicated(["sample_id", "variant"]).any():
        raise ValueError("The fresh 0.1x/1x/10x comparison is incomplete.")
    all_arms = []
    old_map = {"down_0p5": "0.5x old", "control_1p0": "1x old", "up_2p0": "2x old"}
    new_map = {"down_0p1": "0.1x new", "fresh_control_1p0": "1x fresh", "up_10p0": "10x new"}
    for source, batch, mapping in ((old_results, "historical", old_map), (current, "fresh", new_map)):
        d = source[["sample_id", "target", "region", "risk_score", "confidence",
                    "representation" if batch == "historical" else "variant"]].copy()
        d["arm"] = d["representation" if batch == "historical" else "variant"].map(mapping)
        d["batch"] = batch
        d = d.merge(samples[["sample_id", "source_file", "sequence_index"]],
                    on="sample_id", validate="many_to_one")
        d["group_id"] = d.source_file.astype(str) + "::" + d.sequence_index.astype(str)
        all_arms.append(d[["sample_id", "target", "region", "risk_score", "confidence", "arm", "batch", "group_id"]])
    combined = pd.concat(all_arms, ignore_index=True)
    target_table, macro = _arm_metrics(combined, targets)
    paired, paired_summary = _paired_deltas(combined)
    paired_ci = _paired_group_ci(paired)
    usage = _usage_summary(current)
    current.to_csv(out / "new_results.csv", index=False)
    target_table.to_csv(out / "target_comparison.csv", index=False)
    macro.to_csv(out / "macro_comparison.csv", index=False)
    paired.to_csv(out / "paired_sample_deltas.csv", index=False)
    paired_summary.to_csv(out / "paired_delta_by_region.csv", index=False)
    paired_ci.to_csv(out / "paired_group_bootstrap_ci.csv", index=False)
    usage.to_csv(out / "usage_summary.csv", index=False)
    _plot(out, combined, targets, paired_summary)
    total_cost = float(sum(float((r.get("usage") or {}).get("cost") or 0) for r in caller.attempts))
    if total_cost > COST_CAP_USD or len(caller.attempts) > MAX_NEW_ATTEMPTS:
        raise RuntimeError("Recorded extension usage exceeded a hard cap.")
    model_counts = dict(Counter(current.model))
    provider_counts = dict(Counter(current.provider))
    macro_lookup = macro.set_index("arm")
    summary = {
        "experiment": "05a extreme single-feature Robust-scale extension",
        "endpoint": ENDPOINT, "model_requested": MODEL,
        "feature_scaled": FEATURE, "factors": FACTORS,
        "sample_count": 45, "feature_count": 20,
        "new_request_count": len(current),
        "api_attempts_all_extension_executions": len(caller.attempts),
        "api_attempts_this_analysis_process": caller.new_calls,
        "cache_hits_this_analysis_process": caller.cache_hits,
        "pilot_responses_reused_in_original_full_run": 6,
        "parse_errors": int(sum(not a.get("parse_success", False) for a in caller.attempts)),
        "models": model_counts, "providers": provider_counts,
        "input_tokens_new_requests": int(current.usage_input_tokens.sum()),
        "output_tokens_new_requests": int(current.usage_output_tokens.sum()),
        "reported_cost_usd_new_requests": float(current.usage_cost_usd.sum()),
        "reported_cost_usd_all_attempts": total_cost,
        "hard_cost_cap_usd": COST_CAP_USD,
        "historical_0p5_1_2_preserved": True,
        "new_0p1_vs_fresh_1_separation_delta": float(macro_lookup.loc["0.1x new", "separation"] - macro_lookup.loc["1x fresh", "separation"]),
        "new_10_vs_fresh_1_separation_delta": float(macro_lookup.loc["10x new", "separation"] - macro_lookup.loc["1x fresh", "separation"]),
        "fresh_1_vs_historical_1_mean_abs_risk_change": float(paired.loc[paired.comparison == "fresh 1x vs historical 1x", "delta_risk"].abs().mean()),
        "conclusion_scope": "45 selected validation windows, five per target/TTF band; exploratory scale sensitivity only.",
    }
    _write_json(summary_path, summary)
    _write_summary_markdown(out, summary)
    _write_json(out / "run_status.json", {"status": "complete", "api_attempts": len(caller.attempts),
                "successful_cache_entries": len(list(caller.cache.glob("*.json")))})
    return summary
