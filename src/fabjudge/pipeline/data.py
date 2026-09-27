"""Build label-free EvidencePackets and a separately stored dev label table."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..huang2018 import INPUT_COLUMNS
from .config import PACKET_VERSION, PROJECT_ROOT, SAMPLE_CAP
from .features import causal_sensor_features, recent_lstm_features
from .routing import choose_percentile_capacity, percentile_positions
from .schemas import PACKET_FIELDS, make_packet, packet_issues, validate_packet

FAULT_CODES = {"fault1": 1, "fault2": 2, "fault3": 3}
TTF_LIMIT_SECONDS = 5000
ENDPOINTS_PATH = Path("artifacts/06e_fix_and_weakness_diagnosis/long_window_features.parquet")
SEQUENCE_META_PATH = Path("artifacts/huang2018/sequence_metadata.csv")
SPLIT_PATH = Path("artifacts/huang2018/split_metadata.json")


def _opaque(value: str, prefix: str) -> str:
    return hashlib.sha256(f"fabjudge-07a:{prefix}:{value}".encode("utf-8")).hexdigest()[:24]


def build_dev_sample(root: Path = PROJECT_ROOT, sample_cap: int = SAMPLE_CAP) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    root = Path(root).resolve()
    endpoints = pd.read_parquet(root / ENDPOINTS_PATH)
    split = json.loads((root / SPLIT_PATH).read_text(encoding="utf-8"))["splits"]
    dev_ids: set[str] = set()
    test_ids: set[str] = set()
    for fault, details in split.items():
        dev_ids.update(details["train_original_sequences"])
        dev_ids.update(details["validation_original_sequences"])
        test_ids.update(details["test_original_sequences"])
    if dev_ids & test_ids:
        raise ValueError("Development/test sequence overlap")
    if not set(endpoints["sequence_id"].astype(str)).issubset(dev_ids):
        raise ValueError("06e endpoint artifact contains a non-development sequence")
    endpoints = endpoints.loc[endpoints["sequence_id"].astype(str).isin(dev_ids)].copy()
    if "rf_probability_oof" not in endpoints:
        raise ValueError("06e sequence OOF RF probability is missing")
    endpoints = endpoints.sort_values(["sequence_id", "raw_row_index"], kind="stable")
    alarms_by_sequence = {
        sid: group.loc[group["lstm_pred_seconds"] <= TTF_LIMIT_SECONDS].reset_index(drop=True)
        for sid, group in endpoints.groupby("sequence_id", sort=True)
    }
    alarm_counts = [len(group) for group in alarms_by_sequence.values()]
    percentiles = choose_percentile_capacity(alarm_counts, total_cap=sample_cap)

    metadata = pd.read_csv(root / SEQUENCE_META_PATH).set_index("sequence_id")
    sequence_root = (root / "artifacts/huang2018/sequences").resolve()
    packet_rows: list[dict] = []
    label_rows: list[dict] = []
    for sequence_id, alarm_rows in alarms_by_sequence.items():
        if alarm_rows.empty:
            continue
        sequence_endpoints = endpoints.loc[
            endpoints["sequence_id"].astype(str).eq(sequence_id)
        ].reset_index(drop=True)
        meta = metadata.loc[sequence_id]
        relative_path = Path(str(meta["sequence_path"]).replace("\\", "/"))
        sequence_path = (root / relative_path).resolve()
        if not sequence_path.is_relative_to(sequence_root):
            raise ValueError("Sequence source escaped the expected processed artifact directory")
        with np.load(sequence_path) as arrays:
            x = arrays["x"]
            raw_index = arrays["raw_row_index"]
        if x.shape[1] != len(INPUT_COLUMNS) or not np.all(np.diff(raw_index) > 0):
            raise ValueError("Unexpected saved sequence input layout")
        fault_code = FAULT_CODES[str(alarm_rows.iloc[0]["fault"])]
        sequence_group = _opaque(str(sequence_id), "sequence-group")
        take = min(len(percentiles), len(alarm_rows))
        selected = []
        for percentile, position in percentile_positions(len(alarm_rows), percentiles[:take]):
            endpoint = alarm_rows.iloc[position]
            raw_row_index = int(endpoint["raw_row_index"])
            current_position = int(np.searchsorted(raw_index, raw_row_index))
            if current_position >= len(raw_index) or int(raw_index[current_position]) != raw_row_index:
                raise ValueError("06e endpoint could not be joined to its sequence array")
            fields: dict = {name: None for name in PACKET_FIELDS}
            fields["lstm_pred_seconds"] = endpoint["lstm_pred_seconds"]
            fields["rf_probability"] = endpoint["rf_probability_oof"]
            fields["fault_type"] = fault_code
            try:
                fields.update(recent_lstm_features(
                    sequence_endpoints["lstm_pred_seconds"].to_numpy(float),
                    int(np.searchsorted(sequence_endpoints["raw_row_index"].to_numpy(), raw_row_index)),
                ))
                fields.update(causal_sensor_features(x, current_position, INPUT_COLUMNS))
            except (ValueError, IndexError):
                # Missing causal history or non-finite sensor values remain missing; never impute.
                pass
            packet = make_packet(fields)
            issues = packet_issues(packet)
            if not issues:
                validate_packet(packet)
            elif not all(value is None or isinstance(value, (int, float)) for value in packet["fields"].values()):
                raise ValueError("Unexpected non-numeric EvidencePacket value")
            row_id = _opaque(f"{sequence_id}:{raw_row_index}:{percentile}", "row")
            selected.append((percentile, packet, issues, endpoint, row_id))
        # Distinct positions are required; the dataset has far more alarm endpoints than sampled points.
        if len({item[3]["raw_row_index"] for item in selected}) != len(selected):
            raise ValueError("Percentile sample selected duplicate alarm endpoints")
        for _, packet, issues, endpoint, row_id in selected:
            packet_rows.append({
                "row_id": row_id,
                "sequence_group": sequence_group,
                "packet_status": "DATA_REVIEW" if issues else "READY",
                "packet_json": json.dumps(packet, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
            })
            label_rows.append({
                "row_id": row_id,
                "sequence_group": sequence_group,
                "label_near_5k": int(int(endpoint["wall_ttf_seconds"]) <= TTF_LIMIT_SECONDS),
                "fault_type": fault_code,
            })

    sample = pd.DataFrame(packet_rows).sort_values(["sequence_group", "row_id"], kind="stable").reset_index(drop=True)
    labels = pd.DataFrame(label_rows).sort_values(["sequence_group", "row_id"], kind="stable").reset_index(drop=True)
    if sample.empty or sample["row_id"].duplicated().any() or labels["row_id"].duplicated().any():
        raise ValueError("Dev sample is empty or row identifiers collide")
    if "label_near_5k" in sample.columns or "wall_ttf_seconds" in sample.columns:
        raise ValueError("Label or exact TTF leaked into dev_sample.csv")
    # Verify the exact transmitted packet fields contain no identifier, timestamp, or outcome data.
    for packet_text in sample["packet_json"]:
        packet = json.loads(packet_text)
        if set(packet) != {"packet_version", "fields"}:
            raise ValueError("Packet identifier/label leakage detected")
        if packet["packet_version"] != PACKET_VERSION:
            raise ValueError("Unexpected packet version")
    merged = sample.merge(labels, on=["row_id", "sequence_group"], validate="one_to_one")
    stats = {}
    for fault_code in sorted(merged["fault_type"].unique()):
        fault = merged.loc[merged["fault_type"].eq(fault_code)]
        stats[str(fault_code)] = {
            "dev_alarm_samples": int(len(fault)),
            "near_failure_fraction": float(fault["label_near_5k"].mean()),
            "sequence_count": int(fault["sequence_group"].nunique()),
            "data_review_count": int(fault["packet_status"].eq("DATA_REVIEW").sum()),
            "ready_count": int(fault["packet_status"].eq("READY").sum()),
        }
    stats["total"] = {
        "dev_alarm_samples": int(len(sample)),
        "sequence_count": int(sample["sequence_group"].nunique()),
        "data_review_count": int(sample["packet_status"].eq("DATA_REVIEW").sum()),
        "ready_count": int(sample["packet_status"].eq("READY").sum()),
        "percentiles_used": percentiles,
        "per_sequence_alarm_sample_cap": len(percentiles),
        "sample_cap": sample_cap,
        "label_columns_separate": True,
    }
    return sample, labels, stats
