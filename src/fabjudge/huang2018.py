"""Paper-specific PHM 2018 audits and sequence helpers.

This module reads raw data but never modifies ``data/raw``. It is independent
of the FabJudge window-feature pipeline.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json

import numpy as np
import pandas as pd


FAULT_NAMES = (
    "FlowCool Pressure Dropped Below Limit",
    "Flowcool Pressure Too High Check Flowcool Pump",
    "Flowcool leak",
)
GAP_CANDIDATES_SECONDS = (16, 60, 300, 900, 3600, 21600)
TTF_QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


def raw_paths(project_root: Path) -> tuple[list[Path], Path, Path]:
    train_dir = project_root / "data" / "raw" / "phm_data_challenge_2018" / "train"
    sensor_paths = sorted(train_dir.glob("*_DC_train.csv"))
    ttf_dir = train_dir / "train_ttf"
    fault_dir = train_dir / "train_faults"
    if len(sensor_paths) != 20:
        raise ValueError(f"Expected the 20 raw training tools; found {len(sensor_paths)}")
    for path in sensor_paths:
        if not (ttf_dir / path.name).is_file():
            raise FileNotFoundError(f"Missing TTF file for {path.name}")
    return sensor_paths, ttf_dir, fault_dir


def discover_schema(project_root: Path) -> dict:
    sensor_paths, ttf_dir, fault_dir = raw_paths(project_root)
    sensor_header = pd.read_csv(sensor_paths[0], nrows=0).columns.tolist()
    ttf_header = pd.read_csv(ttf_dir / sensor_paths[0].name, nrows=0).columns.tolist()
    for path in sensor_paths[1:]:
        if pd.read_csv(path, nrows=0).columns.tolist() != sensor_header:
            raise ValueError(f"Raw sensor header differs: {path.name}")
        if pd.read_csv(ttf_dir / path.name, nrows=0).columns.tolist() != ttf_header:
            raise ValueError(f"Raw TTF header differs: {path.name}")
    if ttf_header[0] != "time" or len(ttf_header) != 4:
        raise ValueError(f"Unexpected raw TTF header: {ttf_header}")
    sample = pd.read_csv(sensor_paths[0], nrows=1000, low_memory=False)
    if sensor_header[0] != "time":
        raise ValueError(f"Unexpected raw sensor time column: {sensor_header[0]}")
    fault_files = sorted(fault_dir.glob("*_train_fault_data.csv"))
    if len(fault_files) != len(sensor_paths):
        raise ValueError("Fault file inventory does not match the 20 tools")
    name_counts = Counter()
    for path in fault_files:
        frame = pd.read_csv(path, usecols=["fault_name"])
        name_counts.update(frame["fault_name"].dropna().astype(str))
    for fault_name in FAULT_NAMES:
        if fault_name not in name_counts:
            raise ValueError(f"Expected fault label absent from raw fault files: {fault_name}")
    ttf_map = {fault_name: f"TTF_{fault_name}" for fault_name in FAULT_NAMES}
    if set(ttf_map.values()) != set(ttf_header[1:]):
        raise ValueError(f"TTF/fault label mismatch: {ttf_map}, {ttf_header}")
    return {
        "sensor_columns": sensor_header,
        "sensor_dtypes_sample": {k: str(v) for k, v in sample.dtypes.items()},
        "ttf_columns": ttf_header,
        "fault_to_ttf": ttf_map,
        "fault_name_counts": dict(name_counts),
        "sensor_file_count": len(sensor_paths),
        "fault_file_count": len(fault_files),
        "sample_shutter_values": sorted(
            pd.to_numeric(sample["FIXTURESHUTTERPOSITION"], errors="coerce")
            .dropna().unique().astype(int).tolist()
        ),
    }


def _fault_path_for_sensor(path: Path, fault_dir: Path) -> Path:
    stem = path.stem.removesuffix("_DC_train")
    return fault_dir / f"{stem}_train_fault_data.csv"


def audit_raw_labels(project_root: Path, output_dir: Path, *, chunk_rows: int = 250_000) -> dict:
    """Audit units, target ranges, timestamp gaps, and unfiltered events.

    Exact counts/min/max come from all rows. Quantiles use a seeded 1% sample,
    which is reported explicitly; no full raw label array is retained.
    """
    schema = discover_schema(project_root)
    sensor_paths, ttf_dir, fault_dir = raw_paths(project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    target_stats = {
        name: {
            "count": 0, "nan_count": 0, "negative_count": 0,
            "min": float("inf"), "max": float("-inf"),
            "le_5000": 0, "between_5000_50000": 0, "gt_50000": 0,
            "exact_timestamp_matches": 0, "timestamp_mismatches": 0,
            "mismatch_examples": [], "quantile_sample": [],
        }
        for name in FAULT_NAMES
    }
    gap_counts = Counter()
    gap_samples: list[np.ndarray] = []
    file_rows = []
    event_rows = []
    extra_fault_names = Counter()

    for file_number, sensor_path in enumerate(sensor_paths, 1):
        print(f"Auditing raw labels {file_number}/{len(sensor_paths)}: {sensor_path.name}", flush=True)
        fault_path = _fault_path_for_sensor(sensor_path, fault_dir)
        fault_frame = pd.read_csv(fault_path, usecols=["time", "fault_name", "Tool"])
        extra_fault_names.update(
            fault_frame.loc[~fault_frame["fault_name"].isin(FAULT_NAMES), "fault_name"]
            .dropna().astype(str)
        )
        event_times = {
            name: np.sort(
                fault_frame.loc[fault_frame["fault_name"].eq(name), "time"]
                .to_numpy(dtype=np.int64)
            )
            for name in FAULT_NAMES
        }
        time_parts: list[np.ndarray] = []
        last_time = None
        row_count = 0
        for chunk in pd.read_csv(
            ttf_dir / sensor_path.name,
            usecols=["time", *schema["ttf_columns"][1:]],
            chunksize=chunk_rows,
            low_memory=False,
        ):
            times = chunk["time"].to_numpy(dtype=np.int64)
            if len(times) == 0:
                continue
            row_count += len(times)
            time_parts.append(times)
            if last_time is None:
                deltas = np.diff(times)
            else:
                deltas = np.diff(np.concatenate((np.array([last_time]), times)))
            last_time = int(times[-1])
            if deltas.size:
                gap_counts["total"] += len(deltas)
                gap_counts["negative"] += int(np.sum(deltas < 0))
                gap_counts["zero"] += int(np.sum(deltas == 0))
                gap_counts["exact_4_seconds"] += int(np.sum(deltas == 4))
                for threshold in GAP_CANDIDATES_SECONDS:
                    gap_counts[f"gt_{threshold}"] += int(np.sum(deltas > threshold))
                chosen = deltas[rng.random(len(deltas)) < 0.01]
                if chosen.size:
                    gap_samples.append(chosen.astype(np.int64, copy=False))

            for fault_name, ttf_column in schema["fault_to_ttf"].items():
                values = pd.to_numeric(chunk[ttf_column], errors="coerce").to_numpy(dtype=np.float64)
                finite = np.isfinite(values)
                valid = values[finite]
                stats = target_stats[fault_name]
                stats["count"] += int(finite.sum())
                stats["nan_count"] += int((~finite).sum())
                if valid.size:
                    stats["negative_count"] += int(np.sum(valid < 0))
                    stats["min"] = min(stats["min"], float(valid.min()))
                    stats["max"] = max(stats["max"], float(valid.max()))
                    stats["le_5000"] += int(np.sum(valid <= 5000))
                    stats["between_5000_50000"] += int(np.sum((valid > 5000) & (valid <= 50000)))
                    stats["gt_50000"] += int(np.sum(valid > 50000))
                    selected = valid[rng.random(len(valid)) < 0.01]
                    if selected.size:
                        stats["quantile_sample"].append(selected.astype(np.float32))

                faults = event_times[fault_name]
                if faults.size and finite.any():
                    right = np.searchsorted(faults, times[finite], side="left")
                    available = right < len(faults)
                    provided = values[finite][available]
                    expected = faults[right[available]] - times[finite][available]
                    matches = np.isclose(provided, expected, atol=0.0, rtol=0.0)
                    stats["exact_timestamp_matches"] += int(matches.sum())
                    stats["timestamp_mismatches"] += int((~matches).sum())
                    if (~matches).any() and len(stats["mismatch_examples"]) < 5:
                        mismatch_indices = np.flatnonzero(~matches)[: 5 - len(stats["mismatch_examples"])]
                        stats["mismatch_examples"].extend(
                            {"raw_ttf": float(provided[i]), "fault_time_minus_time": int(expected[i])}
                            for i in mismatch_indices
                        )

        all_times = np.concatenate(time_parts) if time_parts else np.empty(0, dtype=np.int64)
        if all_times.size != row_count:
            raise AssertionError("Time inventory length mismatch")
        if np.any(np.diff(all_times) < 0):
            raise ValueError(f"Non-monotonic raw time in {sensor_path.name}")
        file_rows.append({
            "source_file": sensor_path.name,
            "tool": str(fault_frame["Tool"].dropna().iloc[0]) if len(fault_frame) else "",
            "rows": row_count,
            "first_time": int(all_times[0]) if row_count else None,
            "last_time": int(all_times[-1]) if row_count else None,
        })
        deltas = np.diff(all_times)
        gap_indices = {threshold: np.flatnonzero(deltas > threshold) for threshold in GAP_CANDIDATES_SECONDS}
        for fault_name in FAULT_NAMES:
            previous_event_index = -1
            for event_number, fault_time in enumerate(event_times[fault_name], 1):
                event_index = int(np.searchsorted(all_times, fault_time, side="right") - 1)
                exact_time = bool(event_index >= 0 and all_times[event_index] == fault_time)
                observed_time = int(all_times[event_index]) if event_index >= 0 else None
                base_start = previous_event_index + 1
                row = {
                    "sequence_id": f"{sensor_path.name}|{fault_name}|{int(fault_time)}",
                    "source_file": sensor_path.name,
                    "fault_name": fault_name,
                    "fault_time": int(fault_time),
                    "event_number": event_number,
                    "event_row_index": event_index,
                    "event_time_exact_match": exact_time,
                    "event_row_time": observed_time,
                    "event_gap_seconds": int(fault_time - observed_time) if observed_time is not None else None,
                    "base_start_index": base_start,
                    "base_length": max(0, event_index - base_start + 1),
                }
                for threshold, indices in gap_indices.items():
                    preceding = np.searchsorted(indices, event_index, side="left") - 1
                    truncated_start = base_start
                    if preceding >= 0 and indices[preceding] >= base_start:
                        truncated_start = int(indices[preceding]) + 1
                    row[f"start_after_gap_{threshold}"] = truncated_start
                    row[f"length_after_gap_{threshold}"] = max(0, event_index - truncated_start + 1)
                event_rows.append(row)
                previous_event_index = max(previous_event_index, event_index)

    event_table = pd.DataFrame(event_rows)
    event_table.to_csv(output_dir / "raw_failure_events.csv", index=False)
    pd.DataFrame(file_rows).to_csv(output_dir / "raw_label_file_inventory.csv", index=False)
    quantile_names = ("p01", "p05", "p25", "median", "p75", "p95", "p99")
    for fault_name, stats in target_stats.items():
        sample = np.concatenate(stats.pop("quantile_sample"))
        stats["quantile_sample_count"] = len(sample)
        stats["quantile_method"] = "seeded 1% row sample; exact count/min/max and band counts"
        stats.update({k: float(v) for k, v in zip(quantile_names, np.quantile(sample, TTF_QUANTILES))})
        stats["le_5000_ratio"] = stats["le_5000"] / stats["count"] if stats["count"] else None
        stats["between_5000_50000_ratio"] = (
            stats["between_5000_50000"] / stats["count"] if stats["count"] else None
        )
        stats["gt_50000_ratio"] = stats["gt_50000"] / stats["count"] if stats["count"] else None
    gaps = np.concatenate(gap_samples)
    gap_summary = {
        "counts": dict(gap_counts),
        "sample_count": len(gaps),
        "sample_method": "seeded 1% row sample",
        "median": float(np.quantile(gaps, 0.5)),
        "p90": float(np.quantile(gaps, 0.9)),
        "p95": float(np.quantile(gaps, 0.95)),
        "p99": float(np.quantile(gaps, 0.99)),
        "p999": float(np.quantile(gaps, 0.999)),
        "max_sample": int(gaps.max()),
    }
    result = {
        "schema": schema,
        "raw_ttf_unit": "seconds (verified by exact TTF = next fault timestamp - current timestamp relation)",
        "seconds_conversion_factor": 1.0,
        "target_stats": target_stats,
        "gap_summary": gap_summary,
        "file_inventory": file_rows,
        "extra_fault_names": dict(extra_fault_names),
        "raw_failure_events": len(event_table),
        "sequence_candidate_counts": {
            fault_name: {
                str(threshold): int(
                    (event_table.loc[event_table["fault_name"].eq(fault_name), f"length_after_gap_{threshold}"] > 3000).sum()
                )
                for threshold in GAP_CANDIDATES_SECONDS
            }
            for fault_name in FAULT_NAMES
        },
        "input_signature": [
            (str(path.relative_to(project_root)), path.stat().st_size, path.stat().st_mtime_ns)
            for path in [*sensor_paths, *(ttf_dir / p.name for p in sensor_paths), *sorted(fault_dir.glob("*.csv"))]
        ],
    }
    (output_dir / "raw_label_audit.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    return result


def sampled_raw_indices(end_index: int, *, max_length: int, sample_rate: int, offset: int = 0) -> np.ndarray:
    """Return ascending source indices for a backward-sampled run endpoint.

    ``offset=0`` starts at the final measurement; successive offsets move the
    endpoint one raw measurement earlier. A short history is left-padded only
    by the caller, never by this index function.
    """
    if end_index < 0 or max_length < 1 or sample_rate < 1 or offset < 0:
        raise ValueError("Invalid sequence generator arguments")
    endpoint = end_index - offset
    if endpoint < 0:
        return np.empty(0, dtype=np.int64)
    indices = endpoint - sample_rate * np.arange(max_length, dtype=np.int64)
    return indices[indices >= 0][::-1]


def smape_paper(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Huang et al. Eq. (2): 100*mean(|e|/(|pred|+|true|))."""
    actual = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)
    if actual.shape != predicted.shape:
        raise ValueError("SMAPE inputs have different shapes")
    denominator = np.abs(actual) + np.abs(predicted)
    ratio = np.zeros_like(denominator)
    np.divide(np.abs(predicted - actual), denominator, out=ratio, where=denominator > 0)
    return float(100.0 * ratio.mean())


SENSOR_COLUMNS = (
    "IONGAUGEPRESSURE", "ETCHBEAMVOLTAGE", "ETCHBEAMCURRENT",
    "ETCHSUPPRESSORVOLTAGE", "ETCHSUPPRESSORCURRENT", "FLOWCOOLFLOWRATE",
    "FLOWCOOLPRESSURE", "ETCHGASCHANNEL1READBACK", "ETCHPBNGASREADBACK",
    "FIXTURETILTANGLE", "ROTATIONSPEED", "ACTUALROTATIONANGLE",
)
OPERATING_COLUMNS = (
    "ETCHSOURCEUSAGE", "ETCHAUXSOURCETIMER", "ETCHAUX2SOURCETIMER",
    "ACTUALSTEPDURATION",
)
SHUTTER_COLUMN = "FIXTURESHUTTERPOSITION"
SHUTTER_CATEGORIES = (0, 1, 2, 3, 255)
INPUT_RAW_COLUMNS = (*SENSOR_COLUMNS, *OPERATING_COLUMNS, SHUTTER_COLUMN)
INPUT_COLUMNS = (*SENSOR_COLUMNS, *OPERATING_COLUMNS, *(f"{SHUTTER_COLUMN}_{x}" for x in SHUTTER_CATEGORIES))


def encode_inputs(frame: pd.DataFrame) -> np.ndarray:
    """Paper's 12 sensor + 4 operating time + 5 shutter indicator inputs."""
    numeric = frame.loc[:, [*SENSOR_COLUMNS, *OPERATING_COLUMNS]].to_numpy(dtype=np.float32)
    shutter = pd.to_numeric(frame[SHUTTER_COLUMN], errors="coerce").to_numpy(dtype=np.float32)
    onehot = (shutter[:, None] == np.asarray(SHUTTER_CATEGORIES)[None, :]).astype(np.float32)
    values = np.concatenate((numeric, onehot), axis=1)
    if values.shape[1] != 21 or not np.isfinite(values).all():
        raise ValueError("Invalid or non-finite 21-dimensional paper input")
    return values


def extract_raw_inputs(
    project_root: Path,
    output_dir: Path,
    *,
    gap_seconds: int = 3600,
    min_length: int = 3000,
    history_rows: int = 4515,
    chunk_rows: int = 100_000,
    seed: int = 42,
) -> dict:
    """One full sensor/TTF pass for sequence tails, missingness, and RF samples.

    Gap threshold and Bernoulli RF storage rates are reproduction assumptions.
    The raw RF band counts are exact; stored examples are a deterministic
    compute-bounded sample with the inclusion rate recorded in metadata.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    sensor_paths, ttf_dir, _ = raw_paths(project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_dir = output_dir / "sequences"
    sequence_dir.mkdir(exist_ok=True)
    event_table = pd.read_csv(output_dir / "raw_failure_events.csv").drop_duplicates("sequence_id").copy()
    event_table["candidate"] = event_table[f"length_after_gap_{gap_seconds}"] > min_length
    counts = Counter()
    column_missing = Counter()
    shutter_counts = Counter()
    rf_band_counts = {name: Counter() for name in FAULT_NAMES}
    rf_selected_counts = {name: Counter() for name in FAULT_NAMES}
    rng = np.random.default_rng(seed)
    rates = {"fault": 0.05, "boundary": 0.005, "normal": 0.0005}
    writers: dict[str, pq.ParquetWriter] = {}
    window_rows = []
    sample_numeric: list[np.ndarray] = []
    sequence_records = []
    for file_number, sensor_path in enumerate(sensor_paths, 1):
        print(f"Extracting raw inputs {file_number}/{len(sensor_paths)}: {sensor_path.name}", flush=True)
        file_events = event_table.loc[
            event_table["source_file"].eq(sensor_path.name) & event_table["candidate"]
        ].copy()
        windows = []
        for _, row in file_events.iterrows():
            start = max(int(row[f"start_after_gap_{gap_seconds}"]), int(row["event_row_index"]) - history_rows + 1)
            windows.append((start, int(row["event_row_index"]), row))
        window_parts = {row["sequence_id"]: [] for _, _, row in windows}
        file_faults = {
            name: np.sort(
                event_table.loc[
                    event_table["source_file"].eq(sensor_path.name) & event_table["fault_name"].eq(name),
                    "fault_time",
                ].to_numpy(dtype=np.int64)
            )
            for name in FAULT_NAMES
        }
        row_start = 0
        sensor_reader = pd.read_csv(sensor_path, chunksize=chunk_rows, low_memory=False)
        ttf_reader = pd.read_csv(ttf_dir / sensor_path.name, chunksize=chunk_rows, low_memory=False)
        for sensor, ttf in zip(sensor_reader, ttf_reader, strict=True):
            n = len(sensor)
            if len(ttf) != n or not np.array_equal(sensor["time"].to_numpy(), ttf["time"].to_numpy()):
                raise ValueError(f"Sensor/TTF row or time mismatch: {sensor_path.name}, row {row_start}")
            counts["rows_before"] += n
            column_missing.update({col: int(value) for col, value in sensor.isna().sum().items()})
            for col in INPUT_RAW_COLUMNS:
                sensor[col] = pd.to_numeric(sensor[col], errors="coerce")
            complete = sensor.notna().all(axis=1).to_numpy().copy()
            complete &= np.isfinite(sensor.loc[:, INPUT_RAW_COLUMNS].to_numpy(dtype=np.float64)).all(axis=1)
            shutter = sensor[SHUTTER_COLUMN].to_numpy(dtype=np.float64)
            complete &= np.isin(shutter, SHUTTER_CATEGORIES)
            counts["rows_removed_missing_or_invalid"] += int((~complete).sum())
            counts["rows_after"] += int(complete.sum())
            valid_shutter, valid_counts = np.unique(shutter[np.isfinite(shutter)], return_counts=True)
            shutter_counts.update({str(float(k)): int(v) for k, v in zip(valid_shutter, valid_counts)})
            times = sensor["time"].to_numpy(dtype=np.int64)
            if complete.any():
                sampled = np.flatnonzero(complete & (rng.random(n) < 0.0002))
                if sampled.size:
                    sample_numeric.append(sensor.iloc[sampled].loc[:, INPUT_RAW_COLUMNS].to_numpy(dtype=np.float32))
            for start, end, event in windows:
                left = max(start, row_start)
                right = min(end + 1, row_start + n)
                if left < right:
                    part = sensor.iloc[left - row_start:right - row_start].loc[:, ["time", *INPUT_RAW_COLUMNS]].copy()
                    part.insert(0, "raw_row_index", np.arange(left, right, dtype=np.int64))
                    window_parts[event["sequence_id"]].append(part.loc[complete[left - row_start:right - row_start]])
            for fault_name in FAULT_NAMES:
                values = pd.to_numeric(ttf[f"TTF_{fault_name}"], errors="coerce").to_numpy(dtype=np.float64)
                finite = np.isfinite(values) & complete
                band_masks = {
                    "fault": finite & (values <= 5000),
                    "boundary": finite & (values > 5000) & (values <= 50000),
                    "normal": finite & (values > 50000),
                }
                for band, mask in band_masks.items():
                    rf_band_counts[fault_name][band] += int(mask.sum())
                    chosen = np.flatnonzero(mask & (rng.random(n) < rates[band]))
                    rf_selected_counts[fault_name][band] += len(chosen)
                    if chosen.size == 0:
                        continue
                    faults = file_faults[fault_name]
                    next_event = np.searchsorted(faults, times[chosen], side="left")
                    event_time = np.full(len(chosen), -1, dtype=np.int64)
                    available = next_event < len(faults)
                    event_time[available] = faults[next_event[available]]
                    sample = sensor.iloc[chosen].loc[:, INPUT_RAW_COLUMNS].copy()
                    sample.insert(0, "source_file", sensor_path.name)
                    sample.insert(1, "raw_row_index", row_start + chosen)
                    sample.insert(2, "time", times[chosen])
                    sample.insert(3, "fault_time", event_time)
                    sample.insert(4, "sequence_id", [
                        f"{sensor_path.name}|{fault_name}|{int(t)}" if t >= 0 else ""
                        for t in event_time
                    ])
                    sample.insert(5, "ttf_seconds", values[chosen])
                    sample.insert(6, "band", band)
                    for input_column in INPUT_RAW_COLUMNS:
                        sample[input_column] = sample[input_column].astype(np.float32)
                    key = f"{FAULT_NAMES.index(fault_name)+1}_{band}"
                    path = output_dir / f"rf_sample_{key}.parquet"
                    arrow = pa.Table.from_pandas(sample, preserve_index=False)
                    if key not in writers:
                        writers[key] = pq.ParquetWriter(path, arrow.schema, compression="zstd")
                    writers[key].write_table(arrow)
            row_start += n
        for start, end, event in windows:
            sid = event["sequence_id"]
            parts = window_parts[sid]
            if not parts:
                sequence_records.append({**event.to_dict(), "status": "no_complete_rows", "final_length": 0})
                continue
            tail = pd.concat(parts, ignore_index=True)
            gaps = np.diff(tail["time"].to_numpy(dtype=np.int64))
            breaks = np.flatnonzero(gaps > gap_seconds)
            if breaks.size:
                tail = tail.iloc[int(breaks[-1])+1:].reset_index(drop=True)
                counts["sequences_retruncated_after_missing"] += 1
            length = len(tail)
            status = "valid" if length > min_length else "removed_after_missing_or_gap"
            if status == "valid":
                seq_index = len(sequence_records)
                path = sequence_dir / f"sequence_{seq_index:04d}.npz"
                np.savez_compressed(
                    path,
                    x=encode_inputs(tail),
                    time=tail["time"].to_numpy(dtype=np.int64),
                    raw_row_index=tail["raw_row_index"].to_numpy(dtype=np.int64),
                    samples_to_fault=np.arange(length - 1, -1, -1, dtype=np.int32),
                )
                sequence_path = str(path.resolve().relative_to(project_root.resolve()))
            else:
                sequence_path = ""
            sequence_records.append({
                **event.to_dict(), "status": status, "final_length": length,
                "sequence_path": sequence_path,
                "removed_within_saved_tail": int(end - start + 1 - sum(len(p) for p in parts)),
            })
        window_rows.append(len(windows))
    for writer in writers.values():
        writer.close()
    sequence_table = pd.DataFrame(sequence_records)
    sequence_table.to_csv(output_dir / "sequence_metadata.csv", index=False)
    sensor_sample = np.concatenate(sample_numeric) if sample_numeric else np.empty((0, len(INPUT_RAW_COLUMNS)))
    distribution = {}
    for j, col in enumerate(INPUT_RAW_COLUMNS):
        vals = sensor_sample[:, j]
        distribution[col] = {"min": float(np.min(vals)), "median": float(np.median(vals)), "p99": float(np.quantile(vals, 0.99)), "max": float(np.max(vals))} if len(vals) else {}
    summary = {
        "missing": dict(counts), "column_missing": dict(column_missing),
        "shutter_counts": dict(shutter_counts), "input_distribution_0p02pct_sample": distribution,
        "rf_exact_band_counts_complete_rows": {name: dict(v) for name, v in rf_band_counts.items()},
        "rf_stored_band_counts": {name: dict(v) for name, v in rf_selected_counts.items()},
        "rf_storage_rates": rates,
        "gap_seconds": gap_seconds, "min_length_exclusive": min_length,
        "history_rows_saved": history_rows,
        "sequence_status_counts": {
            name: dict(Counter(sequence_table.loc[sequence_table["fault_name"].eq(name), "status"]))
            for name in FAULT_NAMES
        },
        "input_columns": INPUT_COLUMNS,
        "raw_input_columns": INPUT_RAW_COLUMNS,
    }
    (output_dir / "raw_input_extraction.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def audit_sequence_intervals(project_root: Path, output_dir: Path) -> dict:
    """Check the paper's 4-second sample proxy within retained sequences."""
    table = pd.read_csv(output_dir / "sequence_metadata.csv")
    records = []
    for _, row in table.loc[table["status"].eq("valid")].iterrows():
        with np.load(project_root / row["sequence_path"]) as data:
            times = data["time"]
            sample_count = data["samples_to_fault"]
        intervals = np.diff(times)
        if len(intervals) == 0 or not np.array_equal(sample_count, np.arange(len(sample_count)-1, -1, -1)):
            raise ValueError(f"Invalid samples-to-fault alignment for {row['sequence_id']}")
        local_median = float(np.median(intervals))
        records.append({
            "sequence_id": row["sequence_id"], "fault_name": row["fault_name"],
            "length": len(times), "median_interval_seconds": local_median,
            "p90_interval_seconds": float(np.quantile(intervals, 0.9)),
            "fraction_exact_4_seconds": float(np.mean(intervals == 4)),
            "last_observed_to_event_seconds": int(row["fault_time"] - times[-1]),
            "max_sample_rul_seconds_using_4": int(sample_count[0] * 4),
            "max_sample_rul_seconds_using_local_median": float(sample_count[0] * local_median),
        })
    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "sequence_interval_audit.csv", index=False)
    result = {
        "valid_sequence_count": len(frame),
        "sequence_median_interval_seconds": float(frame["median_interval_seconds"].median()),
        "sequences_with_median_4_seconds": int(frame["median_interval_seconds"].eq(4).sum()),
        "median_fraction_exact_4_seconds_within_sequence": float(frame["fraction_exact_4_seconds"].median()),
        "median_last_observation_to_fault_event_seconds": float(frame["last_observed_to_event_seconds"].median()),
        "max_last_observation_to_fault_event_seconds": int(frame["last_observed_to_event_seconds"].max()),
    }
    (output_dir / "sequence_interval_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def plot_pretraining_sequences(project_root: Path, output_dir: Path) -> Path:
    """Show two retained raw pressure histories per fault before model scoring."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = pd.read_csv(output_dir / "sequence_metadata.csv")
    figure, axes = plt.subplots(3, 2, figsize=(12, 9), squeeze=False)
    for fault_number, fault_name in enumerate(FAULT_NAMES):
        examples = table.loc[
            table["fault_name"].eq(fault_name) & table["status"].eq("valid")
        ].head(2)
        if len(examples) < 2:
            raise ValueError(f"Need two valid raw histories for {fault_name}")
        for example_number, (_, row) in enumerate(examples.iterrows()):
            with np.load(project_root / row["sequence_path"]) as data:
                remaining = data["samples_to_fault"]
                pressure = data["x"][:, SENSOR_COLUMNS.index("FLOWCOOLPRESSURE")]
            ax = axes[fault_number, example_number]
            ax.plot(remaining, pressure, linewidth=0.7)
            ax.invert_xaxis()
            ax.set_title(f"Fault {fault_number+1}, raw sequence {example_number+1}")
            ax.set_xlabel("samples to fault")
            ax.set_ylabel("normalized pressure")
    figure.tight_layout()
    path = output_dir / "pretraining_raw_sequences.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path
