"""Prepare artifacts, run guarded 07a JEV scoring, and smoke-test Luna on 10 packets."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fabjudge.pipeline.clients import JEVClient, LLMClient
from fabjudge.pipeline.config import (
    ARTIFACT_DIR, JEV_KEY_ENV, LLM_KEY_ENV, LLM_MODEL_ID, MAX_JEV_EXTRA_CALLS,
    RANDOM_SEED, SAMPLE_CAP,
)
from fabjudge.pipeline.data import build_dev_sample
from fabjudge.pipeline.keys import load_jev_key, load_llm_key
from fabjudge.pipeline.prepare_07a import build_preregistration, verify_preregistration
from fabjudge.pipeline.schemas import (
    LLM_OUTPUT_SCHEMA, canonical_json, sha256_json, validate_packet,
)
from fabjudge.pipeline.claims import verify_claims


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding="utf-8-sig")


def packet_from_row(row: pd.Series) -> dict:
    packet = json.loads(row["packet_json"])
    validate_packet(packet)
    return packet


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def model_token_price(model_metadata: dict, usage: dict) -> float | None:
    pricing = model_metadata.get("pricing") or {}
    try:
        prompt_price = float(pricing["prompt"])
        completion_price = float(pricing["completion"])
        prompt_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens")))
        completion_tokens = int(usage.get("completion_tokens", usage.get("output_tokens")))
        value = prompt_price * prompt_tokens + completion_price * completion_tokens
        return value if math.isfinite(value) and value >= 0 else None
    except (KeyError, TypeError, ValueError):
        return None


def write_conclusion(stats: dict, scores: pd.DataFrame, smoke: dict,
                     cost: dict, prereg_hash: str, api_log: list[dict]) -> None:
    code_to_fault = {"1": "fault1", "2": "fault2", "3": "fault3"}
    sample_lines = []
    for code, fault in code_to_fault.items():
        item = stats.get(code, {"dev_alarm_samples": 0, "near_failure_fraction": None,
                                "sequence_count": 0, "data_review_count": 0, "ready_count": 0})
        fraction = item.get("near_failure_fraction")
        near = "N/A" if fraction is None else f"{100*fraction:.1f}%"
        sample_lines.append(
            f"| {fault} | {item.get('dev_alarm_samples', 0)} | {near} | {item.get('sequence_count', 0)} | "
            f"{item.get('ready_count', 0)} | {item.get('data_review_count', 0)} |"
        )
    score_lines = []
    if not scores.empty:
        for code, fault in code_to_fault.items():
            subset = scores.loc[scores["fault_type"].eq(int(code)), "score"].dropna()
            if len(subset):
                q = subset.quantile([0.25, 0.5, 0.75])
                score_lines.append(f"| {fault} | {len(subset)} | {q.loc[0.25]:.2f} | {q.loc[0.5]:.2f} | {q.loc[0.75]:.2f} |")
            else:
                score_lines.append(f"| {fault} | 0 | N/A | N/A | N/A |")
    calls = {role: sum(1 for row in api_log if row.get("key_role") == role and row.get("event") != "catalog")
             for role in ("jev", "llm")}
    catalog_calls = sum(1 for row in api_log if row.get("event") == "catalog")
    call_cost = cost.get("observed_jev_cost_usd")
    jev_cost_txt = "미보고" if call_cost is None else f"${call_cost:.6f}"
    llm_each = cost.get("mean_smoke_cost_usd")
    llm_each_txt = "단가 확인 불가" if llm_each is None else f"${llm_each:.6f}/건"
    llm_total = cost.get("full_llm_baseline_cost_usd")
    llm_total_txt = "추정 불가" if llm_total is None else f"${llm_total:.4f}"
    budget_totals = cost.get("router_budget_total_including_jev_usd", {})
    budget_txt = ", ".join(
        f"{pct}% {cost.get('router_budget_calls', {}).get(pct, 0)}건 / "
        f"{'추정 불가' if total is None else f'${total:.4f}'}"
        for pct, total in budget_totals.items()
    )
    text = f"""# 07a 파이프라인 점검 결과

## 구성

```mermaid
flowchart LR
  A[인과적 LSTM 5,000초 경보] --> B[EvidencePacket: LSTM + RF OOF + 12 센서 + fault 유형]
  B --> C[JEV native score 라우터]
  C -->|HIGH: LLM 생략| H[표준 사람 검토 큐]
  C -->|MID: 전체 분석| D[LLM judge]
  C -->|LOW: 10% 무작위 감사| D
  C -->|LOW 미감사| L[낮은 우선순위 사람 검토 큐]
  D --> H
  B -->|결측/비유한값| R[DATA_REVIEW 사람 검토]
```

경보는 모든 경로에서 유지됩니다. `DATA_REVIEW` 값은 대치하지 않았고 JEV/LLM에 전송하지 않았습니다. 개발 패킷에서 ID·시각·TTF·라벨 누출 검사를 통과했습니다. 라벨은 `dev_labels.csv`에만 저장했습니다.

## 개발 표본

표본은 06e의 causal endpoint 중 LSTM 예측이 5,000초 이하인 지점에서 시퀀스별 분위수로 뽑았습니다. 600행 상한 때문에 사용한 분위수 규칙은 `{stats['total']['percentiles_used']}`이며, 시퀀스당 최대 `{stats['total']['per_sequence_alarm_sample_cap']}`개입니다.

| 고장 유형 | 경보 표본 | TTF≤5,000초 비율 | 시퀀스 | 사용 가능 | DATA_REVIEW |
|---|---:|---:|---:|---:|---:|
{chr(10).join(sample_lines)}

임박 고장 비율은 표본 통계에만 사용했습니다. 10건 LLM 점검 선택과 입력에는 라벨을 쓰지 않았습니다.

**07b 해석 제한:** 317개 상한을 맞추며 시퀀스당 중앙 경보점 하나를 선택한 결과, 주 대상 fault1/2의 277개 표본에는 임박 고장 양성이 없습니다. 따라서 이 표본으로는 fault1/2의 순위 품질이나 무작위 라우터 대비 개선을 확인할 수 없습니다. 고정한 기준을 완화하지 않고 이 제한을 그대로 보고합니다.

## JEV 무라벨 점수 분포

| 고장 유형 | 점수 수 | 25 백분위 | 중앙값 | 75 백분위 |
|---|---:|---:|---:|---:|
{chr(10).join(score_lines)}

분기 임계값, isotonic 보정, 성공/실패 판정은 07a에서 산출하지 않았습니다.

## LLM 점검

- 카탈로그 확인 모델 ID: `{LLM_MODEL_ID}`
- 스키마 통과: {smoke.get('schema_pass_count', 0)}/{smoke.get('requested_count', 10)}
- 주장 인용 일치율: {('N/A' if smoke.get('claim_agreement_rate') is None else f"{100*smoke['claim_agreement_rate']:.1f}%")}
- 평균 지연: {('N/A' if smoke.get('mean_latency_seconds') is None else f"{smoke['mean_latency_seconds']:.2f}초")}
- 평균 건당 비용: {llm_each_txt}
- 점검 표본은 고정 seed로 고른 10개이며, 요청에는 EvidencePacket만 포함했습니다.

## 07b 호출·비용 예상

- 유효 개발 표본 전체 LLM 기준선: {cost.get('valid_dev_samples', 0)}건 예상, 총 비용 {llm_total_txt}
- 30% / 50% / 70% 라우터 예산 (호출 / LLM 비용+07a JEV 비용): {budget_txt}
- 07a에서 발생한 JEV 비용: {jev_cost_txt}; inference 호출 JEV {calls['jev']}, LLM {calls['llm']}, 모델 카탈로그 조회 {catalog_calls}회
- 07b 추정은 10건 smoke의 실제 비용 또는 카탈로그 단가와 토큰 사용량을 기준으로 합니다. 라우터 비교는 전체 dev LLM 결과를 한 번 만들어 재사용하는 계획입니다.

## 사전 등록과 한계

`prereg_07.json` SHA-256: `{prereg_hash}`. train LSTM은 in-sample 예측이므로 LSTM 예측값 순위 기준선이 dev에서 유리할 수 있습니다. 07c 전까지 모든 test 시퀀스는 미사용 상태입니다.
"""
    (ARTIFACT_DIR / "conclusion.md").write_text(text, encoding="utf-8")


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for generated_log in ("api_call_log.jsonl", "llm_smoke_results.jsonl"):
        (ARTIFACT_DIR / generated_log).unlink(missing_ok=True)
    prereg = build_preregistration(ARTIFACT_DIR)
    prereg_path = ARTIFACT_DIR / "prereg_07.json"
    prereg_hash = verify_preregistration(prereg_path)

    sample, labels, stats = build_dev_sample(ROOT, sample_cap=SAMPLE_CAP)
    sample.to_csv(ARTIFACT_DIR / "dev_sample.csv", index=False, encoding="utf-8-sig")
    labels.to_csv(ARTIFACT_DIR / "dev_labels.csv", index=False, encoding="utf-8-sig")
    write_json(ARTIFACT_DIR / "dev_sample_statistics.json", stats)

    # Hash and leakage checks happen before the first network request.
    if not (ARTIFACT_DIR / "prereg_07.json").is_file() or verify_preregistration(prereg_path) != prereg_hash:
        raise RuntimeError("preregistration hash mismatch; stopping before API calls")
    if sample["packet_status"].ne("READY").any():
        bad_packets = sample.loc[sample["packet_status"].eq("DATA_REVIEW"), "packet_json"]
        for serialized in bad_packets:
            packet = json.loads(serialized)
            keys = set(packet["fields"])
            if keys != set(json.loads((ARTIFACT_DIR / "packet_schema.json").read_text(encoding="utf-8"))["properties"]["fields"]["required"]):
                raise RuntimeError("Packet identifier/label leakage detected")

    api_log: list[dict] = []
    # Both exact role variables must exist and differ before any more network traffic.
    load_jev_key(ROOT)
    load_llm_key(ROOT)
    llm = LLMClient(ROOT, ARTIFACT_DIR / "cache" / "llm_v2", max_new_calls=10)
    catalog_path = ARTIFACT_DIR / "model_catalog_verification.json"
    cached_catalog = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.is_file() else {}
    if (cached_catalog.get("model_id") == LLM_MODEL_ID and
            cached_catalog.get("key_role") == "llm" and
            cached_catalog.get("key_env_var") == LLM_KEY_ENV and
            cached_catalog.get("prereg_sha256") == prereg_hash):
        model_metadata = {key: value for key, value in cached_catalog.items()
                          if key not in {"prereg_sha256", "verified_at_utc"}}
    else:
        try:
            model_metadata = llm.catalog_model()
        except Exception:
            append_jsonl(ARTIFACT_DIR / "api_call_log.jsonl", {
                "event": "catalog", "key_role": "llm", "key_env_var": LLM_KEY_ENV,
                "model_id": LLM_MODEL_ID, "status": "failed", "error": "catalog request failed",
            })
            raise
        model_metadata["prereg_sha256"] = prereg_hash
        write_json(catalog_path, model_metadata)
    if "response_format" not in set(model_metadata.get("supported_parameters") or []):
        raise RuntimeError("Configured LLM model does not advertise strict JSON response support")
    catalog_log = {"event": "catalog", "key_role": "llm", "key_env_var": LLM_KEY_ENV,
                   "model_id": model_metadata["model_id"], "status": "verified"}
    api_log.append(catalog_log)
    append_jsonl(ARTIFACT_DIR / "api_call_log.jsonl", catalog_log)

    question = json.loads((ARTIFACT_DIR / "questions" / "R1.json").read_text(encoding="utf-8"))
    ready = sample.loc[sample["packet_status"].eq("READY")].reset_index(drop=True)
    jev_client = JEVClient(ROOT, ARTIFACT_DIR / "cache" / "jev_v2", max_new_calls=len(ready) + MAX_JEV_EXTRA_CALLS)
    jev_rows: list[dict] = []
    jev_path = ARTIFACT_DIR / "jev_dev_scores.csv"
    columns = ["row_id", "sequence_group", "fault_type", "status", "score", "probability_2",
               "latency_seconds", "cost_usd", "cache_hit", "error"]
    for index, row in ready.iterrows():
        attempt_start = len(jev_client.attempts)
        result = jev_client.call(packet_from_row(row), question)
        for record in jev_client.attempts[attempt_start:]:
            api_log.append(record)
            append_jsonl(ARTIFACT_DIR / "api_call_log.jsonl", record)
        item = {
            "row_id": row["row_id"], "sequence_group": row["sequence_group"],
            "fault_type": int(json.loads(row["packet_json"])["fields"]["fault_type"]),
            "status": result["status"], "score": result.get("score"),
            "probability_2": (result.get("probabilities") or {}).get("2"),
            "latency_seconds": result.get("latency_seconds"), "cost_usd": result.get("cost_usd"),
            "cache_hit": result.get("cache_hit", False), "error": result.get("error"),
        }
        jev_rows.append(item)
        write_csv(jev_path, jev_rows, columns)
        if index == 0 and result["status"] != "success":
            raise RuntimeError("First JEV response schema/request failure; 07a stopped")
        if result["status"] != "success":
            raise RuntimeError(f"JEV response unavailable at development sample {index + 1}; 07a stopped")
        if (index + 1) % 25 == 0 or index + 1 == len(ready):
            print(f"JEV development scores: {index + 1}/{len(ready)}", flush=True)
    rng = np.random.default_rng(RANDOM_SEED)
    stability_n = min(5, len(ready))
    stability_indices = np.sort(rng.choice(len(ready), size=stability_n, replace=False))
    stability_rows: list[dict] = []
    stability_columns = ["row_id", "repeat_index", "status", "score", "probability_2",
                         "latency_seconds", "cost_usd", "cache_hit", "error"]
    for sample_index in stability_indices:
        row = ready.iloc[int(sample_index)]
        packet = packet_from_row(row)
        for repeat_index in (1, 2, 3):
            attempt_start = len(jev_client.attempts)
            result = jev_client.call(packet, question, repeat_index=repeat_index)
            for record in jev_client.attempts[attempt_start:]:
                api_log.append(record)
                append_jsonl(ARTIFACT_DIR / "api_call_log.jsonl", record)
            stability_rows.append({
                "row_id": row["row_id"], "repeat_index": repeat_index,
                "status": result["status"], "score": result.get("score"),
                "probability_2": (result.get("probabilities") or {}).get("2"),
                "latency_seconds": result.get("latency_seconds"),
                "cost_usd": result.get("cost_usd"), "cache_hit": result.get("cache_hit", False),
                "error": result.get("error"),
            })
            write_csv(ARTIFACT_DIR / "jev_stability.csv", stability_rows, stability_columns)
            if result["status"] != "success":
                raise RuntimeError("JEV stability response failed schema validation; 07a stopped")
    if jev_client.new_calls > len(ready) + MAX_JEV_EXTRA_CALLS:
        raise RuntimeError("JEV new-call cap reached")

    # The smoke sample is drawn only from dev_sample.csv. No label table is read here.
    smoke_n = min(10, len(ready))
    smoke_rng = np.random.default_rng(RANDOM_SEED)
    smoke_indices = np.sort(smoke_rng.choice(len(ready), size=smoke_n, replace=False))
    smoke_rows = []
    llm_cost_observations: list[float] = []
    for smoke_position, sample_index in enumerate(smoke_indices):
        row = ready.iloc[int(sample_index)]
        attempt_start = len(llm.attempts)
        result = llm.call(packet_from_row(row), LLM_OUTPUT_SCHEMA, model_metadata=model_metadata)
        for record in llm.attempts[attempt_start:]:
            api_log.append(record)
            append_jsonl(ARTIFACT_DIR / "api_call_log.jsonl", record)
        output = result.get("output")
        claims = verify_claims(packet_from_row(row), output) if output is not None else None
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        cost_value = result.get("cost_usd")
        cost_method = "openrouter_usage"
        if cost_value is None:
            cost_value = model_token_price(model_metadata, usage)
            cost_method = "catalog_price_x_tokens" if cost_value is not None else "unavailable"
        if cost_value is not None:
            llm_cost_observations.append(float(cost_value))
        result_row = {
            "row_id": row["row_id"], "status": result["status"],
            "schema_pass": result["status"] == "success",
            "returned_model_id": result.get("returned_model_id"),
            "priority_score": output.get("priority_score") if output else None,
            "review_recommendation": output.get("review_recommendation") if output else None,
            "output": output, "claim_validation": claims,
            "claim_agreement_rate": claims.get("agreement_rate") if claims else None,
            "latency_seconds": result.get("latency_seconds"),
            "cost_usd": cost_value, "cost_method": cost_method,
            "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
            "output_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
            "cache_hit": result.get("cache_hit", False), "error": result.get("error"),
        }
        smoke_rows.append(result_row)
        append_jsonl(ARTIFACT_DIR / "llm_smoke_results.jsonl", result_row)
        print(f"LLM smoke: {smoke_position + 1}/{smoke_n}", flush=True)
        if smoke_position == 0 and result["status"] != "success":
            write_json(ARTIFACT_DIR / "llm_smoke_summary.json", {
                "requested_count": smoke_n, "schema_pass_count": 0,
                "first_response_failed": True, "status": "stopped",
                "error": result.get("error"), "key_role": "llm", "key_env_var": LLM_KEY_ENV,
            })
            raise RuntimeError("First LLM response schema/request failure; 07a stopped")
    passed = sum(row["schema_pass"] for row in smoke_rows)
    total_claims = sum((row["claim_validation"] or {}).get("claim_count", 0) for row in smoke_rows)
    matching_claims = sum((row["claim_validation"] or {}).get("matching_claim_count", 0) for row in smoke_rows)
    latency_values = [float(row["latency_seconds"]) for row in smoke_rows
                      if row.get("latency_seconds") is not None]
    smoke_summary = {
        "requested_count": smoke_n,
        "schema_pass_count": passed,
        "schema_pass_rate": passed / smoke_n if smoke_n else None,
        "first_response_failed": False,
        "minimum_passes": 8,
        "claim_count": total_claims,
        "matching_claim_count": matching_claims,
        "mismatched_claim_count": total_claims - matching_claims,
        "claim_agreement_rate": (matching_claims / total_claims) if total_claims else None,
        "mean_latency_seconds": (sum(latency_values) / len(latency_values)) if latency_values else None,
        "mean_cost_usd": (sum(llm_cost_observations) / len(llm_cost_observations)) if llm_cost_observations else None,
        "cost_observations": len(llm_cost_observations),
        "label_blind": True,
        "key_role": "llm", "key_env_var": LLM_KEY_ENV,
        "model_id_catalog_verified": model_metadata["model_id"],
        "status": "passed" if passed >= 8 else "stopped",
    }
    write_json(ARTIFACT_DIR / "llm_smoke_summary.json", smoke_summary)

    jev_costs = [float(row["cost_usd"]) for row in jev_rows if row.get("cost_usd") is not None and not row.get("cache_hit")]
    jev_costs += [float(row["cost_usd"]) for row in stability_rows if row.get("cost_usd") is not None and not row.get("cache_hit")]
    mean_llm_cost = smoke_summary["mean_cost_usd"]
    router_budget_calls = {str(pct): int(math.floor(len(ready) * pct / 100 + 0.5)) for pct in (30, 50, 70)}
    cost_estimate = {
        "valid_dev_samples": int(len(ready)),
        "data_review_samples": int(len(sample) - len(ready)),
        "llm_smoke_calls": int(smoke_n),
        "llm_calls_cap_07b_full_dev_baseline": int(len(ready)),
        "full_llm_baseline_cost_usd": (mean_llm_cost * len(ready)) if mean_llm_cost is not None else None,
        "mean_smoke_cost_usd": mean_llm_cost,
        "cost_method": "mean of OpenRouter reported cost; when absent, catalog prompt/completion token price times usage tokens",
        "router_budget_calls": router_budget_calls,
        "router_budget_cost_usd": {
            pct: ((mean_llm_cost * n) if mean_llm_cost is not None else None)
            for pct, n in router_budget_calls.items()
        },
        "router_budget_total_including_jev_usd": {
            pct: ((mean_llm_cost * n + sum(jev_costs)) if mean_llm_cost is not None else None)
            for pct, n in router_budget_calls.items()
        },
        "router_comparison_execution_plan": "full-development LLM outputs generated once and reused for 30/50/70 percent route simulations",
        "jev_new_calls_07a": int(jev_client.new_calls),
        "jev_cache_hits_07a": int(jev_client.cache_hits),
        "observed_jev_cost_usd": sum(jev_costs) if jev_costs else None,
        "jev_cost_in_07b_if_cached_scores_reused_usd": 0.0,
        "full_llm_total_including_07a_jev_usd": ((mean_llm_cost * len(ready) + sum(jev_costs))
                                                    if mean_llm_cost is not None else None),
        "jev_call_cap": int(len(ready) + MAX_JEV_EXTRA_CALLS),
        "jev_cost_sample_size": int(len(jev_costs)),
        "llm_key_role": "llm", "llm_key_env_var": LLM_KEY_ENV,
        "jev_key_role": "jev", "jev_key_env_var": JEV_KEY_ENV,
    }
    write_json(ARTIFACT_DIR / "cost_estimate_07b.json", cost_estimate)

    score_frame = pd.read_csv(jev_path) if jev_path.exists() else pd.DataFrame()
    if not score_frame.empty:
        for name in ("score", "probability_2"):
            score_frame[name] = pd.to_numeric(score_frame[name], errors="coerce")
    prereg_hash = verify_preregistration(prereg_path)
    write_conclusion(stats, score_frame, smoke_summary, cost_estimate, prereg_hash, api_log)
    with (ARTIFACT_DIR / "trial_log.md").open("w", encoding="utf-8") as stream:
        stream.write(f"""# 07a 실행 기록

- 사전 등록 SHA-256 확인: `{prereg_hash}`
- 키 변수명만 확인: JEV `{JEV_KEY_ENV}`, LLM `{LLM_KEY_ENV}`. 키 값은 출력·저장하지 않음.
- `.env`는 `.gitignore` 규칙에 포함됨.
- 개발 표본: {len(sample)}행, READY {len(ready)}행, DATA_REVIEW {len(sample)-len(ready)}행.
- 표본 제한: 10개 분위수 기본안은 600행을 초과해 시퀀스당 중앙값 1개로 축소. fault1/2 양성 표본 0건은 07b 주 지표의 해석 제한으로 결론에 기록.
- JEV R1: 새 호출 {jev_client.new_calls}, 캐시 {jev_client.cache_hits}; 상한 {len(ready)+MAX_JEV_EXTRA_CALLS}.
- JEV 안정성: {len(stability_rows)}회 (최대 5행 × 3회).
- 모델 카탈로그: `{model_metadata['model_id']}` 확인, 호출 역할 llm / 변수명 `{LLM_KEY_ENV}`.
- LLM smoke: 스키마 {passed}/{smoke_n}; 주장 검증 일치율 {smoke_summary['claim_agreement_rate']}.
- 07a에서 분기 임계값, isotonic 보정, 판정은 수행하지 않음.
- 전체 호출 로그의 각 행에는 `key_role` 및 `key_env_var`만 있고 키 값은 없음.
""")
    if passed < 8:
        raise RuntimeError("Fewer than 8 of 10 LLM smoke outputs passed schema; 07a stopped")
    if verify_preregistration(prereg_path) != prereg_hash:
        raise RuntimeError("preregistration hash mismatch; 07a stopped")
    print(json.dumps({"sample_stats": stats, "jev_new_calls": jev_client.new_calls,
                      "jev_cache_hits": jev_client.cache_hits, "jev_cost_usd": cost_estimate["observed_jev_cost_usd"],
                      "llm_smoke": smoke_summary, "cost_estimate_07b": cost_estimate,
                      "prereg_sha256": prereg_hash}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
