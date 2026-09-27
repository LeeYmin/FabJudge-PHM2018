"""Write numeric 06e reports from completed, immutable stage outputs."""

import json
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
out = root / "artifacts/06e_fix_and_weakness_diagnosis"
fix = json.loads((out / "fix_summary.json").read_text(encoding="utf-8"))
selection = json.loads((out / "task_selection.json").read_text(encoding="utf-8"))
prereg = json.loads((out / "prereg_06e.json").read_text(encoding="utf-8"))
superseded = json.loads((out / "prereg_06e_v1_superseded.json").read_text(encoding="utf-8"))
rows = pd.read_parquet(out / "long_window_features.parquet")
weak = pd.read_csv(out / "weakness_map.csv")
dev = pd.read_csv(out / "dev_metric_comparison.csv").set_index("candidate")
test = pd.read_csv(out / "metric_comparison_06d_fix.csv").set_index("candidate")

counts = rows.groupby("fault").agg(rows=("sequence_id", "size"), sequences=("sequence_id", "nunique"),
                                    insufficient=("insufficient_history", "sum"))
summary = {
    "source_06d_config_sha256": fix["source_06d_config_sha256"],
    "frozen_config_06d_fix_sha256": fix["frozen_config_sha256"],
    "prereg_06e_sha256": prereg["sha256"],
    "superseded_prereg_06e_sha256": superseded["sha256"],
    "api_calls_part1": fix["new_jev_calls"], "api_calls_part2": 0,
    "api_calls_total": fix["new_jev_calls"],
    "actual_cost_usd": fix["actual_cost_usd"],
    "estimated_cost_usd_at_2_5e_5_per_call": fix["estimated_cost_cap_usd"],
    "test_06d_cache_requests_verified": fix["test_state_verified"],
    "dev_endpoint_rows": len(rows), "insufficient_history_rows": int(rows.insufficient_history.sum()),
    "sufficient_history_rows": int((~rows.insufficient_history).sum()),
    "fault_counts": counts.to_dict(orient="index"),
    "selection": selection,
    "limits": ["train LSTM predictions in sample", "fault3 validation has four sequences",
               "no fault3 test sequence used in part 2", "original PHM asymmetric cost unavailable"]}
(out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def metric_line(name):
    x = dev.loc[name]
    t = test.loc[name]
    return (f"| {name} | {x.original_dev_auc:.3f} [{x.original_dev_auc_ci_low:.3f}, {x.original_dev_auc_ci_high:.3f}] "
            f"| {x.corrected_dev_auc:.3f} [{x.corrected_dev_auc_ci_low:.3f}, {x.corrected_dev_auc_ci_high:.3f}] "
            f"| {x.original_rf_oof_spearman:.3f} → {x.corrected_rf_oof_spearman:.3f} "
            f"| {x.original_t_keep:.3f}/{x.original_t_low:.3f} → {x.corrected_t_keep:.3f}/{x.corrected_t_low:.3f} "
            f"| {bool(t.constraint_met)}; FP {int(t.removed_fp)}, TP {int(t.lost_tp)} |")

bands = weak[(weak.fault == "fault3") & (weak.unit == "row") & (weak.dimension == "ttf_band")]
band_lines = [f"| {r.group} | {r.n:,} | {r.lstm_error_median_seconds:,.0f} | {r.alarm_rate_5k:.3f} |"
              for r in bands.itertuples()]
machines = weak[(weak.unit == "row") & (weak.dimension == "machine")]
machine_lines = [f"| {r.fault} | {r.group} | {r.lstm_error_median_seconds:,.0f} | {r.rf_auc_5k:.3f} |"
                 for r in machines.itertuples()]
conclusion = f"""# 06e 결론 · 버그 정정과 긴 창 사전 검사

## 1. 정정 후 S2/J2

06d의 dev P2 요청은 `train_oof`의 in-sample RF를 포함했다. 수정 요청은 OOF RF를 포함한다. 06d test P2 23건의 저장된 RF 값과 캐시 요청 SHA-256을 모두 확인하고 재사용했다. 새 P2 요청은 {fix['new_jev_calls']}건, 실제 API 비용은 ${fix['actual_cost_usd']:.6f}다.

| 후보 | 06d dev AUC [95% CI] | 정정 dev AUC [95% CI] | RF OOF Spearman | t_keep/t_low: 기존 → 정정 | 정정 test 제약; FP 제거, TP 손실 |
|---|---:|---:|---:|---:|---:|
{metric_line('J2')}
{metric_line('S2')}

S2−B1 dev AUC 차이는 {fix['s2_auc_difference_vs_b1']['estimate']:.3f}, 시퀀스 bootstrap 95% CI [{fix['s2_auc_difference_vs_b1']['ci'][0]:.3f}, {fix['s2_auc_difference_vs_b1']['ci'][1]:.3f}]이다. 사전 등록 결합 효과 판정은 06d와 같이 **불충족**이다. 정정 후 test에서 S2/J2 모두 FP 제거 0건, TP 손실 0건이다. 다른 후보 B1/B3/S3/J3 수치는 06d 그대로다.

## 2. LSTM/RF 약점 지도

04b 입력은 최대 300행, 15개 원본행 간격이며 대표 샘플 간격 4초 기준 약 17,940초다. 다른 fault의 LSTM/RF artifact도 확인하여 fault1·fault2·fault3 train/validation 총 {len(rows):,} causal endpoint를 분석했다. 아래 LSTM 오차는 예측초−실제 TTF초이며 음수는 이른 예측이다.

| fault3 TTF 구간 | 행 | LSTM 오차 중앙값(초) | 5k 경보율 |
|---|---:|---:|---:|
{chr(10).join(band_lines)}

| fault | 장비 | LSTM 오차 중앙값(초) | RF OOF 5k AUC |
|---|---|---:|---:|
{chr(10).join(machine_lines)}

fault3에서는 M01의 RF AUC가 M02보다 낮다(0.497 대 0.583). fault1/2의 LSTM은 두 장비에서 전체 endpoint의 5k 경보율이 1.000으로, 이른 경보가 심하다. fault3는 5–10k 구간에서도 5k 경보율이 0.473이다. 구간별 RF AUC는 각 TTF 구간에 5k 양성 또는 음성 한 클래스만 들어가므로 정의되지 않는다. 장비·출처별 AUC는 `weakness_map.csv`에 기록했다. PHM 비대칭 비용은 확정된 적용 식이 없어 미적용했다.

## 3. 긴 창 정보와 06f

20,000초 baseline과 10,000초 과거 창의 이력을 모두 요구하므로 {len(rows):,}행 중 {int(rows.insufficient_history.sum()):,}행은 부족하다. 충분한 행은 {int((~rows.insufficient_history).sum())}행: fault1 195행(6시퀀스), fault2 83행(3시퀀스), fault3 4행(1시퀀스)이다. 사전 등록한 클래스별 최소 5시퀀스 규칙에 따라 T1–T4를 모두 제외했다. **긴 창의 추가 신호를 검증할 수 없었으므로 06f 과제와 기준선은 정하지 않는다.** 다음 탐색 후보는 이력 요구량에 맞는 더 짧은 baseline/창 feature군 또는 더 긴 시퀀스 보존 설정이다. 이들은 새 실험으로 사전 등록해야 한다.

한계: train LSTM은 in-sample, fault3 validation은 4시퀀스다. 2부의 과제 선택에는 fault3 test 8시퀀스를 사용하지 않았다. 1부 test는 06d 규칙으로 한 번 평가했다.
"""
(out / "conclusion.md").write_text(conclusion, encoding="utf-8")

trial_log = f"""# 06e trial log

1. 06d 입력 버그 확인: dev P2 state가 `rf_probability`를 읽음. 기본 동작을 유지하면서 `rf_field` 인자를 추가하고 단위 테스트 3개를 통과시킴.
2. 06d test P2 23건의 v2 state RF와 원래 캐시 SHA-256 확인: 23/23 일치. 실패 0건.
3. dev P2 145건을 OOF RF로 재요청: 성공 145/145, 첫 응답 스키마 통과, 상한 초과 0건. 수정 캐시 재생 확인: 추가 API 호출 0건.
4. 기존 후보/임계값/판정 규칙으로 S2/J2 재적합. frozen SHA-256 `{fix['frozen_config_sha256']}`. test는 원래 P2 응답을 재사용해 한 번 평가. S2 결합 효과 불충족.
5. 2부 후보·feature·선택 규칙을 계산 전 `prereg_06e.json`에 기록. SHA-256 `{prereg['sha256']}`. fault1/2/3 artifact 확인. LSTM 재학습 0회, JEV 호출 0회.
6. 초기 v1 등록본 SHA-256 `{superseded['sha256']}`은 보존했다. 긴 창 residual에 04b 전체 학습 관계 모델을 사용하면 train held 시퀀스가 관계 모델 적합에 참여하는 문제를 검토 중 발견했다. 과제/선택 규칙은 유지하고 residual만 RF와 같은 시퀀스 OOF 관계 모델로 바꾸는 v2를 계산 전에 등록했다. 모든 feature와 표를 다시 계산했다.
7. fault1/2/3 train·validation {len(rows):,}개 causal endpoint, 시퀀스 단위 RF OOF, 긴 창 feature를 계산. test 시퀀스는 제외. 첫 prescreen 실행은 빈 결과표의 열 정의 누락으로 보고 단계에서 실패했고, 열 정의를 수정했다. API 추가 호출 0회.
8. T1–T4 모두 클래스별 최소 5시퀀스 조건 미달로 제외. 시도별 상세 사유는 `excluded_tasks.csv`; AUC 추정은 수행하지 않음.

최종 API 호출 수 {fix['new_jev_calls']}건, 관측 비용 ${fix['actual_cost_usd']:.6f}. 사전 등록/동결 해시 불일치 없음.
"""
(out / "trial_log.md").write_text(trial_log, encoding="utf-8")

erratum = f"""# 06d erratum

06d의 dev P2 호출에서 `request_spec`은 `rf_probability`를 읽었고, `stage2_calls`는 train OOF 호출 전에 이를 `rf_probability_oof`로 교체하지 않았다. 따라서 train dev P2 요청에는 in-sample RF가 들어갔다. J2/S2의 dev AUC와 이 값으로 적합한 계수·임계값은 영향을 받았다. P3/J3, B1/B3, 저장된 out-of-sample RF를 사용한 test P2 요청은 영향이 없었다.

정정은 [06e 결과](../06e_fix_and_weakness_diagnosis/conclusion.md)에 있다. dev P2 145건을 다시 호출했고, 06d test P2 요청 23건의 RF state와 캐시를 검증한 뒤 재사용했다. J2 dev AUC {dev.loc['J2'].original_dev_auc:.3f}→{dev.loc['J2'].corrected_dev_auc:.3f}, S2 {dev.loc['S2'].original_dev_auc:.3f}→{dev.loc['S2'].corrected_dev_auc:.3f}. S2 결합 효과 결론은 여전히 불충족이다. 정정 설정 SHA-256: `{fix['frozen_config_sha256']}`.
"""
(root / "artifacts/06d_lstm_jev_oof_edge/ERRATUM.md").write_text(erratum, encoding="utf-8")
print("Wrote summary, conclusion, trial log, and 06d erratum")
