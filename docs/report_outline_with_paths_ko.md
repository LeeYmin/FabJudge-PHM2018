# 경보를 버리지 않는 설계: PHM2018에서 RF 게이트의 누락과 LLM 판단 계층의 역할

부제: 학습된 소형 모델 대비 우월성을 확인하지 못한 LLM·JEV는 어디에 둘 것인가

이 문서는 보고서 집필용 구성안이다. 원안의 장 구성과 분량을 유지하면서 실제 프로젝트 파일, 표·그림, 수치의 적용 범위를 연결했다. 경로는 현재 프로젝트의 절대 경로이며, 별도 표시가 없는 링크는 실재 파일이다. 모든 결과는 저장된 실험에 한정한다. API 비용은 해당 실행 기록이며 현재 요금이나 모든 개별 호출의 상한을 뜻하지 않는다.

## 0. 요약 — 반 쪽

- **질문:** Huang 2018 재현 파이프라인(04b, RF 게이트 + LSTM)에 JEV와 LLM 판단 계층을 붙이면 경보의 판별 또는 우선순위가 개선되는가.
- **핵심 발견:** RF 게이트는 LSTM이 낸 실제 임박 고장 경보 일부를 제거했다. fault3의 endpoint recall은 04b 0.6288, RF 게이트 없는 LSTM 0.9742였다. fault1+2에서는 평가 가능한 고장 시퀀스 52개 중 04b가 46개를 탐지했고, LSTM 경보를 모두 유지하면 52개를 탐지했다.
- **판단 계층:** 사전 등록 H1에서 우선순위 개선을 확인하지 못했다. H2·H3는 순차 규칙상 미검정이다. fault1+2 rescue의 탐색 AUC에서는 JEV·LLM이 RF·LSTM보다 낮았고, E20 precision에서는 로지스틱보다 낮았다. 모든 지표에서 모든 소형 모델에 열세였다는 주장은 하지 않는다.
- **남는 역할:** 평균 LLM 비용 약 $0.00026/경보, 재시도 후 유효 응답 270/271(99.6%), 성공 응답의 인용 수치 1,706개 모두 검증 통과. 경보 취소 권한 없는 참고 계층의 운영 가능성을 제시한다. 사람의 판단 정확도나 검토 시간 개선은 측정하지 않았다.

**근거:** [T5 전체 endpoint 비교](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/tables/T5_full_endpoint_comparison.csv), [T6 사건 단위 집계](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/tables/T6_sequence_event_detection.csv), [07b 확증 검정](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/hypothesis_tests.json), [07b 실행 요약](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/summary.json), [07b LLM 응답·인용 검증 기록](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/llm_noRF_results.jsonl).

## 1. 배경과 질문 — 1쪽

- PHM2018 이온 밀링 장비의 fault1/2/3를 대상으로 5,000초 이내 고장을 예측한다. 데이터 구조와 분석 단위는 [01 데이터 구조 노트북](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/notebooks/01_data_structure.ipynb), 시퀀스 구성은 [sequence_metadata.csv](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/huang2018/sequence_metadata.csv), 분할은 [split_metadata.json](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/huang2018/split_metadata.json)을 따른다.
- 04b는 RF 확률과 LSTM의 잔여시간 예측을 결합한다. 본 보고서의 경보 정의는 `LSTM 예측 ≤ 5,000초 AND RF 확률 ≥ 0.5`다. RF를 단순한 현재 상태 이상 판별기로만 설명하지 않고, 5k fault와 >50k normal을 구분하도록 학습된 분류기라는 맥락을 함께 적는다.
- Q1. LLM 계열 판단 계층이 경보의 판별이나 우선순위를 개선하는가?
- Q2. 우월성을 확인하지 못했다면 파이프라인에서 어떤 역할이 합리적인가?

**구현·실험 출처:** [04b 재현 노트북](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/notebooks/04b_huang2018_reproduction.ipynb), [04b 모델 구현](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/src/fabjudge/huang2018_models.py), [04b 원래 지표와 정의 T5b](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/tables/T5b_04b_original_metrics.csv).

## 2. 시스템과 평가 설계 — 1~1.5쪽

### 2.1 구조

**그림 S1:** 04b와 경보 보존 구조를 비교한다. 새 구조도 원본은 [report_pipeline_structure.mmd](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/docs/figures/report_pipeline_structure.mmd)이다. 기존 F1~F7의 번호는 변경하지 않는다.

- tier1: 04b가 유지한 LSTM 경보. 즉시 검토 대상으로 유지한다.
- rescue: RF가 탈락시킨 LSTM 경보. JEV의 HIGH/MID/LOW 분기와 필요 시 LLM 분석을 거쳐 사람의 검토 대기열에 유지한다.
- 판단 계층은 순서와 분석 깊이만 바꾸며, 경보를 취소하거나 자동 인터록을 결정하지 않는다.
- **실험과 운영안 구분:** 07b에서는 비교를 위해 rescue 271건 모두에 LLM을 호출했다. MID에만 LLM을 붙이는 136건 시나리오는 저장 응답을 재사용한 환산 비교다. 모든 rescue에 참고 설명을 붙이는 방안은 별도의 운영 선택지다.

**정확한 실행 규칙:** [07b 사전 등록 v2.1](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/prereg_07_v2_1.json)의 `tier`, `routing`, `review_budgets` 및 [07b 실행 노트북](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/notebooks/07b_rescue_lane_confirmatory.ipynb). 공용 [routing.py](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/src/fabjudge/pipeline/routing.py)의 초기 07a 라우팅과 07b의 fault별 정수 예산 규칙을 혼동하지 않는다.

### 2.2 입력·출력 및 누수 방지

- JEV·LLM의 API 키 역할을 분리한다. 실제 키 값은 문서에 포함하지 않는다. [키 분리 구현](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/src/fabjudge/pipeline/keys.py), [클라이언트 구현](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/src/fabjudge/pipeline/clients.py).
- EvidencePacket에 정답 TTF, 양성 라벨, 파일·시퀀스 식별자, 시각, 표본 위치를 넣지 않는다. `fault_type`은 포함되지만 임박 고장 정답이 아닌 fault family 조건이다.
- **07a와 07b를 구분:** [07a packet_schema.json](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07a_pipeline_skeleton/packet_schema.json)은 RF 확률을 포함한다. 최종 07b는 `07a-evidence-v1-noRF` 16개 필드로 RF 확률을 제외한다. [v2.1 사전 등록](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/prereg_07_v2_1.json), [실제 요청 본문 예시](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/actual_request_body_example.json), [07b 노트북](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/notebooks/07b_rescue_lane_confirmatory.ipynb)을 최종 입력 정의로 사용한다.
- 출력은 근거·반대 근거·불확실성·추가 확인·우선순위·검토 권고다. [LLM 출력 기본 스키마](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07a_pipeline_skeleton/llm_output_schema.json)와 07b 노트북의 noRF 검증을 함께 인용한다.

### 2.3 평가 원칙

- 사전 등록, 파일 해시 고정, 정답 라벨 개봉 전 순위 확정, H1 → H2 → H3 순차 검정.
- 양성은 실제 `TTF ≤ 5,000초`. 예측 임계값과 정답 정의를 구별한다.
- **전체 endpoint 비교:** 저장된 04b test 예측을 사용하는 API 없는 사후 기술·탐색 분석. “전체”는 원시 센서 전 행이 아닌 저장된 test endpoint 전체다.
- **07b 확증:** 고정 percentile endpoint 표본 총 720행. 주 모집단은 fault1+2 rescue 249행(양성 32, 음성 217, 시퀀스 37개), fault3는 별도 보고한다. E20은 fault별 rescue의 20%를 반올림 규칙에 따라 검토한다.

**근거:** [사전 등록 v2.1](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/prereg_07_v2_1.json), [개정·순위 고정·라벨 개봉 시각](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/trial_log.md), [07b 무결성 점검](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/tables/07b_integrity_checks.csv), [전체 endpoint 커버리지](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/tables/full_endpoint_coverage.csv), [최종 비교 구현](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/src/fabjudge/final_comparison.py).

## 3. 핵심 발견: RF 게이트가 실제 고장 경보를 버린다 — 2쪽

### 3.1 fault3: 선택적 경보에서 게이트의 비용

**표 T5 fault3 행:** [T5_full_endpoint_comparison.csv](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/tables/T5_full_endpoint_comparison.csv).

| 방식 | 양성 endpoint | TP | FN | 검토 endpoint | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| 04b | 466 | 293 | 173 | 439 | 0.6288 | 0.6674 |
| LSTM 단독 / rescue 전부 유지 | 466 | 454 | 12 | 747 | 0.9742 | 0.6078 |

- rescue는 추가 308개 endpoint이며, 이 중 양성 161개: `161 / 308 = 52.3%`.
- 추가 검토 `308 / 161 = 1.91개 endpoint`당 양성 endpoint 1개를 회수한다. 이를 독립 고장 사건 1건으로 표현하지 않는다.
- 메시지: RF 게이트는 endpoint recall 약 **34.5%p**와 precision 약 **6.0%p**를 맞바꾼다.

**원천 예측:** [fault3_test_predictions.parquet](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/huang2018/fault3_test_predictions.parquet). 위 파생 비율은 T5의 TP·검토 건수 차이로 계산한다.

### 3.2 fault1+2: 사건 단위 누락

**표 T6:** [집계](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/tables/T6_sequence_event_detection.csv), [시퀀스별 상세](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/tables/T6_sequence_event_details.csv).

- 전체 66개 시퀀스 중 5,000초 이내 양성 endpoint가 있는 시퀀스는 52개다.
- 이 52개에 한정하면 04b는 **46/52**, LSTM 경보 전체 유지는 **52/52**다. RF 게이트가 놓친 사건은 6개 시퀀스다.
- 각주: T6 원본의 `시퀀스 수`는 66이다. 본문의 52는 `66 − 14`로 계산한 평가 가능 분모다. 제외된 14개(fault1 9개, fault2 5개)는 저장 endpoint에 양성 평가 시점이 없어 이 사건 탐지 정의로 평가할 수 없다. 정상 사건 또는 성공적으로 탐지한 사건으로 취급하지 않는다. 그 발생 원인을 일률적으로 추정하지 않는다.
- fault3에서는 같은 T6 정의상 RF로 추가 누락된 사건 수가 0이다. 3.1의 endpoint 회수와 사건 회수를 구별한다.

### 3.3 fault1/2의 한계 맥락

- LSTM은 저장된 fault1+2 test endpoint 18,880개 모두에 경보를 낸다. 이 임계값의 검토량은 전수 검토와 같다.
- 전체 endpoint 기준 04b precision은 **0.1867**, 기저율은 **0.1652**다. RF 게이트의 양성 농축은 제한적이며 누락이 발생한다.
- **그림 F7(A):** [F7_full_data_review_vs_recall.png](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/figures/F7_full_data_review_vs_recall.png). [곡선 원자료](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/tables/F7_full_endpoint_curve.csv), [T5 수치·구간](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/tables/T5_full_endpoint_comparison.csv).
- rescue를 무작위 순서로 검토하는 E20 기대값에서 추가 1,426개 endpoint를 검토하며, recall은 0.7038 → 0.7630, **+5.9%p**(95% CI 3.5~8.4%p)다. 실제 JEV·LLM 정렬 성능으로 해석하지 않는다.
- 메시지: 이 설정에서 경보 판별은 약하고, RF가 탈락시킨 경보에도 검토할 고장 신호가 남아 있다. “모든 단계가 어떤 정보도 갖지 않는다”로 확대하지 않는다.

**원천 예측:** [fault1](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/huang2018/fault1_test_predictions.parquet), [fault2](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/huang2018/fault2_test_predictions.parquet).

### 3.4 해석

- 비교는 저장된 04b 예측과 고정 RF 0.5·LSTM 5,000초 임계값을 사용한다. 08 분석에서 test에 맞춰 재학습하거나 임계값을 조정하지 않았다.
- 권고: 이 파이프라인에서는 RF를 경보 폐기 조건 대신 tier와 검토 순서를 정하는 신호로 사용하는 방안을 제안한다. 검토 용량과 운영 효과는 별도로 검증해야 한다.

**재현·추적:** [08 최종 비교 노트북](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/notebooks/08_final_comparison.ipynb), [계산 구현](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/src/fabjudge/final_comparison.py), [원천 파일 해시 목록](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/tables/source_manifest.csv), [일관성 점검](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/tables/final_consistency_checks.csv).

## 4. 판단 계층의 판별력: 학습된 소형 모델 대비 우월성 미확인 — 1.5~2쪽

### 4.1 JEV 단독 — 06 시리즈

**그림 F6:** [F6_jev_auc_history.png](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/figures/F6_jev_auc_history.png), [AUC 원자료](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/tables/F6_auc_source_data.csv). **표 T3:** [T3_experiment_history.csv](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/tables/T3_experiment_history.csv).

- 06a·06b: 제거 경보 0건. [06a larger_test 결과](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/06a_lstm_jev/larger_test_01/summary.json), [06b 결과](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/06b_lstm_jev_sensor_skip/summary.json).
- 06c: in-sample 성격의 dev에서 정한 P2 임계값이 test에서 일반화에 실패했다. P2 recall 0.5000, LSTM 0.9375. [06c 결과](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/06c_lstm_jev_semantic_gate/summary.json), [06c 노트북](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/notebooks/06c_lstm_jev_semantic_gate.ipynb).
- 06d: dev P2 요청에 in-sample RF가 들어간 누수 버그. 영향받은 J2/S2 dev 결과와 적합 설정은 무효다. [ERRATUM.md](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/06d_lstm_jev_oof_edge/ERRATUM.md).
- 06e: 정정 후 J2 dev AUC 0.5932. [정정 전후 dev 지표](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/06e_fix_and_weakness_diagnosis/dev_metric_comparison.csv), [06e 결론](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/06e_fix_and_weakness_diagnosis/conclusion.md).
- 메시지: 초기 JEV 결과는 RF 입력과 누수의 영향을 분리해야 했고, 정정 후에도 강한 독립 판별 신호를 확인하지 못했다. 서로 다른 표본의 AUC 변화를 동일 모집단의 성능 추세처럼 해석하지 않는다.

### 4.2 확증 검정 — 07b

- **표 T2:** [T2_confirmatory_hypothesis_tests.csv](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/tables/T2_confirmatory_hypothesis_tests.csv), [원본 검정 JSON](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/hypothesis_tests.json).
- H1: JEV routing precision@E20 **0.1020** 대 rescue 기저율 **0.1285**. 차이 −0.0265, 단측 95% 하한 −0.0719. `FAIL_TO_REJECT`는 개선 근거를 확보하지 못했다는 뜻이며 동등성 입증이 아니다.
- H2·H3: `NOT_TESTED_DUE_TO_GATEKEEPING`. LLM 비열등성이나 무작위 라우팅 대비 우월성이 검증됐다고 쓰지 않는다.
- **그림 F4:** [F4_rescue_signal_auc.png](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/figures/F4_rescue_signal_auc.png), [원자료](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/tables/F4_auc_source_data.csv). fault1+2 rescue AUC: JEV 0.4837, LLM 0.4750, LSTM 0.5386, RF 0.5662. LLM은 실패 1건을 제외한 248행, 나머지는 249행이다. 이 비교는 탐색이다.
- **그림 F2:** [F2_rescue_precision_E20.png](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/figures/F2_rescue_precision_E20.png), [탐색 비교 원자료](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/exploratory.csv), [고정 순위](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/rankings.csv). fault1+2 E20에서 로지스틱 precision 0.1837, full LLM 0.1020, LSTM-only 0.0816이다. 따라서 “LLM이 LSTM을 모든 지표에서 넘지 못했다”는 문장은 사용하지 않는다.

### 4.3 해석

- 같은 noRF 수치 입력을 사용하는 로지스틱과 비교했을 때, 해당 표본의 E20에서는 범용 모델의 우위를 확인하지 못했다.
- RF·LSTM AUC 비교는 저장 모델 신호의 비교이며 완전히 동일한 입력의 통제 실험은 아니다.
- 원인으로 입력 정보의 한계와 분포 차이를 논의할 수 있으나, “모델 크기가 아니라 입력 정보가 원인이다”라는 인과 결론은 내리지 않는다. 이를 검증하려면 입력·모델을 통제한 추가 실험이 필요하다.

## 5. 낮은 API 비용의 참고 계층으로서의 역할 — 1~1.5쪽

### 5.1 측정된 속성 — 본문용 새 표

아래 표는 기존 로그를 요약한 본 문서의 새 표다. 기존 T1~T6의 재생성 파일로 취급하지 않는다.

| 속성 | 보고할 값 | 정확한 근거·해석 |
|---|---|---|
| LLM 비용 | 메인 실행 비용 $0.0714501 / 271경보 = 약 $0.000264/경보; 1,000경보 환산 약 $0.264 | [07b summary](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/summary.json)의 `cost_and_calls`; 재시도 포함, 안정성 반복 제외. 건당 상한이 아닌 평균 |
| JEV 비용 | $0.013059438 / 271 = 약 $0.0000482/경보 | [동일 실행 요약](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/summary.json) |
| 유효 응답 | 270/271 = 99.63% | [LLM 결과](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/llm_noRF_results.jsonl); 동일 요청 최대 2회 재시도 후, 최종 실패 1건. 첫 응답 성공률과 다름 |
| 인용 수치 일치 | 성공 응답 270개, 인용 1,706개 모두 검증 통과 | [LLM 결과](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/llm_noRF_results.jsonl)의 `citation_count`, `citation_agreement`; [07b 노트북](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/notebooks/07b_rescue_lane_confirmatory.ipynb)의 인용 검증. 문장의 의미·인과 해석 정확도를 보증하지 않음 |
| 반복 안정성 | priority_score 표본 표준편차 3.37~8.81점, 범위 8~19점 | [exploratory.csv](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/exploratory.csv)의 `scope=stability`, `difference`; 5개 사례 각각 메인 응답 + 3회 반복, ddof=1. [반복 원자료](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/llm_stability.csv) |
| JEV 라우팅 환산 | LLM 271 → 136건, 49.82% 감소 | [실행 요약](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/summary.json); 실제 호출 절감 실증이 아닌 저장 응답 기반 운영 시나리오 |
| 라우팅 E20 precision | JEV routing과 full LLM 모두 0.1020 | [검정 결과](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/hypothesis_tests.json); H2 미검정이므로 기술 통계이며 비열등성 확증 아님 |

07a의 71/71 인용 점검은 [llm_smoke_summary.json](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07a_pipeline_skeleton/llm_smoke_summary.json)에 보존하되, 본문 대표 수치는 07b 전체 성공 응답 집계로 대체한다.

### 5.2 역할 정의

- 결정권과 경보 취소 권한을 주지 않는다. 실패 응답도 검토 대상에서 제거하지 않는다.
- 산출물은 근거 요약, 반대 근거, 알려지지 않은 사항, 추가 확인 항목이다. 모델 점수는 검토 참고 자료다.
- 관측한 API 단가에서는 모든 rescue에 설명을 붙이는 방안을 검토할 수 있다. 사람 검토 비용 대비 비중, 통합·감사·유지 비용은 측정하지 않았으므로 “무시할 수 있는 총비용”으로 단정하지 않는다.

**설계·검증 근거:** [최종 모델 설계 문서](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/docs/jev_final_model_design.md), [출력 스키마](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07a_pipeline_skeleton/llm_output_schema.json), [수치 인용 검증 구현](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/src/fabjudge/pipeline/claims.py), [07b 결론](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/conclusion.md).

### 5.3 측정하지 않은 것 — 반드시 명시

사람의 판단 정확도, 검토 시간, 신뢰도에 주는 효과는 측정하지 않았다. “비용 효율적”이라는 표현의 실증 범위는 낮은 API 비용과 인용 수치의 검증 가능성까지다. 비용 대비 사람의 효용은 향후 사용자 연구 과제다.

## 6. 방법론적 교훈과 평가 함정 — 1쪽

**표 T4:** [T4_pitfalls.csv](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/tables/T4_pitfalls.csv). **그림 F5:** [F5_position_leakage.png](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/figures/F5_position_leakage.png), [위치별 양성률](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/tables/F5_position_rates.csv), [경보 구간 길이](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/tables/F5_alarm_lengths.csv).

- **06d RF 누수와 06e 정정:** [06d 정오표](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/06d_lstm_jev_oof_edge/ERRATUM.md), [06e dev 비교](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/06e_fix_and_weakness_diagnosis/dev_metric_comparison.csv). 무효 결과는 성능 근거에서 제외한다.
- **in-sample dev의 임계값 붕괴:** [06c summary](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/06c_lstm_jev_semantic_gate/summary.json), [dev 행](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/06c_lstm_jev_semantic_gate/dev_rows.csv), [test 결과](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/06c_lstm_jev_semantic_gate/test_results.csv).
- **라벨 경계 밖 이른 경보:** 공통 80행 LSTM FP 8개 중 7개는 실제 TTF 5,546~8,344초다. 운영상 의미를 논의할 수 있으나 공식 평가에서는 FP를 유지한다. “가짜 FP”는 평가 오류로 오해될 수 있어 “5,000초 밖 이른 경보”로 표현한다. [T4](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/tables/T4_pitfalls.csv), [06c test 원자료](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/06c_lstm_jev_semantic_gate/test_results.csv).
- **07a 양성 0건:** 전체 상한 600행, 시퀀스당 1행으로 줄어든 표본에서 fault1/2 양성 0건. [표본 통계](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07a_pipeline_skeleton/dev_sample_statistics.json).
- **위치 누수:** 고장 시각에 맞춰 구성한 시퀀스의 상대 위치는 미래 고장과 기계적으로 연동될 수 있다. F5는 배포 불가능한 신호의 진단 그림이다. 원안의 **AUC 0.93은 확인한 F5 원자료·최종 비교 파일에서 직접 확인되지 않아 확정 수치에서 제외**한다. 이를 유지하려면 정확한 모집단·점수 정의·계산 출처를 별도로 확보해야 한다.
- **개정 이력:** [07 v1](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07a_pipeline_skeleton/prereg_07.json) → [v2](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/prereg_07_v2.json) → [v2.1](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/prereg_07_v2_1.json). 07b 평가 라벨 개봉 전에 개정했다는 범위로 쓴다. 07a 표본의 양성 부재를 확인한 사실과 fault3의 이전 사용까지 부정하는 “모든 라벨을 한 번도 보지 않았다”는 표현은 피한다.

**메시지:** 부정적 결과도 누수 정정, 사전 등록, 미검정 상태, 개정 시점을 공개하면 검증 가능한 결과가 된다. [07b trial_log](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/trial_log.md), [resume_audit](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/resume_audit.json).

## 7. 한계 — 반 쪽

- 07b는 고정 percentile의 endpoint 표본이다. 전체 데이터의 rescue 곡선은 무작위 순서 기대값이며 실제 판단 계층 결과가 아니다. [07b 사전 등록](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/prereg_07_v2_1.json), [F7 원자료](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/tables/F7_full_endpoint_curve.csv).
- fault3 test는 06 시리즈에서 이미 사용했다. [실험 연표 T3](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/tables/T3_experiment_history.csv), [07b 결론](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/conclusion.md).
- train LSTM 예측은 in-sample이며, RF OOF 정정만으로 전체 학습 경로의 낙관성이 제거되지는 않는다. [06e 결론](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/06e_fix_and_weakness_diagnosis/conclusion.md), [LSTM 입력 점검](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/06e_fix_and_weakness_diagnosis/lstm_input_check.json).
- 시퀀스 수와 장비가 제한돼 있다. endpoint 수가 많아도 독립 고장 사건 수와 같지 않다. [분할 메타데이터](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/huang2018/split_metadata.json), [T6](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/tables/T6_sequence_event_detection.csv).
- 5,000초 단일 임계값, 한정된 모델·프롬프트·입력 구성의 결과다. 입력 정보 부족을 원인으로 확정하지 않는다.
- 사람 대상 평가가 없으며, 인용 수치의 일치가 설명의 유용성·정확성을 모두 보증하지 않는다.

## 8. 결론과 향후 과제 — 반 쪽

**결론 3줄**

1. 본 데이터·설정에서는 RF 게이트를 경보 폐기 조건 대신 우선순위 신호로 사용하는 방안을 권고한다.
2. JEV·LLM의 우선순위 개선은 확증되지 않았으며, 학습된 소형 모델의 대체 근거를 확보하지 못했다.
3. 낮은 API 비용과 검증 가능한 수치 인용을 가진 참고 계층으로 고려할 수 있지만, 사람에게 주는 효용은 아직 미측정이다.

**향후 과제**

- 고장 시각이나 failure-anchored 위치에 의존하지 않는 causal 추세 feature 설계.
- 설명 제공 유무를 비교해 정확도·검토 시간·적절한 신뢰를 측정하는 사용자 연구.
- 다른 장비·데이터셋·시간 기준에서 재검증.

결론의 근거 범위는 [T5](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/tables/T5_full_endpoint_comparison.csv), [T6](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/tables/T6_sequence_event_detection.csv), [07b 가설 검정](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/hypothesis_tests.json), [07b 실행 요약](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/summary.json)으로 한정한다.

## 부록

### A. 실험 연표 06a~08과 각 단계의 결정

| 단계 | 실행 노트북 또는 스크립트 | 핵심 기록 |
|---|---|---|
| 06a | [06a_lstm_jev_hybrid.ipynb](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/notebooks/06a_lstm_jev_hybrid.ipynb) | [larger_test summary](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/06a_lstm_jev/larger_test_01/summary.json) |
| 06b | [06b_lstm_jev_sensor_skip.ipynb](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/notebooks/06b_lstm_jev_sensor_skip.ipynb) | [summary](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/06b_lstm_jev_sensor_skip/summary.json) |
| 06c | [06c_lstm_jev_semantic_gate.ipynb](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/notebooks/06c_lstm_jev_semantic_gate.ipynb) | [summary](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/06c_lstm_jev_semantic_gate/summary.json) |
| 06d | [06d_lstm_jev_oof_edge.ipynb](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/notebooks/06d_lstm_jev_oof_edge.ipynb) | [ERRATUM](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/06d_lstm_jev_oof_edge/ERRATUM.md) |
| 06e | [06e_fix_and_weakness_diagnosis.ipynb](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/notebooks/06e_fix_and_weakness_diagnosis.ipynb) | [conclusion](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/06e_fix_and_weakness_diagnosis/conclusion.md) |
| 07a | [run_07a.py](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/scripts/run_07a.py) | [conclusion](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07a_pipeline_skeleton/conclusion.md) |
| 07b | [07b_rescue_lane_confirmatory.ipynb](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/notebooks/07b_rescue_lane_confirmatory.ipynb) | [conclusion](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/conclusion.md) |
| 08 | [08_final_comparison.ipynb](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/notebooks/08_final_comparison.ipynb) | [한국어 결과 요약](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/summary_ko.txt) |

07a는 기존 메타데이터에 노트북 경로가 기록되어 있으나 현재 파일 목록에서 확인되지 않아, 실제 존재하는 실행 스크립트를 연결했다. 통합 연표의 06a~07b 행은 [T3](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/tables/T3_experiment_history.csv)를 사용하고 08의 사후 비교를 별도 행으로 추가한다.

### B. 사전 등록 해시와 개정 기록

| 파일 | 기록된 SHA-256 |
|---|---|
| [prereg_07.json](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07a_pipeline_skeleton/prereg_07.json) | `9e65c4a8220248b01108e2124697ad282bc6e17c58cfd7072d952484a127e725` |
| [prereg_07_v2.json](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/prereg_07_v2.json) | `9eb9fe21d46451d1bc022c75d28436460d6b411b425e005a6bd8e14d31524cbe` |
| [prereg_07_v2_1.json](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/prereg_07_v2_1.json) | `5112fdb2be2a1618cc52ad3c5e8932d974a2362f04fa245204c5562f406942eb` |
| [rankings.csv](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/rankings.csv) | `26ffc1a4d6bea99fde201a26c784927dd6529e9c576479930478af8eb6fa36eb` |

해시·개정 사유 출처: [summary.json](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/summary.json), [trial_log.md](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/trial_log.md). 06e 별도 사전 등록은 [prereg_06e.json](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/06e_fix_and_weakness_diagnosis/prereg_06e.json), 대체된 초기판은 [prereg_06e_v1_superseded.json](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/06e_fix_and_weakness_diagnosis/prereg_06e_v1_superseded.json)에 있다.

### C. API 호출 수와 비용 총계

- **07b 실행 범위:** JEV 271회, 메인 LLM 282회(재시도 포함), 안정성 LLM 15회, 합계 568회. 총 기록 비용 **$0.088071538**. 이를 06~08 전체 프로젝트 누적 비용이라고 부르지 않는다.
- [summary.json](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/summary.json)의 `cost_and_calls`, [api_attempt_log.jsonl](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/api_attempt_log.jsonl), [resume_audit.json](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/resume_audit.json)로 확인한다. 재개·중단 로그를 합산해 중복 계상하지 않는다.
- 07a 비용은 [api_call_log.jsonl](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07a_pipeline_skeleton/api_call_log.jsonl), [cost_estimate_07b.json](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07a_pipeline_skeleton/cost_estimate_07b.json)을 별도로 인용하며 관측값과 예상값을 구분한다.
- [F3_cost_curve.png](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/figures/F3_cost_curve.png)와 [cost_curve.csv](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/cost_curve.csv)는 `검토 건수 + FN × 누락 비용 가중치` 비교다. 달러 단위 API 비용 표로 사용하지 않는다.

### D. 04b 원래 지표 — 본문과 분리

- [T5b_04b_original_metrics.csv](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/08_final_comparison/tables/T5b_04b_original_metrics.csv), 원본 [metrics.json](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/huang2018/metrics.json).
- RF의 5k fault 대 >50k normal 분류, IPW 지표, sample-count RUL 회귀, 기존 combined 지표는 본문 전체 endpoint TTF 평가와 정의가 다르다. 동일 지표처럼 직접 비교하지 않는다.
- 04b R²는 저장값이 없으므로 다른 노트북의 R²로 대체하지 않는다.

### E. EvidencePacket 필드 목록과 LLM 출력 스키마

최종 07b noRF 필드는 다음 16개다.

| 구분 | 필드 |
|---|---|
| LSTM 요약 | `lstm_pred_seconds`, `recent_lstm_delta`, `recent_lstm_std` |
| 현재 센서 입력 | `ROTATIONSPEED`, `IONGAUGEPRESSURE`, `ETCHSUPPRESSORCURRENT`, `FLOWCOOLPRESSURE`, `ETCHBEAMCURRENT`, `FLOWCOOLFLOWRATE` |
| causal trailing feature | `ROTATIONSPEED__rms`, `IONGAUGEPRESSURE__shape_factor`, `ETCHSUPPRESSORCURRENT__peak_abs`, `FLOWCOOLPRESSURE__shape_factor`, `FLOWCOOLPRESSURE__peak_abs`, `FLOWCOOLFLOWRATE__peak_abs` |
| fault family | `fault_type` |

- LSTM 예측·변동 필드는 초 단위다. 센서 값은 정규화된 모델 입력이며 물리 단위가 확인되지 않았으므로 임의의 물리 단위를 부여하지 않는다.
- 07a의 `rf_probability`는 07b에서 제거했다. 기본 스키마를 그대로 최종 입력 스키마로 제시하지 않는다.
- 출력 최상위 필드: `evidence_for`, `evidence_against`, `unknowns`, `additional_checks`, `priority_score`, `review_recommendation`. 근거 항목은 `field`, `value`, `claim`을 포함한다.
- 우선순위는 0~100, 권고는 `urgent_review`, `standard_review`, `low_priority_review`다. 모두 검토 권고이며 고장 확정이나 경보 취소 명령이 아니다.

**원본:** [schemas.py](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/src/fabjudge/pipeline/schemas.py), [07a packet schema](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07a_pipeline_skeleton/packet_schema.json), [LLM output schema](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07a_pipeline_skeleton/llm_output_schema.json), [07b prereg v2.1](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/prereg_07_v2_1.json), [실제 요청 예시](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/actual_request_body_example.json), [JEV R2 질문](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/artifacts/07b_rescue_lane_confirmatory/questions/R2.json), [07b 실행·검증 코드](C:/projects/FabJudge-PHM2018-starter/FabJudge-PHM2018/notebooks/07b_rescue_lane_confirmatory.ipynb).
