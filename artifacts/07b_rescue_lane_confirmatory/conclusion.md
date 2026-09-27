### v2.1 개정 및 실패 처리

라벨 개봉과 순위 확정 전에 v2.1로 개정했습니다: 첫 LLM 응답 1건이 JSON 뒤 추가 텍스트로 파싱에 실패해 STOP_FIRST_LLM_SCHEMA_FAILURE로 중단됨.
라벨 미개봉, 순위 미확정 상태에서 실행 규칙만 개정함. 가설, 지표, 표본, 정렬 규칙은 변경 없음.
요청의 strict json_schema는 이미 적용되어 설정을 유지했습니다. 실패 시 동일 요청을 최대 2회 재시도했습니다.
Main rescue MODEL_UNAVAILABLE은 **1/271건 (0.4%)**이며, 안정성 호출에서는 **0/15건**입니다. v2의 실패 원문은 당시 저장되지 않아 복구할 수 없고, v2.1 실패 원문은 failed_raw에 기록했습니다.

### 확증 결과

- H1: FAIL_TO_REJECT; 차이=-0.026, one-sided 95% lower bound=-0.072, 기준>0.
- H2: gatekeeping 때문에 검정하지 않았습니다(`NOT_TESTED_DUE_TO_GATEKEEPING`).
- H3: gatekeeping 때문에 검정하지 않았습니다(`NOT_TESTED_DUE_TO_GATEKEEPING`).

확증 표본 조건: fault1+2 rescue positive 32, negative 217, sequences 37 — **PASS**. Primary 분석은 fault1+2의 rescue만 대상으로 하며 tier1은 H1 precision의 분모에 넣지 않았습니다.

### 04b 대비 효과 크기

E20에서 rescue 49행을 추가 검토해 5건을 회수했습니다. rescue recovery rate는 15.6%이며, sampled 전체 positives 기준 recall은 04b 70.6%에서 구조 lane 추가 후 75.2%로 바뀌었습니다(4.6%p 증가). 보조 E10은 3/25 회수, E30은 9/75 회수입니다. fault3 E20은 별도 보고로 1/4 회수했습니다.

### LLM 호출 절감

Full LLM 기준 271 rescue 행에 대해 호출했고, JEV routing의 실제 정수 MID 예산은 136행(49.8% fewer LLM-equivalent calls)입니다. 현 실험의 실제 API requests는 JEV 271, main LLM 282, stability LLM 15회입니다. 응답 usage cost와 필요한 경우 07a 단가 fallback 기준으로 full LLM equivalent cost $0.067353, JEV+MID LLM scenario cost $0.047572, 추정 절감 $0.019781 (29.369385179576263%)입니다. 실제 07b 실행 API spend는 reported $0.088072 + fallback estimate $0.000000입니다.

### MODEL_UNAVAILABLE 제외 민감도 분석

주 분석의 고정 순위에서 실패 행을 제외하고 E20·prevalence·sequence bootstrap을 다시 계산했습니다. 아래 결과는 민감도 분석이며 최종 판정에는 주 분석만 사용합니다.

- H1: 차이=-0.0270, 하한=-0.07236479697871286, 상태=FAIL_TO_REJECT
- H2: 차이=0.0000, 하한=None, 상태=NOT_TESTED_DUE_TO_GATEKEEPING
- H3: 차이=-0.0289, 하한=None, 상태=NOT_TESTED_DUE_TO_GATEKEEPING

### 탐색 결과

아래 routing 비교, logistic/LSTM-only 비교, E10/E30, 개별 fault 및 LLM stability 결과는 모두 **탐색**이며 추가 p-value나 formal hypothesis로 해석하지 않습니다. 결과는 `exploratory.csv`에 있습니다.

### 한계

- 평가 표본은 총 720행이며 fault1+2 primary rescue 249행입니다. 표본 조건 상태는 **PASS**입니다.
- 표본은 각 test sequence의 LSTM alarm endpoint에서 고정 percentile로 뽑았습니다. 모든 alarm endpoint나 운영 전체의 event rate를 대표하지는 않습니다.
- fault3 test는 06 series에서 이미 확인한 이력이 있어 완전히 unseen인 확증 세트가 아닙니다.
- primary confirmatory 집단은 fault1+2이며, fault3 결과는 별도로 보고합니다. 결과는 이 표본/프로토콜 밖으로 일반화하지 않습니다.

### 08 최종 보고용 고정 수치

- 04b/06a regression: TP=9, FP=4, FN=7, TN=60.
- JEV routing E20 precision=0.102; rescue prevalence=0.129; full LLM E20 precision=0.102; random routing mean=0.130.
- H1/H2/H3: FAIL_TO_REJECT / NOT_TESTED_DUE_TO_GATEKEEPING / NOT_TESTED_DUE_TO_GATEKEEPING; prereg SHA-256 `5112fdb2be2a1618cc52ad3c5e8932d974a2362f04fa245204c5562f406942eb`; rankings SHA-256 `26ffc1a4d6bea99fde201a26c784927dd6529e9c576479930478af8eb6fa36eb`.
- Stop condition: 없음. 07c는 생성하지 않았습니다.
