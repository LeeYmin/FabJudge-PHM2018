# FabJudge
## JEV–LLM은 이상 탐지에서 Edge를 만들 수 있는가?

PHM2018 이온 밀링 장비 데이터로 검증한 RF–LSTM 경보 보존 및 선택적 판단 계층

**프로젝트 사례 연구 · 2026년 9월 · 저장된 실험 결과 기준**

### 한눈에 보는 결과

FabJudge의 출발점은 모델을 하나 더 붙이는 일이 아니었다. 기존 RF–LSTM 파이프라인에서 RF 게이트가 LSTM의 임박 고장 경보를 버릴 때, 실제로 무엇을 잃는지 먼저 확인했다. 저장된 test endpoint에서 fault3의 recall은 04b 경보 규칙 0.6288, RF 게이트를 제거한 LSTM 경보 0.9742였다. fault1+2에서는 평가 가능한 고장 시퀀스 52개 중 04b가 46개를 탐지했고 LSTM 경보 전체 유지는 52개를 탐지했다. 이는 이 프로젝트의 고정된 04b baseline에 대한 비교이며 Huang et al. 원 논문이나 공식 challenge 점수와의 비교가 아니다. [A1][A2]

그 다음 질문은 RF가 탈락시킨 경보를 JEV와 LLM이 사람의 검토 순서에 유의미하게 배치할 수 있는가였다. 사전 등록한 주 검정 H1에서 JEV routing의 E20 precision은 0.1020으로 rescue 집단의 양성률 0.1285보다 낮았다. 개선 근거를 확보하지 못했으므로 H2와 H3는 순차 검정 규칙에 따라 검정하지 않았다. 탐색 비교에서도 같은 noRF 수치 입력을 받은 로지스틱 모델의 E20 precision이 LLM보다 높았다. [A3][A4]

따라서 이 사례가 지지하는 설계는 **경보 보존과 판단 지원의 분리**다. LSTM 경보를 없애지 않고, RF는 검토 tier를 구분하는 신호로 쓴다. JEV와 LLM은 검토 순서와 설명의 깊이를 보조할 수 있지만, 고장 정답이나 자동 인터록 권한을 갖지 않는다. LLM 설명은 재시도 후 271건 중 270건에서 유효했고 성공 응답에 포함된 수치 인용 1,706개가 입력과 일치했지만, 사람의 판단 정확도와 검토 시간 개선은 아직 측정하지 않았다. [A5][A6]

# 1. 문제: 좋은 게이트가 경보를 놓칠 수 있다

PHM Society의 2018 Data Challenge는 이온 밀링 장비의 시계열 센서 데이터로 고장을 진단하고 남은 시간을 예측하는 문제였다. 공식 문제 정의는 예측 시점의 현재 및 과거 데이터만 사용하도록 요구하며, 공개 데이터의 센서 물리 단위는 익명화되어 있다. 본 사례는 fault1, fault2, fault3에 대해 실제 고장까지 남은 시간(TTF)이 5,000초 이하인 endpoint를 양성으로 정의한다. [R1][A13]

Huang et al.은 abrupt failure에서 degradation mode를 먼저 탐지하고, 그 후 LSTM으로 RUL을 추정하는 2단 구조를 제안했다. FabJudge의 04b는 이 아이디어와 5k fault 대 >50k normal 구분을 참고해 만든 **Huang-inspired RF–LSTM baseline**이다. 저장된 기록만으로 논문의 모든 전처리·분할·학습 설정을 동일하게 복원했다고 증명할 수 없으므로, 이 포트폴리오는 exact reproduction이라는 표현을 쓰지 않는다. 관련 연구인 Singh et al.의 health-score/DTW 접근과 Vishnu et al.의 RNN 접근은 연구 맥락으로만 인용한다. [R2][R3][R4]

04b의 운영 경보는 `LSTM 예측 TTF ≤ 5,000초` **그리고** `RF 확률 ≥ 0.5`일 때 발생한다. 이 논리곱에서 RF는 판별력을 높이는 필터인 동시에 LSTM 경보를 없애는 스위치다. FabJudge는 “JEV–LLM을 붙이면 성능이 좋아지는가?”보다 앞서, 그 스위치가 제거한 경보의 실제 TTF와 사건 단위 결과를 측정했다. 이어서 제거 경보를 rescue 집단으로 분리해 판단 계층의 우선순위 성능을 검증했다. [A1][A2]

# 2. 데이터와 평가 설계

분석 단위는 센서 원시 행 전체가 아니다. 저장된 04b test 예측의 **endpoint**이며, 같은 고장 시퀀스에서 나온 endpoint는 서로 독립적인 고장 사건으로 세지 않는다. endpoint recall은 임박 고장 시점의 회수율을, sequence/event detection은 평가 가능한 고장 시퀀스 중 적어도 한 번 경보한 비율을 나타낸다. 이 두 지표를 분리해야 수백 개의 endpoint 회수를 수백 건의 독립 고장 예방으로 잘못 표현하지 않는다. 시퀀스 구성·분할·저장 예측은 프로젝트 메타데이터와 원천 해시로 추적했다. [A1][A2][A13][A15]

평가는 두 범위로 나눴다. 첫째, 저장된 test endpoint 전체에 고정된 04b 규칙을 다시 적용한 **사후 기술·탐색 분석**이다. 08 분석 단계에서 test에 맞춰 모델을 재학습하거나 임계값을 조정하지 않았다. 둘째, 07b의 **사전 등록 확증 표본**이다. 각 test 시퀀스의 LSTM 경보 endpoint에서 고정 percentile로 선택한 총 720행 중 rescue는 271행이었다. 주 검정 모집단은 fault1+2 rescue 249행(양성 32, 음성 217, 시퀀스 37개)이고 fault3는 별도로 보고했다. 따라서 아래의 전체 endpoint 기대값과 07b 판단 계층 성능은 서로 다른 모집단의 결과다. [A3][A5][A8]

07b는 라벨 개봉 전에 순위와 파일 해시를 고정하고, H1 → H2 → H3 순차 검정을 등록했다. E20은 fault별 rescue의 20%를 정수 반올림 규칙에 따라 검토하는 예산이다. 첫 LLM JSON 응답의 파싱 실패로 실행이 중단됐을 때에는 라벨 미개봉·순위 미확정 상태에서 재시도 규칙만 v2.1로 개정했다. 가설, 표본, 지표, 정렬 규칙은 유지했다. 이 이력을 기록했기에 사후에 편리한 결과만 선택했다는 해석을 줄일 수 있다. [A3][A16]

# 3. 파이프라인: 경보는 보존하고 판단은 순서를 돕는다

그림 S1의 위쪽은 04b 규칙이다. 센서 관측에서 만든 결정적 수치 특징이 LSTM과 RF에 들어가고, LSTM이 5,000초 이내 고장을 경보해도 RF 확률이 0.5 미만이면 경보가 버려진다. 아래쪽의 제안 구조는 같은 LSTM 경보를 모두 검토 대기열에 남긴다. RF가 통과시킨 경보는 **tier 1**로 즉시 검토하고, RF가 탈락시킨 경보는 **rescue lane**으로 보낸다. [A1][A3]

rescue lane에서는 RF 확률과 정답 라벨을 제외한 EvidencePacket을 JEV에 제공한다. 07b의 최종 noRF 패킷은 LSTM 예측·최근 변동, 현재 센서 입력, 과거 구간의 causal 특징, fault family 등 16개 필드다. 정답 TTF·양성 라벨·파일 및 시퀀스 식별자·시각·표본 위치도 제외했다. JEV의 HIGH/MID/LOW는 교정된 고장 확률이 아니라 fault별 25%/50%/25% 검토 순서 배분이다. 필요하면 MID에 LLM의 근거·반대 근거·불확실성·추가 확인 설명을 붙인다. 어느 분기에서도 경보를 자동 취소하거나 인터록을 결정하지 않는다. [A3][A14]

실험과 운영 시나리오는 구분해야 한다. 07b에서는 비교를 위해 rescue 271건 모두에 LLM을 호출했다. MID 136건에만 호출하는 그림의 점선 경로는 이미 저장된 응답을 재사용해 계산한 가상 운영 시나리오다. LLM 실패 응답도 사람의 검토 대기열에서 제거하지 않는다. [A5]

**그림 S1.** 04b의 경보 폐기 규칙과 FabJudge의 경보 보존·JEV/LLM 검토 라우팅. 프로젝트의 `S1_alarm_preserving_pipeline.pdf` 원본을 삽입했다.

![그림 S1: 경보 보존 파이프라인](figures/S1_alarm_preserving_pipeline.png)

# 4. 핵심 발견: RF가 버린 경보 안에 양성이 있다

fault3의 저장 test endpoint 466개는 실제 TTF가 5,000초 이내였다. 04b는 이 중 293개를 회수하고 173개를 놓쳤다. RF 게이트 없이 LSTM 경보를 모두 유지하면 454개를 회수하고 12개를 놓친다. recall은 0.6288에서 0.9742로 34.5%p 높아지는 반면 precision은 0.6674에서 0.6078로 약 6.0%p 낮아진다. 추가 검토 308개 endpoint 중 161개가 양성이었다(52.3%). 이것은 **endpoint** 회수이며 독립 고장 사건 161건을 뜻하지 않는다. [A1]

| 방식 | 양성 endpoint | TP | FN | 검토 endpoint | Recall | Precision |
|:--|--:|--:|--:|--:|--:|--:|
| 04b: LSTM ∧ RF | 466 | 293 | 173 | 439 | 0.6288 | 0.6674 |
| LSTM 경보 전체 유지 | 466 | 454 | 12 | 747 | 0.9742 | 0.6078 |

fault1+2에서는 사건 단위 결과가 더 직접적이다. 저장 test 시퀀스는 66개지만 그중 14개에는 이 분석에 필요한 5,000초 이내 양성 endpoint가 없다. 평가 가능한 나머지 52개에서 04b는 46개, LSTM 경보 전체 유지는 52개를 탐지했다. RF 게이트 때문에 새로 놓친 고장 시퀀스는 6개다. 제외된 14개를 정상 사건이나 성공 탐지로 해석하지 않았다. fault3에서는 같은 사건 정의로 RF가 추가로 놓친 시퀀스는 0개다. 따라서 fault3의 endpoint 회수와 fault1+2의 사건 회수를 분리해 보고한다. [A2]

경보 보존에는 검토 비용이 따른다. fault1+2 저장 test endpoint 18,880개에는 LSTM이 모두 경보를 냈다. 이 임계값에서 LSTM 전체 유지는 사실상 전수 검토다. 04b의 전체 endpoint precision 0.1867은 양성 기저율 0.1652보다 높지만 농축 폭이 크지 않고, 그 과정에서 누락이 발생했다. rescue를 무작위 순서로 E20만 검토한다는 **탐색적 기대값**은 추가 검토 1,426개에 recall 0.7038 → 0.7630, 차이 +5.9%p(95% CI 3.5~8.4%p)였다. 이는 실제 JEV·LLM 정렬의 효과가 아니다. 현실적인 다음 설계는 RF를 경보 폐기 조건보다 tier 및 검토 순서의 신호로 쓰고, 허용 가능한 검토량을 별도로 정하는 것이다. [A1][A8]

# 5. JEV·LLM은 우선순위를 개선했는가

탐색의 초기 단계는 실패까지 포함해 남겨 두었다. 06a·06b의 규칙은 제거 경보가 0건이어서 선택적 gate의 효용을 입증하지 못했다. 06c에서는 in-sample 성격의 dev에서 정한 P2 임계값이 test에 일반화되지 않아 P2 recall 0.5000, LSTM 0.9375가 나왔다. 06d의 dev P2 요청에는 in-sample RF가 들어간 누수가 발견되어 영향을 받은 J2/S2 dev 결과와 적합 설정을 무효화했다. 06e 정정 후 J2 dev AUC는 0.5932였다. 서로 다른 표본에서 나온 수치를 한 줄의 성능 향상 추세처럼 연결하지 않았다. [A9][A10][A11][A12]

07b의 확증 H1은 “JEV가 fault1+2 rescue에서 E20 검토 예산을 사용하면 양성률보다 높은 precision을 얻는가”였다. 실제 E20 검토 49행에서 양성 5행을 회수해 precision은 0.1020이었다. rescue 전체 양성률 0.1285와의 차이는 −0.0265, 시퀀스 단위 bootstrap의 단측 95% 하한은 −0.0719였다. 결과는 **FAIL_TO_REJECT**다. 이는 개선 근거를 얻지 못했다는 뜻이며 JEV와 기저율의 동등성을 증명하는 결과가 아니다. 사전 등록된 gatekeeping에 따라 H2의 LLM 대비 비열등성과 H3의 무작위 라우팅 대비 우월성은 미검정이다. [A3][A4]

| 07b fault1+2 rescue | 관측값 | 해석 |
|:--|--:|:--|
| 주 모집단 | 249행, 양성 32행 | 37개 rescue 시퀀스 |
| H1: JEV precision@E20 | 0.1020 (5/49) | 양성률 0.1285 대비 개선 미확인 |
| H1 차이 / 단측 95% 하한 | −0.0265 / −0.0719 | FAIL_TO_REJECT |
| H2·H3 | 미검정 | H1 gatekeeping 적용 |
| 04b 대비 E20 recall | 0.7064 → 0.7523 | 표본 내 추가 검토 49행, 양성 5행 회수 |

탐색적 순위 비교도 별도로 본다. fault1+2 rescue AUC는 JEV 0.4837, LLM 0.4750, LSTM 신호 0.5386, RF 신호 0.5662였다. LLM은 최종 실패 1건을 제외한 248행이고 나머지는 249행이므로 완전히 같은 분모의 단순 순위표가 아니다. 같은 noRF 수치 입력을 사용한 로지스틱의 E20 precision은 0.1837, full LLM은 0.1020, LSTM-only는 0.0816이었다. 따라서 이 표본의 E20에서는 로지스틱 대비 LLM 우위를 확인하지 못했지만, “LLM이 모든 소형 모델보다 모든 지표에서 낮다”는 결론도 성립하지 않는다. 원인을 모델 크기나 입력 정보 중 하나로 단정하려면 입력과 모델을 통제한 새 실험이 필요하다. [A4][A5]

# 6. 판단 계층에 남는 역할: 검증 가능한 설명

우선순위 개선이 확증되지 않았어도 운영 관찰값은 남는다. 07b 메인 실행에서 LLM 사용 기록 비용은 $0.0714501/271경보, 평균 약 **$0.000264/경보**였다. JEV는 $0.013059438/271경보, 약 $0.0000482/경보였다. 재시도 후 메인 rescue 271건 중 270건이 유효한 LLM 응답을 얻었다(99.63%). 성공한 응답의 수치 인용 1,706개는 패킷의 입력값과 모두 일치했다. 이 검증은 숫자 복사 오류를 확인할 뿐 설명의 의미, 인과관계, 사람에게 주는 효용을 보증하지 않는다. [A5][A6]

JEV 라우팅을 적용해 MID 136건에만 LLM을 호출하는 경우는 full 271건 대비 LLM 대상이 49.82% 적다. 이는 저장된 응답으로 계산한 환산치이며 실제 07b가 절약한 호출 수가 아니다. 07b 실행 기록은 JEV 271회, 메인 LLM 282회(재시도 포함), 안정성 LLM 15회, 합계 568회와 총 비용 $0.088071538이다. 이 금액은 해당 실행 기록이지 현재 단가나 전체 프로젝트 누적 비용이 아니다. 사람의 검토 비용, 통합·감사·유지 비용도 포함하지 않는다. [A5]

이 관찰은 JEV/LLM을 **결정권 없는 참고 계층**으로 좁혀 배치할 근거가 된다. 사람에게 보여 줄 출력은 근거, 반대 근거, 알려지지 않은 사항, 추가 확인 항목이다. priority score는 검토 편의를 위한 신호이며 고장 확정 또는 경보 취소 명령이 아니다. 다음 단계의 핵심 평가는 설명 제공 여부를 바꿨을 때 사람이 더 정확하고 빠르게, 적절한 신뢰 수준으로 판단하는지 측정하는 사용자 연구다. [A14]

# 7. 실패를 남긴 방법론과 한계

이 프로젝트의 중요한 산출물 중 하나는 실패 조건의 기록이다. 06d의 RF 입력 누수는 06e 정정 이전 J2/S2 dev 근거를 무효화했다. 06c의 dev 임계값 붕괴는 좋은 개발 결과가 test 일반화의 증거가 아님을 보여 줬다. 07a의 시퀀스당 1행·최대 600행 표본에는 fault1/2 양성이 0건이어서 판별력을 확증할 수 없었고, 이후 07b 표본 설계를 바꿨다. 고장 시각에 맞춰 자른 시퀀스의 상대 위치는 미래 고장과 기계적으로 연결될 수 있어 배포 feature가 아니라 누수 진단에만 사용했다. [A7][A9][A10][A12]

라벨 경계의 해석도 엄격히 했다. 공통 80행 실험에서 LSTM의 FP 8개 중 7개는 실제 TTF가 5,546~8,344초였다. 운영상 조기 경보로 논의할 수는 있지만 5,000초 양성 정의 밖이므로 공식 평가에서는 FP로 남겼다. 반대로 endpoint 수가 많다고 독립 사건 수가 많은 것은 아니다. 시퀀스·장비 수가 제한되고 동일 고장 경로에서 여러 시점을 뽑았기 때문에 신뢰구간과 일반화 해석은 시퀀스 단위에 묶여야 한다. [A2][A7][A13]

07b는 고정 percentile 표본이며 운영 현장의 연속 스트림을 대표하지 않는다. fault3 test는 06 시리즈에서 이미 사용했으므로 새 확증 세트로 내세우지 않았다. train LSTM 예측의 in-sample 성격은 RF OOF 정정만으로 해결되지 않는다. 단일 5,000초 임계값, 제한된 모델·프롬프트·입력 구성, 사람 대상 평가 부재도 남아 있다. 이 한계 안에서 “JEV/LLM이 불필요하다”보다 정확한 결론은 **현재 실험에서 우선순위 개선을 확인하지 못했고, 설명 계층의 사용자 효용은 아직 검증되지 않았다**는 것이다. [A3][A5][A11]

# 8. 결론과 다음 실험

첫째, 이 데이터와 고정 04b 규칙에서는 RF 게이트가 실제 LSTM 임박 고장 경보를 제거했다. RF가 제공하는 농축 이익과 놓친 endpoint·사건을 함께 보면, 경보를 폐기하기보다 검토 tier에 활용하는 설계가 더 검증할 가치가 있다. 둘째, JEV와 LLM의 rescue 우선순위 개선은 사전 등록 검정에서 입증되지 않았다. 학습된 소형 모델을 대체할 근거도 없다. 셋째, 낮은 실행 기록상의 API 비용과 입력 수치 대조가 가능한 응답은 참고 설명의 가능성을 보여 주지만, 사람의 의사결정 개선은 별도의 실험이 필요하다. [A1][A2][A4][A5]

다음 실험에서는 failure-anchored 위치를 쓰지 않는 causal 추세 특징을 설계하고, 고장·장비·시간을 분리한 새 평가 세트를 확보해야 한다. 그 위에서 같은 입력의 로지스틱·RF·LSTM·JEV·LLM을 비교하고, 설명 유무를 무작위로 배정한 사람 대상 실험으로 정확도·검토 시간·과신을 측정한다. 경보 보존 구조의 실효성은 모델 점수 하나가 아니라 허용 검토량과 놓친 사건의 비용을 함께 놓고 판단해야 한다.

## 근거와 재현 경로

아래 경로는 프로젝트 루트 기준이다. 이 문서의 수치는 저장된 artifact의 범위에 한정된다.

| ID | 근거 |
|:--|:--|
| A1 | `artifacts/08_final_comparison/tables/T5_full_endpoint_comparison.csv` |
| A2 | `artifacts/08_final_comparison/tables/T6_sequence_event_detection.csv` |
| A3 | `artifacts/07b_rescue_lane_confirmatory/prereg_07_v2_1.json` |
| A4 | `artifacts/07b_rescue_lane_confirmatory/hypothesis_tests.json` |
| A5 | `artifacts/07b_rescue_lane_confirmatory/summary.json`, `conclusion.md` |
| A6 | `artifacts/07b_rescue_lane_confirmatory/llm_noRF_results.jsonl` |
| A7 | `artifacts/08_final_comparison/tables/T4_pitfalls.csv` |
| A8 | `artifacts/08_final_comparison/tables/F7_full_endpoint_curve.csv` |
| A9 | `artifacts/08_final_comparison/tables/T3_experiment_history.csv` |
| A10 | `artifacts/06d_lstm_jev_oof_edge/ERRATUM.md` |
| A11 | `artifacts/06e_fix_and_weakness_diagnosis/conclusion.md` |
| A12 | `artifacts/06c_lstm_jev_semantic_gate/summary.json` |
| A13 | `artifacts/huang2018/sequence_metadata.csv`, `split_metadata.json` |
| A14 | `artifacts/07b_rescue_lane_confirmatory/actual_request_body_example.json`, `notebooks/07b_rescue_lane_confirmatory.ipynb` |
| A15 | `artifacts/08_final_comparison/tables/source_manifest.csv`, `final_consistency_checks.csv` |
| A16 | `artifacts/07b_rescue_lane_confirmatory/trial_log.md` |

## 외부 자료

**R1** PHM Society. *PHM Data Challenge 2018 – Ion Mill Etch Tool*. https://phmsociety.org/conference/annual-conference-of-the-phm-society/annual-conference-of-the-prognostics-and-health-management-society-2018-b/phm-data-challenge-6/

**R2** Huang, W. et al. (2018). *Remaining Useful Life Estimation for Systems with Abrupt Failures*. Annual Conference of the PHM Society, 10(1). https://doi.org/10.36001/phmconf.2018.v10i1.590

**R3** Singh, K. et al. (2018). *Concurrent Estimation of Remaining Useful Life for Multiple Faults in an Ion Etch Mill: A Data-driven Approach*. Annual Conference of the PHM Society, 10(1). https://doi.org/10.36001/phmconf.2018.v10i1.591

**R4** Vishnu T. V. et al. (2018). *Recurrent Neural Networks for Online Remaining Useful Life Estimation in Ion Mill Etching System*. Annual Conference of the PHM Society, 10(1). https://doi.org/10.36001/phmconf.2018.v10i1.589
