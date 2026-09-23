# 집에서 이어서 작업할 AI를 위한 인수인계

기준일: 2026-09-23. 먼저 `README.md`, `data/reviewed_evidence.json`, `output/live/report.md`를 읽어라. 이 문서는 현재 상태와 변경 금지 원칙을 요약한다. 저장소: https://github.com/msj0726/skala_RAG_project

## 과제 범위와 고정 결정

- 비교 대상은 사람이 고정 선정한 **DeepSeek-V2 MLA**(SW)와 **1M-Token LLM Inference의 CXL-PNM / PNM-KV·PnG-KV**(HW)다. 기술 선정 에이전트를 만들지 않는다.
- 평가 범위는 **시장성 M1~M3**와 **AI 코딩·업무 에이전트 도메인 D1~D2**뿐이다. 기술 성숙도·이해관계자 관점은 제외한다.
- 보고서는 추천·총점·우열 판정이 아니라 관점별 차이를 설명한다. 맨 앞 `SUMMARY`는 짧은 핵심 요약, 맨 뒤 `REFERENCE`는 실제 사용 자료만 둔다. 기존 목차와 표 중심 구성을 유지한다.
- 과제 제출물은 개발 산출물, 공개 Git 링크와 PDF다. 사용자가 후편집할 수 있도록 Word도 제공한다. 제출 기한은 사용자가 제시한 **9월 23일(수) 자정**이다.

## 현재 결과와 해석 경계

2026-09-23 `live` 실행이 완료됐다. 최종 판정은 `data/reviewed_evidence.json`의 사람이 검토한 근거 원장이 결정하고, 실시간 LLM 제안·웹검색 결과는 `output/live/evaluation.json`에 감사 추적으로 남는다. 웹에서 URL을 찾았다는 사실만으로 점수를 바꾸지 않는다.

| 기준 | MLA | CXL-PNM | 주의 |
| --- | --- | --- | --- |
| M1 성장성 | 높음 4/6 | 보통 3/6 | 향후 3년 정성 판단, 시장 규모 예측 아님 |
| M2 채택 | 선정 V2 L1 / MLA 계열 L4 | 선정 PNM L1 / CXL 계열 L3 | 계열 사례를 논문 구현의 상용화로 옮기지 않음 |
| M3 생태계 | L2 | L2 | Y 2개씩; 원문 청크로만 확인 |
| D1 장기 세션 수용성 | L3, KV 예산 환산 14.925배 | L0, 같은 조건의 증가 배수 근거 없음 | MLA 수치는 실측 세션 길이가 아님; PNM L0는 실패가 아닌 근거 부족 |
| D2 동시 세션에서의 생성 처리량 개선 | L3, 자체 기준 대비 5.76배 | L3, 자체 기준 대비 최대 21.9배 | 서로 다른 기준·부하·검증 단계이므로 배수로 우열 비교 금지 |

D2의 코드 구간은 4배 이상 L3, 2배 이상 L2, 1배 초과 L1, 1배 이하 L0, 수치 근거 없으면 `판정 유보`다. MLA 5.76배는 8×H800 단일 노드에서 DeepSeek 67B 서비스의 길이 분포를 사용한 **DeepSeek-V2 전체 서빙 구성** 결과다. MLA 단독 효과가 아니며 FP8·6비트 KV 양자화·모델 차이가 포함된다. PNM 21.9배는 **사이클 수준 시뮬레이션의 최상 구성** 결과다. 두 논문 모두 코딩 에이전트의 실제 다중 도구 세션을 직접 실측한 결과가 아니며 TTFT·비용·SLA 충족을 뜻하지 않는다. 기존 D2 `응답성과 비용` 기준이나 두 기술의 `판정 유보` 문구를 되살리지 말 것.

## 코드·근거 위치

- `main.py`: LangGraph의 시장/도메인 병렬 평가 → 종합 → 보고서. `--mode live`는 OpenAI와 웹검색을 실행하고, `--mode replay`는 저장된 검토 원장을 재평가한다.
- `agents.py`: 에이전트 프롬프트·구조화 출력·검색 계획·재검색 1회·근거 검증. `scoring.py`: 점수 구간만 계산. `report.py`: 확정 State에서 PDF/Markdown 작성. `word_report.py`: Markdown에서 DOCX 작성.
- `data/papers/2405.04434v5.pdf`와 `data/papers/2511.00321v1.pdf`: 선정 논문 원문이며 Git에 포함. `data/sources.json`: S1~S11 문서 풀과 논문 해시. `data/reviewed_evidence.json`: 채택 주체·수치·근거 ID·제약이 적힌 최종 판정 원장. `data/source_verification.json`: 웹 원문 검증 기록.
- `output/live/report.pdf`, `.md`, `.docx`: 현재 제출/편집용 파일이며 Git에 포함. `output/live/evaluation.json`, `web_evidence.json`, `checkpoints/`, `data/raw/`, `data/index/`, `.cache/`, `.env`는 로컬 파일이며 Git에서 제외된다.

## 집에서 재현·수정

```bash
git clone https://github.com/msj0726/skala_RAG_project.git
cd skala_RAG_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
cp .env.example .env  # 키가 필요할 때만; 값을 Git/채팅에 올리지 말 것
python -m unittest -v
python main.py --mode replay
```

새 컴퓨터에서 최초 실행은 공식 문서·임베딩 모델 다운로드가 필요하다. `live` 재실행 시 `.env`에 본인의 `OPENAI_API_KEY`를 직접 입력하고 `python main.py --mode live`를 사용한다. OpenAI 웹검색 비용이 생길 수 있다. 새 근거로 점수를 바꾸려면 원문·비교 기준·선정/계열 범위를 확인하고 `data/reviewed_evidence.json`을 수정한 뒤 재실행한다. 단순히 LLM 제안 또는 검색 스니펫만 편집하지 않는다. `python-docx`가 설치돼 있으면 `python word_report.py output/live`로 Word를 다시 만든다. 현재 `requirements.lock`에는 `python-docx`가 없으므로 집에서 Word 재생성이 필요하면 별도 설치해야 한다.

최종 파일 변경 후 테스트, PDF 10쪽 이하·한글·표·REFERENCE를 확인하고 PDF와 Word를 함께 갱신한다. 현재 PDF는 8쪽이고 13개 단위 테스트를 통과했다. 이 컴퓨터의 LibreOffice DOCX 미리보기는 한글을 잘못 표시했지만 DOCX 내부 텍스트와 표는 확인됐다. 집의 Word에서 실제 화면을 재확인하라. 팀원 기여 정보는 아직 비어 있어 `README.md`의 Contributors를 실제 내용으로 채워야 한다.
