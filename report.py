"""Render the evaluated state, never ask the writer to recalculate a grade."""
import os
from pathlib import Path
from xml.sax.saxutils import escape

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

from agents import refs_in

NAMES = {"MLA": "DeepSeek-V2 MLA", "PNM": "CXL-PNM (PNM-KV / PnG-KV)"}


def citation(ids):
    return " " + " ".join(f"[{i}]" for i in dict.fromkeys(ids)) if ids else ""


def build_sections(state):
    market = {x["tech"]: x for x in state["market"]}
    domain = {x["tech"]: x for x in state["domain"]}
    synth = state["synthesis"]
    sections = []
    first = [("p", synth["summary"] + citation(synth["refs"])),
             ("meta", f"평가 기준일: {state['as_of']} | 대상: AI 코딩·업무 에이전트 | 시장·도메인 2개 관점"),
             ("meta", "실행: " + ("검토 기록 재현 + 실제 원문 벡터 검색 + 코드 채점 (API 미사용)" if state["mode"] == "replay" else "OpenAI 실시간 조사 + WebSearch + 원문 벡터 검색; 최종 판정은 검토 원장과 대조")),
             ("h", "평가 결과를 읽는 법"),
             ("table", [["기준", "MLA", "CXL-PNM"],
                        ["M1 성장성", f"{market['MLA']['scores']['M1']['label']} ({market['MLA']['scores']['M1']['score']}/6)", f"{market['PNM']['scores']['M1']['label']} ({market['PNM']['scores']['M1']['score']}/6)"],
                        ["M2 채택 (선정 / 계열)", f"{market['MLA']['scores']['M2_selected']} / {market['MLA']['scores']['M2_family']}", f"{market['PNM']['scores']['M2_selected']} / {market['PNM']['scores']['M2_family']}"],
                        ["M3 생태계", market['MLA']['scores']['M3']['label'], market['PNM']['scores']['M3']['label']],
                        ["D1 장기 세션 수용성", domain['MLA']['scores']['D1']['label'], domain['PNM']['scores']['D1']['label']],
                        ["D2 응답성·비용", domain['MLA']['scores']['D2']['label'], domain['PNM']['scores']['D2']['label']]]),
             ("p", "M2의 선정/계열 범위는 분리 표기한다. D1의 L0는 개선 부재 또는 비교 근거 부족을 뜻하므로 사유를 함께 읽어야 한다. 전체 합산 점수와 기술 순위는 만들지 않는다."),
             ("h", "1. 분석 배경"),
             ("p", "자기회귀 생성은 이전 토큰의 Key와 Value를 보관해 반복 계산을 줄인다. 코드·대화·도구 출력이 누적되면 KV 메모리와 읽기 부담이 커진다. 두 접근은 이 병목을 각각 표현 압축과 외부 메모리/근접 연산으로 다룬다. [S1] [S2]")]
    sections.append(("SUMMARY", first))
    sections.append(("2. 기술 선정과 개요", [
        ("h", "Human 기반 고정 선정"),
        ("p", "기술 선택은 사람이 완료했다. 실행마다 같은 조합을 비교해 검색 전략과 문서 풀을 고정한다. 선정 기준은 접근의 대표성, 변경 전제의 대조성, 시장·도메인 두 관점의 자료 확보 가능성이다. 선정 능력 자체는 평가하지 않는다."),
        ("table", [["비교축", "DeepSeek-V2 MLA", "CXL-PNM"],
                   ["변경 지점", "모델의 KV 표현과 어텐션 구조", "서버 메모리 계층과 근접 연산"],
                   ["요구 전제", "MLA 모델·서빙 커널", "CXL-PNM 장치·드라이버·런타임"],
                   ["분석 초점", "세션당 캐시 예산", "외부 캐시 수용과 선택 연산"]]),
        ("h", "2.1 SW: DeepSeek-V2 MLA"),
        ("p", "저차원 잠재 벡터로 KV를 표현하는 아키텍처 개선 접근이다. 본 분석은 2024년 6월 개정본을 사용한다. 기존 GPU 인프라에서 실행할 수 있지만 모델 구조와 대응 커널이 전제된다. 사후 양자화처럼 임의 모델에 그대로 적용하는 방식이 아니다. [S1] [S3]"),
        ("h", "2.2 HW: 1M-Token LLM Inference의 CXL-PNM"),
        ("p", "CXL 메모리의 PNM 가속기로 KV 관련 선택·연산을 옮긴다. PNM-KV와 GPU 병행 방식 PnG-KV를 다루는 2025년 10월 31일 원문을 선정했다. PIM/CXL은 과제상의 포괄 명칭이고, 원문의 구체 방식은 Processing-Near-Memory다. [S2]"),
        ("h", "평가 단위"),
        ("p", "도메인의 정량치는 선정 논문 단위로 평가한다. 시장은 MLA 계열과 PIM/PNM·CXL 메모리 계열까지 확장하되 원 기술과 구분한다. 일반 CXL 제품과 규격은 해당 PNM 구현의 상용 운영 증거가 아니다. [S2] [S6] [S7]")]))
    sections.append(("3. 평가 방법", [
        ("h", "M1 성장성: 2026-09-23 이후 3년"),
        ("p", "세부 항목은 수요 동인·채택 확대·공급/생태계 투자다. 각 0점=긍정 근거 미확인, 1점=단일 사례 또는 간접 동인, 2점=독립된 복수 사례 또는 시간상 확장 근거. 합계 0~1 낮음, 2~3 보통, 4~5 높음, 6 매우 높음으로 코드가 계산한다. 항목 구체화는 구현 시 보완한 운영 규칙이며 성장률 예측 모델은 아니다."),
        ("h", "M2 채택 / M3 생태계"),
        ("p", "M2는 채택 주체와 근거가 있는 사례의 최고 단계를 계산한다: L4 상용 운영, L3 제품화, L2 시범·PoC, L1 연구, L0 미확인. 선정 기술과 계열의 최고 단계를 각각 산출한다."),
        ("p", "M3는 프레임워크·벤더 제품·표준화·제3자 연구/도구의 네 항목을 원문 청크로 교차 확인한다. Y 3~4개 L3, 1~2개 L2, 0개이지만 원저자 구현이 있으면 L1, 둘 다 없으면 L0. 같은 구현을 중복 집계하지 않는다. N은 수집 자료 내 미확인이다."),
        ("h", "D1 장기 세션 수용성"),
        ("p", "KV 감소율 r은 1/(1-r)의 동일 예산 환산치로 변환하거나, 같은 조건의 최대 문맥·세션 증가 배수를 사용한다. 4배 이상 L3, 2배 이상~4배 미만 L2, 1배 초과~2배 미만 L1, 1배 이하·근거 없음 L0. 2배·4배 경계는 상위 구간으로 통일했다. 예산 환산치와 실제 세션 길이는 구별한다."),
        ("h", "D2 응답성과 비용 효율"),
        ("p", "동일 조건 TTFT·TPOT 중 큰 지연 증가율로 판정한다. 비용 절감이 있고 SLA를 만족하면서 지연 증가 없음 L3, 10% 이내 L2, 10% 초과 L1. 비용 절감 없음 또는 SLA 미충족은 L0. 필요한 지표가 없으면 판정 유보로 남긴다. SLA 수치는 도입 환경에 따라 별도 정의해야 한다."),
        ("p", "모든 점수는 구조화된 근거를 검증한 뒤 코드로 계산한다. LLM은 근거 해석과 설명을 담당한다. 자료가 부족하면 재검색을 최대 1회 수행하며 미확인을 숫자로 보충하지 않는다.")]))
    for tech in ("MLA", "PNM"):
        item = market[tech]
        body = [("h", f"M1 성장성: {item['scores']['M1']['label']} ({item['scores']['M1']['score']}/6)")]
        body += [("p", f"{x['name']} {x['points']}/2: {x['reason']}" + citation(x["refs"])) for x in item["growth"]]
        body += [("h", f"M2 채택: 선정 {item['scores']['M2_selected']} / 계열 {item['scores']['M2_family']}")]
        body += [("p", f"{x['actor']} · {'선정' if x['scope']=='selected' else '계열'} · L{x['level']}: {x['reason']}" + citation(x["refs"])) for x in item["cases"]]
        body += [("h", f"M3 생태계: {item['scores']['M3']['label']} ({item['scores']['M3']['count']}/4)")]
        body += [("p", f"{x['name']} {'Y' if x['present'] else 'N'}: {x['reason']}" + citation(x["refs"])) for x in item["ecosystem"]]
        body += [("meta", "원저자 구현: " + ("확인" if item["author_implementation"] else "미확인") + citation(item["author_refs"])), ("meta", item["caveat"])]
        sections.append((f"4. 시장성 평가 / {NAMES[tech]}", body))
    body = []
    for tech in ("MLA", "PNM"):
        item = domain[tech]
        body.append(("h", NAMES[tech]))
        factor = item["scores"]["D1"]["factor"]
        body.append(("p", f"D1: {item['scores']['D1']['label']}" + (f" / {factor:.3f}배 예산·용량 비교" if factor else " / 비교 근거 없음") + ". " + item["capacity"]["basis"] + citation(item["capacity"]["refs"])))
        body.append(("p", f"D2: {item['scores']['D2']['label']}. " + item["responsiveness"]["basis"] + citation(item["responsiveness"]["refs"])))
        body.append(("p", "도메인 해석: " + item["application"]))
        body.append(("meta", "적용 한계: " + item["limitations"]))
    body.append(("p", "장문맥의 수용과 유효한 기억은 같지 않다. 코드 변경 추적, 검색 정확도, 도구 호출 성공률과 같은 업무 품질은 이 KV 비교만으로 검증되지 않는다."))
    sections.append(("5. 도메인 평가 / AI 코딩·업무 에이전트", body))
    sections.append(("6. 시사점과 한계", [("h", "관점 사이의 차이")] + [("p", p) for p in synth["implications"]] +
                     [("h", "공개 정보와 평가 절차의 한계")] + [("p", p) for p in synth["limitations"]] +
                     [("h", "실행 추적"), ("meta", f"원문 {len(state['sources'])}개, PDF 실페이지와 웹 환산페이지 합계 {state['corpus_pages']}페이지. 웹은 3,000자당 1페이지로 별도 환산했다. 임베딩: {state['embedding_model']}. 검색 청크 ID와 원문 해시, 검색어, 판정 입력은 evaluation.json과 실행 trace에 기록했다."),
                      ("meta", "시장 평가는 WebSearch와 M3 원문 교차 확인, 도메인 평가는 VectorRetriever와 WebSearch를 사용한다. 실시간 모드의 모델 제안은 evaluation.json에 남긴다. 검토 원장과 불일치한 수치·해석은 보고서에 반영하지 않고, 종합 의견도 검토된 기록으로 구성한다. 재현 모드는 WebSearch·OpenAI 호출을 생략한다.")]))
    used = set(refs_in({"market": state["market"], "domain": state["domain"], "synthesis": synth})) | {"S1", "S2", "S3", "S6", "S7"}
    bibliography = []
    for source in state["sources"]:
        if source["id"] not in used:
            continue
        when = source["date"] or "발행일 미표기"
        bibliography.append(("ref", f"[{source['id']}] {source['author']} ({when}). {source['title']}. {source['venue']}. {source['url']} (확인: {state['as_of']})"))
    bibliography.append(("meta", "arXiv 자료는 프리프린트로 기재했다. 확인되지 않은 학술지·권호·페이지와 발행일은 만들어 넣지 않았다. 웹 문서 확인일은 발행일이 아니다."))
    sections.append(("REFERENCE", bibliography))
    return sections


def render(state, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    candidates = [os.getenv("REPORT_FONT", ""), "/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/System/Library/Fonts/Supplemental/AppleGothic.ttf"]
    font = next((p for p in candidates if p and Path(p).is_file()), None)
    if not font:
        raise RuntimeError("REPORT_FONT에 한글을 포함하는 TTF 경로를 지정하세요.")
    pdfmetrics.registerFont(TTFont("Korean", font))
    styles = {"p": ParagraphStyle("body", fontName="Korean", fontSize=10, leading=16, spaceAfter=9, wordWrap="CJK", textColor=colors.HexColor("#243044")),
              "h": ParagraphStyle("sub", fontName="Korean", fontSize=12, leading=18, spaceBefore=11, spaceAfter=7, textColor=colors.HexColor("#126d80")),
              "meta": ParagraphStyle("meta", fontName="Korean", fontSize=8.7, leading=14, spaceAfter=7, wordWrap="CJK", textColor=colors.HexColor("#526074")),
              "ref": ParagraphStyle("ref", fontName="Korean", fontSize=8.5, leading=14, spaceAfter=11, wordWrap="CJK"),
              "title": ParagraphStyle("title", fontName="Korean", fontSize=21, leading=29, spaceAfter=20, textColor=colors.HexColor("#102c43"))}
    story, markdown = [], []
    for i, (heading, blocks) in enumerate(build_sections(state)):
        if i:
            story.append(PageBreak())
        story.append(Paragraph(escape(heading), styles["title"]))
        markdown.append("# " + heading + "\n")
        for kind, content in blocks:
            if kind == "table":
                rows = [[Paragraph(escape(str(c)), styles["meta"]) for c in row] for row in content]
                table = Table(rows, colWidths=[165, 165, 165], hAlign="LEFT", repeatRows=1)
                table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2f1f4")),
                                           ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                           ("LINEBELOW", (0, 0), (-1, -1), .4, colors.HexColor("#dce2e8")),
                                           ("LEFTPADDING", (0, 0), (-1, -1), 8),
                                           ("TOPPADDING", (0, 0), (-1, -1), 8)]))
                story.extend([table, Spacer(1, 12)])
                markdown.append("\n".join("| " + " | ".join(map(str, row)) + " |" for row in [content[0], ["---"] * len(content[0])] + content[1:]) + "\n")
            else:
                story.append(Paragraph(escape(content), styles[kind]))
                markdown.append(("## " if kind == "h" else "") + content + "\n")
    def footer(canvas, doc):
        canvas.setFont("Korean", 8)
        canvas.setFillColor(colors.HexColor("#68788b"))
        canvas.drawString(50, 28, "KV CACHE | MLA & CXL-PNM | " + state["mode"].upper())
        canvas.drawRightString(A4[0] - 50, 28, str(doc.page))
    temporary = output / "report.pending.pdf"
    SimpleDocTemplate(str(temporary), pagesize=A4, rightMargin=50, leftMargin=50,
                      topMargin=45, bottomMargin=48, title="KV cache 최적화 기술 다관점 평가", author="KV Cache Evaluation Pipeline").build(story, onFirstPage=footer, onLaterPages=footer)
    pages = len(PdfReader(temporary).pages)
    if pages > 10:
        raise ValueError(f"보고서가 {pages}페이지입니다. 본문을 줄여 10페이지 이내로 재생성하세요.")
    temporary.replace(output / "report.pdf")
    (output / "report.md").write_text("\n".join(markdown))
    return {"pdf": str(output / "report.pdf"), "markdown": str(output / "report.md"), "pages": pages}
