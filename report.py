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


def grid(headers, rows, widths):
    return ("table", {"rows": [headers, *rows], "widths": widths})


def build_sections(state):
    market = {x["tech"]: x for x in state["market"]}
    domain = {x["tech"]: x for x in state["domain"]}
    synth = state["synthesis"]
    sections = []
    sections.append(("SUMMARY", [
        ("p", "시장에서는 MLA 계열의 서비스 운영과 CXL 계열의 제품 기반이 확인된다. D2 생성 처리량은 각 논문의 자체 기준 대비 MLA 5.76배, CXL-PNM 최대 21.9배로 모두 L3다. 실서빙 구성과 시뮬레이션의 조건이 달라 두 배수를 직접 비교하지 않는다. [S1] [S2] [S5] [S6]")]))
    sections.append(("1. 분석 배경", [
        ("lead", "긴 코딩 에이전트 세션은 KV cache의 저장·읽기 부담을 키운다."),
        ("p", "자기회귀 생성은 이전 토큰의 Key와 Value를 보관해 반복 계산을 줄인다. 코드·대화·도구 출력이 누적될수록 이 캐시가 커지므로, 세션당 캐시를 줄이는 MLA와 외부 메모리 수용량을 늘리는 CXL-PNM을 비교한다. [S1] [S2]"),
        grid(["비교 관점", "DeepSeek-V2 MLA", "CXL-PNM"], [
            ["병목 대응", "KV 표현을 저차원으로 압축", "KV 수용량 확장·근접 연산"],
            ["변경 대상", "모델 구조·서빙 커널", "메모리 계층·서빙 경로"]], [105, 195, 195]),
        ("meta", f"평가 기준일 {state['as_of']} · 대상 AI 코딩·업무 에이전트 · 시장성/도메인 2개 관점")]))
    sections.append(("2. 기술 선정과 개요", [
        ("lead", "서로 다른 변경 지점을 대표하는 두 기술을 사람이 고정 선정했다."),
        ("h", "Human 기반 고정 선정"),
        ("p", "선정 기준은 접근의 대표성, 요구 전제의 대조성, 두 관점의 자료 확보 가능성이다. 기술 조합을 고정해 실행 간 비교 가능성을 유지한다."),
        grid(["비교축", "DeepSeek-V2 MLA", "CXL-PNM"], [
            ["핵심 접근", "KV를 저차원 잠재공간으로 압축", "CXL 메모리에 수용하고 근접 연산"],
            ["요구 전제", "MLA 모델·대응 서빙 커널", "CXL-PNM 장치·드라이버·런타임"],
            ["평가 초점", "세션당 KV 예산", "외부 KV 수용·선택 연산"]], [105, 195, 195]),
        ("h", "2.1 SW: DeepSeek-V2 MLA"),
        ("p", "모델의 어텐션 구조를 바꾸는 방식이다. 기존 GPU에서 실행할 수 있지만 MLA 모델과 대응 커널이 필요해 임의 모델에 설정만으로 적용할 수 없다. [S1] [S3]"),
        ("h", "2.2 HW: 1M-Token LLM Inference의 CXL-PNM"),
        ("p", "CXL 메모리의 PNM 가속기로 KV 관련 선택·연산을 옮긴다. 선정 논문은 PNM-KV와 GPU 병행 방식 PnG-KV를 평가한다. PIM/CXL은 포괄 명칭이며 논문의 구체 방식은 PNM이다. [S2]"),
        ("h", "평가 단위"),
        ("p", "도메인 수치는 선정 논문 단위, 시장은 기술 계열까지 조사한다. CXL 제품·규격은 선정 PNM 구현의 상용 운영 증거로 간주하지 않는다. [S2] [S6] [S7]")]))
    sections.append(("3. 평가 방법", [
        ("lead", "시장성과 AI 코딩·업무 에이전트 적용성만 평가하며, 근거 부족은 숫자로 메우지 않는다."),
        ("h", "시장성: 성장·채택·생태계"),
        grid(["기준", "판정 방식", "해석상 주의"], [
            ["M1 성장성", "수요·채택 확대·공급 투자 각 0~2점; 합계 0~1 낮음, 2~3 보통, 4~5 높음, 6 매우 높음", "3년 전망의 정성 평가이며 성장률 예측이 아님"],
            ["M2 채택", "주체가 확인된 최고 단계: L4 운영, L3 제품, L2 PoC, L1 연구, L0 미확인", "선정 기술과 기술 계열을 분리"],
            ["M3 생태계", "프레임워크·제품·표준·제3자 도구의 Y 수: 3~4 L3, 1~2 L2, 원저자 구현만 L1, 모두 없으면 L0", "Y는 원문 청크로 확인; N은 부존재가 아닌 미확인"]], [88, 245, 162]),
        ("h", "도메인: 수용성·동시 세션 생성 처리량"),
        grid(["기준", "판정 방식", "해석상 주의"], [
            ["D1 장기 세션", "동일 조건 KV 예산·최대 문맥의 증가 배수: 4배 이상 L3, 2~4배 L2, 1~2배 L1, 개선·근거 없음 L0", "KV 감소율의 역수는 예산 환산치이지 실측 세션 길이가 아님"],
            ["D2 생성 처리량", "각 논문 자체 기준 대비 동시 생성 처리량: 4배 이상 L3, 2~4배 L2, 1배 초과~2배 L1, 1배 이하 L0", "근거 없으면 판정 유보; 서로 다른 기준·실험을 직접 순위화하지 않음"]], [88, 245, 162]),
        ("meta", "M1 세부 항목과 D2 배수 구간은 과제 범위에 맞춘 운영 규칙이다. 점수는 코드가 계산하고 원문 근거를 대조한다.")]))
    for tech in ("MLA", "PNM"):
        item = market[tech]
        body = [("lead", f"성장성 {item['scores']['M1']['label']} ({item['scores']['M1']['score']}/6), 채택은 선정 {item['scores']['M2_selected']}·계열 {item['scores']['M2_family']}, 생태계는 {item['scores']['M3']['label']}다."),
                ("h", "M1 성장성"),
                grid(["항목", "점수", "판정 근거"],
                     [[x["name"], f"{x['points']}/2", x["reason"] + citation(x["refs"])] for x in item["growth"]],
                     [105, 52, 338]),
                ("h", "M2 채택"),
                grid(["범위·단계", "채택 주체", "판정 근거"],
                     [[f"{'선정' if x['scope']=='selected' else '계열'} L{x['level']}", x["actor"], x["reason"] + citation(x["refs"])] for x in item["cases"]],
                     [82, 120, 293]),
                ("h", "M3 생태계"),
                grid(["항목", "확인", "판정 근거"],
                     [[x["name"], "Y" if x["present"] else "N", x["reason"] + citation(x["refs"])] for x in item["ecosystem"]],
                     [115, 42, 338]),
                ("meta", "원저자 구현 " + ("확인" if item["author_implementation"] else "미확인") + citation(item["author_refs"]) + " · " + item["caveat"])]
        sections.append((f"4. 시장성 평가 / {NAMES[tech]}", body))
    body = [("lead", "D2는 두 기술 모두 L3지만, 서로 다른 기준과 실험 단계의 최대 생성 처리량이므로 현장 성능의 동등성은 아니다."),
            ("h", "D1 장기 세션 수용성"),
            grid(["기술", "판정", "같은 조건의 근거와 한계"],
                 [[NAMES[tech], domain[tech]["scores"]["D1"]["label"],
                   domain[tech]["capacity"]["basis"] + citation(domain[tech]["capacity"]["refs"])] for tech in ("MLA", "PNM")],
                 [115, 55, 325]),
            ("h", "D2 동시 세션에서의 생성 처리량 개선"),
            grid(["기술", "판정", "자체 기준 대비 근거와 한계"],
                 [[NAMES[tech], domain[tech]["scores"]["D2"]["label"],
                   domain[tech]["throughput"]["basis"] + citation(domain[tech]["throughput"]["refs"])] for tech in ("MLA", "PNM")],
                 [115, 65, 315]),
            ("h", "AI 코딩·업무 에이전트에 적용할 때"),
            grid(["기술", "적용 해석", "제약"],
                 [[NAMES[tech], domain[tech]["application"], domain[tech]["limitations"]] for tech in ("MLA", "PNM")],
                 [105, 195, 195]),
            ("meta", "장문맥 수용과 유효한 기억은 다르다. 코드 변경 추적·검색 정확도·도구 호출 성공률은 이 KV 비교만으로 검증되지 않는다.")]
    sections.append(("5. 도메인 평가 / AI 코딩·업무 에이전트", body))
    sections.append(("6. 시사점과 한계", [
        ("lead", "시장 채택과 자체 기준 대비 생성 처리량은 서로 다른 질문이므로 한 점수로 합치지 않는다."),
        ("h", "관점 사이의 차이"),
        grid(["쟁점", "해석"], [[f"차이 {i}", p] for i, p in enumerate(synth["implications"], 1)], [90, 405]),
        ("h", "공개 정보와 평가 절차의 한계"),
        grid(["구분", "한계"], [[f"한계 {i}", p] for i, p in enumerate(synth["limitations"], 1)], [90, 405]),
        ("h", "실행 추적"),
        ("meta", f"원문 {len(state['sources'])}개, PDF 실페이지와 웹 환산페이지 합계 {state['corpus_pages']}페이지(웹 3,000자당 1페이지). 임베딩 {state['embedding_model']}. 청크 ID·원문 해시·검색어·판정 입력은 evaluation.json에 기록했다."),
        ("meta", "실시간 검색 결과는 원문 검증 후 검토 원장과 대조한다. 일치하지 않는 모델 제안은 보고서에 반영하지 않는다. 재현 모드는 웹검색과 OpenAI 호출을 생략한다.")]))
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
              "lead": ParagraphStyle("lead", fontName="Korean", fontSize=11, leading=17, spaceAfter=12, wordWrap="CJK", textColor=colors.HexColor("#126d80")),
              "h": ParagraphStyle("sub", fontName="Korean", fontSize=12, leading=18, spaceBefore=11, spaceAfter=7, textColor=colors.HexColor("#126d80")),
              "meta": ParagraphStyle("meta", fontName="Korean", fontSize=8.7, leading=14, spaceAfter=7, wordWrap="CJK", textColor=colors.HexColor("#526074")),
              "cell": ParagraphStyle("cell", fontName="Korean", fontSize=8.5, leading=12.5, wordWrap="CJK", textColor=colors.HexColor("#243044")),
              "ref": ParagraphStyle("ref", fontName="Korean", fontSize=8.5, leading=14, spaceAfter=11, wordWrap="CJK"),
              "title": ParagraphStyle("title", fontName="Korean", fontSize=21, leading=29, spaceAfter=20, textColor=colors.HexColor("#102c43"))}
    story, markdown = [], []
    for i, (heading, blocks) in enumerate(build_sections(state)):
        if i and i != 1:
            story.append(PageBreak())
        story.append(Paragraph(escape(heading), styles["title"]))
        markdown.append("# " + heading + "\n")
        for kind, content in blocks:
            if kind == "table":
                rows = content["rows"]
                cells = [[Paragraph(escape(str(c)), styles["cell"]) for c in row] for row in rows]
                table = Table(cells, colWidths=content["widths"], hAlign="LEFT", repeatRows=1)
                table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2f1f4")),
                                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafb")]),
                                           ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                           ("LINEBELOW", (0, 0), (-1, -1), .4, colors.HexColor("#dce2e8")),
                                           ("LEFTPADDING", (0, 0), (-1, -1), 7),
                                           ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                                           ("TOPPADDING", (0, 0), (-1, -1), 7),
                                           ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
                story.extend([table, Spacer(1, 12)])
                markdown.append("\n".join("| " + " | ".join(map(str, row)) + " |" for row in [rows[0], ["---"] * len(rows[0])] + rows[1:]) + "\n")
            else:
                story.append(Paragraph(escape(content), styles[kind]))
                markdown.append(("## " if kind == "h" else "**" if kind == "lead" else "") + content + ("**" if kind == "lead" else "") + "\n")
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
