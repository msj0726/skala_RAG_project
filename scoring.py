"""Rubric arithmetic only. A missing measurement is not a failed measurement."""
import math


def number(value, low=0, high=math.inf):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"Invalid numeric evidence: {value!r}")
    return value


def growth(points):
    if len(points) != 3 or any(type(p) is not int or not 0 <= p <= 2 for p in points):
        raise ValueError("M1 requires exactly three integer scores in 0..2")
    score = sum(points)
    return {"score": score, "label": "낮음" if score <= 1 else "보통" if score <= 3 else "높음" if score <= 5 else "매우 높음"}


def adoption(cases, scope=None):
    levels = []
    for case in cases:
        if type(case["level"]) is not int or not 0 <= case["level"] <= 4:
            raise ValueError("M2 level outside 0..4")
        if case["level"] and (not case["actor"].strip() or not case["refs"]):
            raise ValueError("M2 requires actor and evidence")
        if scope is None or case["scope"] == scope:
            levels.append(case["level"])
    return f"L{max(levels, default=0)}"


def ecosystem(flags, author_implementation):
    if len(flags) != 4 or any(type(f) is not bool for f in flags) or type(author_implementation) is not bool:
        raise ValueError("M3 requires four booleans and original implementation status")
    count = sum(flags)
    return {"count": count, "label": "L3" if count >= 3 else "L2" if count else "L1" if author_implementation else "L0"}


def capacity(reduction_pct=None, expansion=None):
    if reduction_pct is not None and expansion is not None:
        raise ValueError("D1: choose one matched comparison, not two different baselines")
    if reduction_pct is not None:
        number(reduction_pct, 0, 100)
        if reduction_pct == 100:
            raise ValueError("100% KV removal cannot be converted to a finite expansion")
        expansion = 1 / (1 - reduction_pct / 100)
    if expansion is None:
        return {"label": "L0", "factor": None, "status": "근거 없음"}
    number(expansion)
    return {"label": "L3" if expansion >= 4 else "L2" if expansion >= 2 else "L1" if expansion > 1 else "L0", "factor": round(expansion, 3), "status": "확인"}


def throughput(improvement_factor):
    if improvement_factor is None:
        return {"label": "판정 유보", "factor": None, "status": "자체 기준 대비 생성 처리량 근거 부족"}
    number(improvement_factor)
    return {"label": "L3" if improvement_factor >= 4 else "L2" if improvement_factor >= 2 else
            "L1" if improvement_factor > 1 else "L0", "factor": improvement_factor,
            "status": "선정 논문 내 기준 대비 최대 생성 처리량"}
