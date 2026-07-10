from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Callable

import pandas as pd

from src.data_build.render import render_application

QUESTION_PREFIX = "If the applicant's "
QUESTION_SUFFIX = ", how would the approved amount change?"


# --------------------------------------------------------------------------
# quant perturbations
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class QuantSpec:
    key: str
    phrase: str
    apply: Callable[[pd.Series], pd.Series]


def _mul(col: str, factor: float, lo: float = 0.0, hi: float = float("inf")):
    def f(row: pd.Series) -> pd.Series:
        r = row.copy()
        r[col] = float(min(max(r[col] * factor, lo), hi))
        return r
    return f


def _add(col: str, delta: float, lo: float = -float("inf"), hi: float = float("inf")):
    def f(row: pd.Series) -> pd.Series:
        r = row.copy()
        r[col] = float(min(max(r[col] + delta, lo), hi))
        return r
    return f


def _add_ext(delta: float):
    def f(row: pd.Series) -> pd.Series:
        r = row.copy()
        for c in ("ext_score_1", "ext_score_2", "ext_score_3"):
            r[c] = float(min(max(r[c] + delta, 0.0), 1.0))
        return r
    return f


DEFAULT_QUANT: tuple[QuantSpec, ...] = (
    QuantSpec("income_up",   "annual income were 10% higher", _mul("annual_income", 1.10)),
    QuantSpec("income_down", "annual income were 10% lower",  _mul("annual_income", 0.90)),
    QuantSpec("delay_down",  "average payment delay were 5 days lower",  _add("avg_payment_delay_days", -5.0)),
    QuantSpec("delay_up",    "average payment delay were 5 days higher", _add("avg_payment_delay_days", +5.0)),
    QuantSpec("util_down",   "average credit card utilization were 10 points lower",  _add("avg_cc_utilization", -0.10, lo=0.0, hi=3.0)),
    QuantSpec("util_up",     "average credit card utilization were 10 points higher", _add("avg_cc_utilization", +0.10, lo=0.0, hi=3.0)),
    QuantSpec("ext_up",      "external credit scores were 0.1 higher", _add_ext(+0.10)),
    QuantSpec("ext_down",    "external credit scores were 0.1 lower",  _add_ext(-0.10)),
)


# --------------------------------------------------------------------------
# categorical targets
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CatSpec:
    attribute: str
    level: object
    phrase: str


DEFAULT_CATEGORICAL: tuple[CatSpec, ...] = (
    CatSpec("employment_type", "Unemployed", "employment type were Unemployed"),
    CatSpec("employment_type", "Businessman", "employment type were Businessman"),
    CatSpec("education", "Higher education", "education were Higher education"),
    CatSpec("education", "Lower secondary", "education were Lower secondary"),
    CatSpec("own_home", 1, "applicant owned their home"),
    CatSpec("own_home", 0, "applicant did not own their home"),
)


# --------------------------------------------------------------------------
# question text + canonical answer
# --------------------------------------------------------------------------

def question_text(phrase: str) -> str:
    return f"{QUESTION_PREFIX}{phrase}{QUESTION_SUFFIX}"


def name_point_phrase(target_name: str) -> str:
    return f"name were {target_name}"


def name_categorical_phrase(race: str, gender: str) -> str:
    article = "an" if race[:1].lower() in "aeiou" else "a"
    return f"name were {article} {race.capitalize()} {gender} name"


def canonical_delta(delta: float, round_to: int = 100) -> str:
    d = int(round(delta / round_to) * round_to)
    return f"+{d}" if d >= 0 else str(d)


# --------------------------------------------------------------------------
# cf-text construction
# --------------------------------------------------------------------------

@dataclass
class CFInstance:
    question_class: str
    phrase: str
    question: str
    cf_text: str
    is_point: bool
    target_meta: dict = dc_field(default_factory=dict)


def quant_instances(row: pd.Series, current_name: str,
                    specs: tuple[QuantSpec, ...] = DEFAULT_QUANT) -> list[CFInstance]:
    out = []
    for s in specs:
        cf_row = s.apply(row)
        out.append(CFInstance(
            "quant", s.phrase, question_text(s.phrase),
            render_application(cf_row, current_name), True,
            {"quant_key": s.key},
        ))
    return out


def categorical_instances(row: pd.Series, current_name: str,
                          specs: tuple[CatSpec, ...] = DEFAULT_CATEGORICAL) -> list[CFInstance]:
    out = []
    for s in specs:
        if row[s.attribute] == s.level:
            continue
        cf_row = row.copy()
        cf_row[s.attribute] = s.level
        out.append(CFInstance(
            "categorical", s.phrase, question_text(s.phrase),
            render_application(cf_row, current_name), True,
            {"attribute": s.attribute, "level": s.level},
        ))
    return out


def name_point_instance(row: pd.Series, target_name: str, target_meta: dict) -> CFInstance:
    phrase = name_point_phrase(target_name)
    return CFInstance(
        "name_point", phrase, question_text(phrase),
        render_application(row, target_name), True,
        {"target_name": target_name, **target_meta},
    )


def name_categorical_cf_text(row: pd.Series, mc_name: str) -> str:
    return render_application(row, mc_name)
