import pandas as pd
import pytest

from src.cf_dataset import questions as Q
from src.cf_dataset.splits import Holdout, assign_split, coarse_split


@pytest.fixture
def row():
    return pd.Series({
        "age_years": 40, "household_size": 3, "education": "Higher education",
        "employment_type": "Working", "occupation": "Managers", "years_employed": 5.0,
        "housing_type": "House / apartment", "own_home": 1, "annual_income": 100000,
        "requested_amount": 500000, "purchase_price": 500000, "monthly_payment": 15000,
        "ext_score_1": 0.5, "ext_score_2": 0.6, "ext_score_3": 0.7, "n_open_loans": 1,
        "total_outstanding_debt": 200000, "prev_approval_rate": 0.5,
        "avg_payment_delay_days": 2.0, "avg_cc_utilization": 0.3,
    })


def test_quant_swaps_one_field_and_phrases_target_only(row):
    insts = Q.quant_instances(row, "Brad")
    keys = {i.target_meta["quant_key"] for i in insts}
    assert {"income_up", "income_down", "delay_down", "ext_up"} <= keys
    up = next(i for i in insts if i.target_meta["quant_key"] == "income_up")
    assert "110,000" in up.cf_text and "100,000" not in up.cf_text
    assert up.question == "If the applicant's annual income were 10% higher, how would the approved amount change?"
    assert "100000" not in up.question


def test_categorical_skips_noops_and_flips_own_home(row):
    insts = Q.categorical_instances(row, "Brad")
    own = [i for i in insts if i.target_meta["attribute"] == "own_home"]
    assert len(own) == 1 and own[0].target_meta["level"] == 0
    assert "Owns home: No" in own[0].cf_text
    edu_levels = {i.target_meta["level"] for i in insts if i.target_meta["attribute"] == "education"}
    assert "Higher education" not in edu_levels and "Lower secondary" in edu_levels


def test_name_point_replaces_name(row):
    inst = Q.name_point_instance(row, "Aisha", {"target_race": "black"})
    assert "Applicant: Aisha" in inst.cf_text
    assert inst.question == "If the applicant's name were Aisha, how would the approved amount change?"


@pytest.mark.parametrize("delta,expected", [
    (12340, "+12300"), (-4560, "-4600"), (0, "+0"), (49, "+0"), (-51, "-100"),
])
def test_canonical_delta(delta, expected):
    assert Q.canonical_delta(delta) == expected


def test_name_categorical_phrase():
    assert Q.name_categorical_phrase("black", "female") == "name were a Black female name"
    assert Q.name_categorical_phrase("asian", "male") == "name were an Asian male name"


HELD = {"Xavier", "Yolanda"}
HO = Holdout()


@pytest.mark.parametrize("qclass,meta,app_s1,expect", [
    ("name_point", {"target_name": "Bob", "target_race": "asian", "cell": "asian_male"}, False, "sense2_race"),
    ("name_point", {"target_name": "Xavier", "target_race": "white", "cell": "white_male"}, False, "sense2_names"),
    ("name_categorical", {"target_race": "black", "cell": "black_female"}, False, "sense2_phrasing"),
    ("name_categorical", {"target_race": "asian", "cell": "asian_female"}, True, "sense2_race"),
    ("categorical", {"attribute": "own_home", "level": 0}, False, "sense2_attribute"),
    ("categorical", {"attribute": "education", "level": "Lower secondary"}, True, "sense1_test"),
    ("quant", {"quant_key": "income_up"}, False, "pcd_train"),
    ("quant", {"quant_key": "income_up"}, True, "sense1_test"),
])
def test_assign_split(qclass, meta, app_s1, expect):
    assert assign_split(qclass, meta, app_s1, HO, HELD) == expect


def test_sense2_precedence_over_app_sense1():
    s = assign_split("categorical", {"attribute": "own_home", "level": 0}, True, HO, HELD)
    assert coarse_split(s) == "sense2"
