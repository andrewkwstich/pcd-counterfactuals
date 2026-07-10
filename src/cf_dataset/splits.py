from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Holdout:
    race: str = "asian"
    phrasing_cell: str = "black_female"
    attribute: str = "own_home"


SENSE2 = {"sense2_race", "sense2_names", "sense2_phrasing", "sense2_attribute"}


def coarse_split(fine: str) -> str:
    if fine in SENSE2:
        return "sense2"
    return fine


def assign_split(
    question_class: str,
    target_meta: dict,
    app_is_sense1: bool,
    holdout: Holdout,
    heldout_names: set,
) -> str:
    if question_class == "name_point":
        if target_meta.get("target_race") == holdout.race:
            return "sense2_race"
        if target_meta.get("target_name") in heldout_names:
            return "sense2_names"
    elif question_class == "name_categorical":
        if target_meta.get("target_race") == holdout.race:
            return "sense2_race"
        if target_meta.get("cell") == holdout.phrasing_cell:
            return "sense2_phrasing"
    elif question_class == "categorical":
        if target_meta.get("attribute") == holdout.attribute:
            return "sense2_attribute"
    return "sense1_test" if app_is_sense1 else "pcd_train"
