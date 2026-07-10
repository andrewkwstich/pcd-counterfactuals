from __future__ import annotations

import pandas as pd

ANCHOR = "AMOUNT APPROVED: $"

TEMPLATE = """LOAN APPLICATION

Applicant: {name}
Age: {age}
Household size: {household}
Education: {education}
Employment type: {employment}
Occupation: {occupation}
Years employed: {years_employed}
Housing type: {housing}
Owns home: {own_home}
Annual income: ${income:,}
Requested loan amount: ${requested:,}
Purchase price: ${purchase:,}
Proposed monthly payment: ${payment:,}
External credit scores: {ext1:.2f} / {ext2:.2f} / {ext3:.2f}
Open loans: {open_loans}
Total outstanding debt: ${debt:,}
Previous application approval rate: {approval_rate}%
Average payment delay: {delay:.1f} days
Average credit card utilization: {utilization}%

{anchor}"""


def render_application(row: pd.Series, name: str) -> str:
    return TEMPLATE.format(
        name=name,
        age=int(round(row["age_years"])),
        household=int(row["household_size"]),
        education=row["education"],
        employment=row["employment_type"],
        occupation=row["occupation"],
        years_employed=f"{row['years_employed']:.1f}",
        housing=row["housing_type"],
        own_home="Yes" if row["own_home"] == 1 else "No",
        income=int(round(row["annual_income"])),
        requested=int(round(row["requested_amount"])),
        purchase=int(round(row["purchase_price"])),
        payment=int(round(row["monthly_payment"])),
        ext1=row["ext_score_1"],
        ext2=row["ext_score_2"],
        ext3=row["ext_score_3"],
        open_loans=int(row["n_open_loans"]),
        debt=int(round(row["total_outstanding_debt"])),
        approval_rate=int(round(100 * row["prev_approval_rate"])),
        delay=row["avg_payment_delay_days"],
        utilization=int(round(100 * row["avg_cc_utilization"])),
        anchor=ANCHOR,
    )
