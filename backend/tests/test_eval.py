from scripts.run_eval import has_expected_term, load_cases


def test_numeric_screening_does_not_confuse_years_with_forecasts():
    assert has_expected_term("Growth is 4.0% in 2026", "4")
    assert not has_expected_term("The 2024 forecast is 4.4%", "4")


def test_gold_cases_are_parseable_and_have_unique_ids():
    from pathlib import Path

    cases = load_cases(Path("eval/gold.jsonl"))
    assert len(cases) == 16
    assert len({case["id"] for case in cases}) == len(cases)
