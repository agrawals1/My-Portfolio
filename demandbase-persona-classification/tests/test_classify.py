import pandas as pd
import pytest

from persona.classify import OUTPUT_COLUMNS, classify_titles
from persona.config import Persona, PersonaConfig
from persona.llm import ScriptedLLM, validate_llm_result
from tests.conftest import frame


def by_id(df):
    return df.set_index("contact_id")


def test_sample_end_to_end(contacts, personas, llm):
    out = by_id(classify_titles(contacts, personas, llm=llm))
    assert out.loc[1, "personas"] == ["P_MKT_LEAD"] and out.loc[1, "method"] == "keyword"
    assert (out.loc[1, "level"], out.loc[1, "function"]) == ("Director", "Marketing")
    assert out.loc[2, "personas"] == ["P_FIN"] and out.loc[2, "method"] == "llm_translate+embed"
    assert out.loc[2, "level"] == "C-Level"                    # level comes from the translation
    assert out.loc[3, "personas"] == [] and out.loc[3, "method"] == "exclude_rule" and out.loc[3, "confidence"] == 1.0
    assert out.loc[3, "level"] == "Intern" and out.loc[3, "function"] == "Marketing"
    assert out.loc[4, "personas"] == ["P_IT_SEC"] and out.loc[4, "function"] == "IT/Security"
    assert out.loc[5, "personas"] == [] and out.loc[5, "method"] == "empty_input" and out.loc[5, "confidence"] == 0.0
    assert out.loc[6, "personas"] == [] and out.loc[6, "method"] == "llm" and out.loc[6, "confidence"] == 0.40
    assert out.loc[6, "level"] == "C-Level" and out.loc[6, "function"] == "HR"
    assert "below threshold" in out.loc[6, "reason"]


def test_output_shape_order_and_no_input_mutation(contacts, personas, llm):
    before = contacts.copy()
    out = classify_titles(contacts, personas, llm=llm)
    assert list(out.columns) == OUTPUT_COLUMNS
    assert out["contact_id"].tolist() == contacts["contact_id"].tolist()
    pd.testing.assert_frame_equal(contacts, before)


def test_empty_input_and_junk_titles_never_reach_the_llm(personas):
    llm = ScriptedLLM()
    out = classify_titles(frame("", None, float("nan"), "🎉", 7), personas, llm=llm)
    assert set(out["method"]) == {"empty_input"} and not llm.translate_calls and not llm.classify_calls


def test_duplicates_are_classified_once(personas):
    llm = ScriptedLLM(classifications={"chief happiness officer": {"personas": [], "level": "C-Level",
                                                                    "function": "HR", "confidence": 0.9}})
    titles = ["Chief Happiness Officer"] * 50 + ["chief happiness officer!!"] * 50
    out = classify_titles(frame(*titles), personas, llm=llm)
    assert sum(len(c) for c in llm.classify_calls) == 1       # 100 contacts, 1 LLM item
    assert len(out) == 100 and out["method"].nunique() == 1


def test_llm_budget_is_a_hard_cap(personas):
    titles = [f"Chief Wizard {i} Officer" for i in range(5)]
    llm = ScriptedLLM(classifications={f"chief wizard {i} officer": {"personas": [], "level": "C-Level",
                                                                     "function": "Unknown", "confidence": 0.9}
                                       for i in range(5)})
    out = classify_titles(frame(*titles), personas, PersonaConfig(max_llm_titles=2), llm=llm)
    assert sum(len(c) for c in llm.classify_calls) == 2
    assert (out["method"] == "budget_exceeded").sum() == 3


def test_without_llm_unmatched_titles_are_reported_not_guessed(personas):
    out = classify_titles(frame("Chief Happiness Officer", "Wunderbarer Zauberer"), personas)
    assert out["personas"].tolist() == [[], []] and set(out["method"]) == {"no_match"}


def test_embedder_catches_cognates_even_without_translation(personas):
    # 'financier' shares the 'finan' stem with 'finance': cheap cross-lingual recall, level stays Unknown
    out = classify_titles(frame("Directeur Financier"), personas)
    assert out.loc[0, "personas"] == ["P_FIN"] and out.loc[0, "method"] == "embed"
    assert out.loc[0, "level"] == "Unknown" and out.loc[0, "confidence"] <= 0.5


def test_level_filter_blocks_manager_for_leader_persona(personas):
    out = classify_titles(frame("Marketing Manager", "Marketing Director"), personas)
    assert out.loc[0, "personas"] == [] and out.loc[0, "method"] == "level_filter"
    assert out.loc[1, "personas"] == ["P_MKT_LEAD"]


def test_unknown_level_is_neutral_but_costs_confidence(personas):
    out = classify_titles(frame("Marketing", "Marketing Director"), personas)
    assert out.loc[0, "personas"] == ["P_MKT_LEAD"] and out.loc[0, "confidence"] == 0.85
    assert out.loc[1, "confidence"] == 0.95


def test_short_keyword_is_not_substring_matched(personas):
    out = classify_titles(frame("Chief Mission Officer"), personas)  # contains 'cmo'-like letters, not the token
    assert out.loc[0, "personas"] == []


def test_exclude_applies_to_embedding_and_llm_output_too(personas):
    llm = ScriptedLLM(classifications={"assistant to the cfo office": {"personas": ["P_FIN"], "level": "Staff",
                                                                       "function": "Finance", "confidence": 0.95}})
    out = classify_titles(frame("Financial Assistant", "Assistant to the CFO office"), personas, llm=llm)
    assert out["personas"].tolist() == [[], []]
    assert set(out["method"]) == {"exclude_rule"}


def test_multi_label_when_title_matches_two_personas(personas):
    out = classify_titles(frame("VP Finance and Security"), personas)
    assert out.loc[0, "personas"] == ["P_IT_SEC", "P_FIN"] or sorted(out.loc[0, "personas"]) == ["P_FIN", "P_IT_SEC"]
    assert out.loc[0, "function"] == "Unknown"  # tied functions are not guessed


def test_deterministic_under_shuffle(contacts, personas, llm):
    a = classify_titles(contacts, personas, llm=llm).sort_values("contact_id").reset_index(drop=True)
    b = classify_titles(contacts.sample(frac=1, random_state=3), personas, llm=llm)
    b = b.sort_values("contact_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


# --- LLM output validation -------------------------------------------------------------------------
@pytest.mark.parametrize("raw", [None, "yes", [], {"personas": "P_FIN", "confidence": 0.9},
                                 {"personas": [], "confidence": "high"}, {"personas": [], "confidence": True},
                                 {"personas": [], "confidence": float("nan")}, {"personas": []}])
def test_validate_rejects_malformed(raw):
    assert validate_llm_result(raw, {"P_FIN"}) is None


def test_validate_drops_hallucinated_ids_and_clips():
    res = validate_llm_result({"personas": ["P_FIN", "P_MADE_UP", 3], "level": "Wizard", "function": "Finance",
                               "confidence": 7}, {"P_FIN"})
    assert res == {"personas": ["P_FIN"], "level": "Unknown", "function": "Finance", "confidence": 1.0}


def test_invalid_llm_answer_is_recorded_not_crashed(personas):
    llm = ScriptedLLM(classifications={"chief wizard officer": {"personas": "all", "confidence": "high"}})
    out = classify_titles(frame("Chief Wizard Officer"), personas, llm=llm)
    assert out.loc[0, "method"] == "llm_invalid" and out.loc[0, "personas"] == []


def test_low_confidence_llm_persona_is_not_stamped(personas):
    llm = ScriptedLLM(classifications={"chief wizard officer": {"personas": ["P_FIN"], "level": "C-Level",
                                                                "function": "Finance", "confidence": 0.3}})
    out = classify_titles(frame("Chief Wizard Officer"), personas, llm=llm)
    assert out.loc[0, "personas"] == [] and "not stamped" in out.loc[0, "reason"]


# --- config / input validation ---------------------------------------------------------------------
def test_bad_input_raises_clear_errors(personas):
    with pytest.raises(ValueError, match="contact_id"):
        classify_titles(pd.DataFrame({"title": ["x"]}), personas)
    with pytest.raises(ValueError, match="duplicate persona_id"):
        classify_titles(frame("x"), personas + personas[:1])
    with pytest.raises(ValueError, match="unknown levels"):
        Persona("P", "n", levels=("Godlike",))
    with pytest.raises(ValueError):
        PersonaConfig(embed_accept=1.5)


def test_empty_frame_and_no_personas():
    assert classify_titles(frame(), []).empty
    assert classify_titles(frame("CFO"), []).loc[0, "personas"] == []
