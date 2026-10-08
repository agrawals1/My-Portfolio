"""Scripted LLM answers for the bundled sample, so `python -m persona.cli` runs offline."""
from persona.llm import ScriptedLLM

DEMO_LLM = ScriptedLLM(
    translations={"Directeur Financier": "Chief Financial Officer"},
    classifications={"chief happiness officer": {"personas": [], "level": "C-Level", "function": "HR", "confidence": 0.40}},
)
