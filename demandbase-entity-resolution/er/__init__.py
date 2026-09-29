"""Company entity resolution: match CRM accounts to a canonical company universe."""
from er.config import MatchConfig, Weights
from er.resolve import resolve

__all__ = ["MatchConfig", "Weights", "resolve"]
