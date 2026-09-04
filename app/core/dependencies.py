
# app/core/dependencies.py
from functools import lru_cache

from app.agents import *


@lru_cache
def get_dependencies():
    return {
        "planner": PlannerAgent(),
        "researcher": ResearcherAgent(),
        "summarizer": SummariserAgent(),
        "judge": JudgeAgent(),
    }