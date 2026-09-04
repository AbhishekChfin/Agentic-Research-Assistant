from dataclasses import dataclass
from functools import lru_cache

from app.agents import *


@dataclass(frozen=True)
class AppDependencies:
    planner: PlannerAgent
    researcher: ResearcherAgent
    summarizer: SummariserAgent
    judge: JudgeAgent


@lru_cache
def get_dependencies() -> AppDependencies:
    return AppDependencies(
        planner=PlannerAgent(),
        researcher=ResearcherAgent(),
        summarizer=SummariserAgent(),
        judge=JudgeAgent(),
    )