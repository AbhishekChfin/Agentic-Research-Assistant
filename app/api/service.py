"""
service.py
  -> workflow.py
    -> nodes.py
      -> agents / synthesis
"""

from app.core.schemas import FinalResponse
from app.graph.workflow import run_workflow


def run_research(query: str) -> FinalResponse:
    return run_workflow(query)

if __name__=="__main__":
    query = "Define markov chain. If an event at time t+1 depends on event at current time as well as previous times then what the process will it be called?."
    response = run_research(query)
    print(response)