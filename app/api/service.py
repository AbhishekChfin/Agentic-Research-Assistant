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
    query = "What are the principals of natural justice?. Also explain audi alterum partem."
    response = run_research(query)
    print(response)