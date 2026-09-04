from app.core.schemas import EvidenceItem, ResearchPlan
from app.tools.db_search import RAGSearch
from app.tools.web_search import web_search



class ResearcherAgent:
    def __init__(self, rag_search: RAGSearch | None = None):
        self.rag_search = rag_search or RAGSearch()

    def run(self, plan: ResearchPlan) -> list[EvidenceItem]:
        findings: list[EvidenceItem] = []

        for subtask in plan.subtasks:

            if subtask.mode == "web":
                results = web_search(subtask.task)

            elif subtask.mode == "rag":
                results = self.rag_search.search(subtask.task)

            else:
                continue

            for result in results:
                findings.append(
                    EvidenceItem(
                        subtask_id=subtask.id,
                        claim=result.claim,
                        source_title=result.source_title,
                        source_url=result.source_url,
                        confidence=result.confidence,
                    )
                )
        return findings




# if __name__=="__main__":
#     researcher = ResearcherAgent()

