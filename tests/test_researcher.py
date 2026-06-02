from app.agents import researcher as researcher_module
from app.agents.researcher import ResearcherAgent
from app.core.schemas import EvidenceItem, ResearchPlan, Subtask


def test_researcher_routes_subtasks_to_retrieval(monkeypatch):
    def fake_web_search(query: str) -> list[EvidenceItem]:
        return [
            EvidenceItem(
                claim=f"Fake web finding for: {query}",
                source_title="Fake Web Source",
                source_url="https://example.com/web",
                confidence=0.8,
            )
        ]

    class FakeRAGSearch:
        def search(self, query: str) -> list[EvidenceItem]:
            return [
                EvidenceItem(
                    claim=f"Fake internal finding for: {query}",
                    source_title="Fake Internal Note",
                    source_url="internal://fake",
                    confidence=0.8,
                )
            ]

    monkeypatch.setattr(researcher_module, "web_search", fake_web_search)

    plan = ResearchPlan(
        query="Compare internal RAG notes with current public best practices.",
        subtasks=[
            Subtask(
                id="s1",
                task="Find current public RAG best practices.",
                mode="web",
            ),
            Subtask(
                id="s2",
                task="Find internal notes about RAG evaluation.",
                mode="rag",
            ),
        ],
    )

    researcher = ResearcherAgent(rag_search=FakeRAGSearch())
    findings = researcher.run(plan)

    assert findings
    assert all(isinstance(item, EvidenceItem) for item in findings)
    assert {item.source_title for item in findings} == {
        "Fake Web Source",
        "Fake Internal Note",
    }
