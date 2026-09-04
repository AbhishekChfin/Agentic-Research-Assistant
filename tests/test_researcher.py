# from app.agents import researcher as researcher_module
from app.agents.researcher import ResearcherAgent
from app.core.schemas import EvidenceItem, ResearchPlan, Subtask
from app.tools.db_search import RAGSearch
from app.tools.web_search import web_search


plan = ResearchPlan(
    query='what are class objects in python', 
    subtasks=[
        Subtask(
            id='sync-1', 
            task='Define what a class is in Python and explain its role in object-oriented programming.', 
            mode='web', 
            success_criteria=None
            ), 
        Subtask(
            id='sync-2', 
            task='Explain how class objects are created and used to instantiate instances in Python.', 
            mode='web', 
            success_criteria=None
            )
        ]
    )

researcher = ResearcherAgent(rag_search=RAGSearch())
findings = researcher.run(plan)
for finding in findings:
    print(f'ID: {finding.subtask_id}, Content: {finding.claim}')

