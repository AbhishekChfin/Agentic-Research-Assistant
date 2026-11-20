"""
service.py
  -> workflow.py
    -> nodes.py
      -> agents / synthesis
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

# importing pipeline modules
from app.core.schemas import FinalResponse
from app.graph.workflow import run_workflow

# Initialize fastapi application
app = FastAPI()

# Set up a jinja2template engine for rendering HTML template
templates = Jinja2Templates(directory='templates')

# Allow all origins for cross-origin resources sharing (CORS)
origins = ["*"]

# configure middleware to handle CORS, allowing requests from any origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# runs the multi agentic response for query response
@app.post("/")
async def run_research(query: str) -> FinalResponse:
    return run_workflow(query)

# # Route to render the main page with the form
# @app.get("/", tags=["authentication"])
# async def index(request: Request):
#     """
#     Renders the main HTML webpage for query and file inputs
#     """
#     return templates.TemplateResponse(
#         request=Request,
#         name=
#         context=
#     )



# if __name__=="__main__":
#     query = "Define markov chain. If an event at time t+1 depends on event at current time as well as previous times then what the process will it be called?."
#     response = run_research(query)
#     print(response)