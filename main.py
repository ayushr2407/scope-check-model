from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict
from scope_checker import assess_scope, generate_followup_questions, generate_soap_and_treatment

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# CORS setup for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Root route for health check
@app.get("/")
def read_root():
    return {"status": "running", "message": "API is live!"}

# ✅ Serve HTML pages from /templates
@app.get("/acne-form", response_class=HTMLResponse)
def acne_form(request: Request):
    return templates.TemplateResponse("acne_form.html", {"request": request})

@app.get("/assessment", response_class=HTMLResponse)
def assessment(request: Request):
    return templates.TemplateResponse("assessment.html", {"request": request})

@app.get("/scope-result", response_class=HTMLResponse)
def scope_result(request: Request):
    return templates.TemplateResponse("scope_result.html", {"request": request})


# ✅ Pydantic Models
class ScopeCheckRequest(BaseModel):
    name: str
    gender: str
    dob: str
    condition: str
    answers: Dict[str, str]

class AssessmentRequest(BaseModel):
    name: str
    gender: str
    dob: str
    condition: str
    answers: Dict[str, str]
    scope: str

class FinalSubmissionRequest(BaseModel):
    name: str
    gender: str
    dob: str
    condition: str
    answers: Dict[str, str]
    scope: str
    followup_answers: Dict[str, str]

# ✅ Step 1: Check Scope
@app.post("/check-scope")
async def check_scope(request_data: ScopeCheckRequest):
    result = assess_scope(
        name=request_data.name,
        gender=request_data.gender,
        dob=request_data.dob,
        condition=request_data.condition,
        answers=request_data.answers
    )
    return result

# ✅ Step 2: Start Assessment
@app.post("/start-assessment")
async def start_assessment(request_data: AssessmentRequest):
    questions = generate_followup_questions(
        name=request_data.name,
        gender=request_data.gender,
        dob=request_data.dob,
        condition=request_data.condition,
        answers=request_data.answers,
        scope=request_data.scope
    )
    return {"questions": questions}

# ✅ Step 3: Final Submission
@app.post("/submit-assessment")
async def submit_assessment(request_data: FinalSubmissionRequest):
    result = generate_soap_and_treatment(
        name=request_data.name,
        gender=request_data.gender,
        dob=request_data.dob,
        condition=request_data.condition,
        answers=request_data.answers,
        followup_answers=request_data.followup_answers,
        scope=request_data.scope
    )
    return result
