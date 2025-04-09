from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
from scope_checker import assess_scope, generate_followup_questions, generate_soap_and_treatment

app = FastAPI()

# CORS setup for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
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

    @app.get("/")
def read_root():
    return {"status": "running", "message": "API is live!"}


# 🔹 Step 1: Check Scope
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

# 🔹 Step 2: Start Assessment (generate follow-up questions)
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

# 🔹 Step 3: Final Submission (generate SOAP + treatment)
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

