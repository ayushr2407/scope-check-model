from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict
from scope_checker import assess_scope, generate_followup_questions, generate_soap_and_treatment
import os
from fastapi import HTTPException

app = FastAPI()

# Get the base URL from environment variable or default to local
BASE_URL = os.getenv("BASE_URL", "https://scope-check-model.onrender.com")

# Update CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[BASE_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
templates.env.globals["BASE_URL"] = BASE_URL

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

# Add health check endpoint
@app.get("/health")
async def health_check():
    return JSONResponse({"status": "healthy", "environment": os.getenv("ENV", "production")})

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
    print("Received request data:", request_data.dict())  # Debug log
    try:
        result = assess_scope(
            name=request_data.name,
            gender=request_data.gender,
            dob=request_data.dob,
            condition=request_data.condition,
            answers=request_data.answers
        )
        print("Assessment result:", result)  # Debug log
        
        # Ensure we're returning a proper JSON response
        return JSONResponse(
            content=result,
            headers={
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        )
    except Exception as e:
        print("Error in check_scope:", str(e))  # Debug log
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
            headers={
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        )

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
