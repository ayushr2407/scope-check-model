


# import os
# import logging
# import time

# # Google Generative AI SDK for generate_content
# import google.generativeai as genai
# genai.configure(api_key="AIzaSyCnPSEGivtGVUDzDDZQKh9gT4Z1tkNRCVM")
# genai_model = genai.GenerativeModel("models/gemini-2.0-flash")

# # LangChain wrapper for Gemini 1.5
# from langchain_google_genai import ChatGoogleGenerativeAI
# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash",
#     google_api_key="AIzaSyCnPSEGivtGVUDzDDZQKh9gT4Z1tkNRCVM"
# )

import os
import logging
import time

# Google Generative AI SDK for generate_content
import google.generativeai as genai

# Load API key from environment variable
genai_api_key = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=genai_api_key)
genai_model = genai.GenerativeModel("models/gemini-2.0-flash")

# LangChain wrapper for Gemini 1.5
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=genai_api_key
)

# Vector & document tools
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Setup logging
logging.basicConfig(level=logging.INFO)

# HuggingFace embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Load or build a vectorstore for multiple PDFs (guidelines)
def load_or_build_vectorstore(pdf_folder: str, index_path: str):
    if os.path.exists(index_path):
        logging.info(f"Loading existing vectorstore from {index_path}.")
        return FAISS.load_local(index_path, embedding, allow_dangerous_deserialization=True)
    else:
        # Get all PDFs in the folder for the given condition
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
        
        all_chunks = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_folder, pdf_file)
            logging.info(f"Processing PDF: {pdf_path}")
            
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
            logging.info(f"Processed {pdf_file} and added {len(chunks)} chunks.")
        
        # Embed the chunks and save to FAISS
        vectordb = FAISS.from_documents(all_chunks, embedding)
        vectordb.save_local(index_path)
        logging.info(f"Vectorstore saved at {index_path}.")
        return vectordb

def assess_scope(name: str, gender: str, dob: str, condition: str, answers: dict):
    try:
        start_time = time.time()

        # Path to the directory containing your PDF files (guidelines)
        pdf_folder = "guidelines"
        index_path = "vectorstore/condition_index"  # Store all vectorized data

        # Build/load vectorstore with all documents
        vectordb = load_or_build_vectorstore(pdf_folder, index_path)

        # Your existing logic for checking red flags
        red_flag_1 = ["age_under_12", "family_history_scarring", "drug_induced", "onset_after_30", "psychosocial_impact"]
        red_flag_2 = ["moderate_severe_symptoms", "widespread_distribution", "differential_diagnosis", "hyperandrogenism_signs", "systemic_symptoms"]
        mild_check = ["mild_acne_typical"]

        for flag in red_flag_1:
            if answers.get(flag) == "yes":
                return {"scope": "Out of Scope", "reason": f"Referral required due to: {flag.replace('_', ' ').capitalize()}.", "time_taken": f"{time.time() - start_time:.2f}s"}

        for flag in red_flag_2:
            if answers.get(flag) == "yes":
                return {"scope": "Out of Scope", "reason": f"Referral required due to: {flag.replace('_', ' ').capitalize()}.", "time_taken": f"{time.time() - start_time:.2f}s"}

        for check in mild_check:
            if answers.get(check) == "no":
                return {"scope": "Out of Scope", "reason": f"Referral required due to: {check.replace('_', ' ').capitalize()}.", "time_taken": f"{time.time() - start_time:.2f}s"}

        return {
            "scope": "In Scope",
            "reason": "Meets pharmacist prescribing criteria.",
            "time_taken": f"{time.time() - start_time:.2f}s"
        }

    except Exception as e:
        logging.error(f"Error while assessing scope: {e}")
        return {
            "scope": "Error",
            "reason": f"Internal error occurred: {str(e)}",
            "source": "",
            "time_taken": f"{time.time() - start_time:.2f}s"
        }

def generate_followup_questions(name, gender, dob, condition, answers, scope):
    case_text = f"Patient Name: {name}\nGender: {gender}\nDOB: {dob}\n" + \
                "\n".join([f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in answers.items()])

    prompt = (
        f"You are a clinical assistant helping a pharmacist follow-up with a patient regarding '{condition}'.\n"
        f"The patient has been flagged as '{scope}'.\n"
        f"Using the initial intake details below, generate exactly 5 highly relevant follow-up questions to better understand the condition progression or severity.\n"
        f"Use the patient's earlier answers as context to create focused, non-generic questions.\n"
        f"DO NOT include formatting, summaries, or numbers. Return only raw questions.\n"
        f"Start your response with '###FOLLOWUP###' and list each question on a new line.\n\n"
        f"Patient Intake:\n{case_text}\n\n"
        f"###FOLLOWUP###"
    )

    response = genai_model.generate_content(prompt)
    raw_text = response.text.strip()

    # Extract clean question lines
    if "###FOLLOWUP###" in raw_text:
        questions_section = raw_text.split("###FOLLOWUP###")[-1].strip()
        return questions_section
    else:
        logging.warning("Gemini response did not include marker. Returning full text.")
        return raw_text



def generate_soap_and_treatment(name, gender, dob, condition, answers, followup_answers, scope):
    from datetime import datetime

    # Calculate age from DOB
    try:
        age = datetime.now().year - int(dob.split("-")[0])
    except:
        age = "unknown"

    # Build a structured summary from answers
    context_lines = "\n".join([
        f"- {k.replace('_', ' ').capitalize()}: {v}" for k, v in answers.items()
    ])

    # Determine scope label
    scope_label = "eligible for pharmacist management" if scope.lower() == "in scope" else "not eligible for pharmacist management"

    # Compose SOAP prompt
    soap_prompt = f"""
You are a clinical assistant helping pharmacists generate SOAP notes for minor ailments.
Write a SOAP note in a structured format (Subjective, Objective, Assessment, Plan)
for a patient presenting with {condition}, who is {scope_label}.

The note should reflect a professional tone. For patients not eligible for pharmacist management, elaborate on why pharmacist care is not suitable, and clearly recommend referral to a physician. Make the assessment clinically sound and documented properly for escalation.

Use this information to generate the note:
---
Patient Info:
- Name: {name}
- Age: {age}
- Gender: {gender}
- Condition: {condition}
{context_lines}
---

Write the note in this structure:

**Subjective:**  
(A clear summary of the patient’s reported symptoms, history, and concerns)

**Objective:**  
- **Appearance:**  
- **Lesion Type:**  
- **Distribution:**  
- **Other Signs:**  
- **Mental Health Impact:**  
- **History:**  

**Assessment:**  
(Clinical interpretation of symptoms and scope determination. Include red flags or atypical presentation if applicable.)

**Plan:**  
(If in scope: recommended treatments and follow-up. If out of scope: advise referral and supportive care. Document recommendations clearly.)
""".strip()

    # Generate SOAP note using LLM
    soap_note = genai_model.generate_content(soap_prompt).text.strip()

    treatment = None

    # Only generate treatment if patient is in scope
    if scope.lower() == "in scope":
        pdf_folder = "guidelines"
        index_path = "vectorstore/condition_index"
        vectordb = load_or_build_vectorstore(pdf_folder, index_path)

        treatment_prompt = (
            f"Based on the British Columbia Acne Guidelines, recommend an evidence-based treatment "
            f"for the following patient SOAP note. Provide a concise recommendation only, in plain text.\n\n"
            f"{soap_note}"
        )

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectordb.as_retriever(),
            return_source_documents=True
        )

        result = chain.invoke({"query": treatment_prompt})
        treatment = result["result"]

    return {
        "soap_note": soap_note,
        "treatment": treatment,
        "referral": scope.lower() != "in scope"
    }

