# main.py
import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI, APIError
from dotenv import load_dotenv
from typing import Annotated # For FastAPI versions requiring Annotated for Body

# Import Pydantic models (if using a separate schemas.py)
# from core.schemas import (
#     ScriptCorrectionRequest, ScriptCorrectionResponse,
#     AIScriptWriterRequest, AIScriptWriterResponse,
#     AudioTranscriptionResponse, ErrorResponse
# )
# Or define them directly in main.py if preferred for simplicity

# --- Pydantic Models (if not in schemas.py) ---
from pydantic import BaseModel
from typing import Optional

class ScriptCorrectionRequest(BaseModel):
    text: str

class ScriptCorrectionResponse(BaseModel):
    correctedText: str
    suggestions: Optional[str] = None

class AIScriptWriterRequest(BaseModel):
    topic: str

class AIScriptWriterResponse(BaseModel):
    script: str

class AudioTranscriptionResponse(BaseModel):
    rawTranscription: str
    broadcastReadyText: str

class ErrorResponse(BaseModel):
    error: str
# --- End Pydantic Models ---


# Load environment variables from .env file (for local development)
load_dotenv()

app = FastAPI(title="JournalAIse API")

# CORS Configuration
# Adjust origins based on your frontend URLs
origins = [
    os.getenv("FRONTEND_URL_DEV", "http://localhost:3000"), # For local React dev server
    os.getenv("FRONTEND_URL_PROD"),                         # For your deployed Vercel frontend
    # Add other origins if necessary
]
# Filter out None values from origins if FRONTEND_URL_PROD is not set
origins = [origin for origin in origins if origin]

if not origins: # Fallback if no env vars are set, for very basic local testing
    origins = ["http://localhost:3000"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# Initialize OpenAI client
# The API key will be picked up from the OPENAI_API_KEY environment variable
try:
    client = OpenAI()
    # Test with a simple model listing to ensure API key is working (optional)
    # models = client.models.list()
    # print("OpenAI client initialized successfully.")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None # Set to None if initialization fails

# --- Helper Function for Error Handling ---
def handle_openai_error(e: Exception, context: str = "OpenAI API call"):
    print(f"Error during {context}: {str(e)}")
    if isinstance(e, APIError):
        raise HTTPException(status_code=e.status_code or 500, detail=e.message or "OpenAI API error")
    raise HTTPException(status_code=500, detail=f"An error occurred with the AI service: {str(e)}")

# --- API Endpoints ---

@app.get('/')
async def home():
    return {"message": "JournalAIse FastAPI Backend is running!"}

# 1. Audio Transcription Endpoint
@app.post('/api/transcribe-audio', response_model=AudioTranscriptionResponse, responses={500: {"model": ErrorResponse}})
async def transcribe_audio_endpoint(audioFile: UploadFile = File(...)):
    if not client:
        raise HTTPException(status_code=503, detail="OpenAI client not initialized.")
    if not audioFile:
        raise HTTPException(status_code=400, detail="No audio file provided")
    if audioFile.filename == '':
        raise HTTPException(status_code=400, detail="No selected file")

    tmp_audio_file_path = None
    try:
        # Save the audio file temporarily
        # Suffix should ideally match the file type, or convert it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
            contents = await audioFile.read()
            tmp_audio_file.write(contents)
            tmp_audio_file_path = tmp_audio_file.name
        
        with open(tmp_audio_file_path, "rb") as file_to_transcribe:
            transcript_response = client.audio.transcriptions.create(
                model="whisper-1",
                file=file_to_transcribe
            )
        
        raw_transcription = transcript_response.text

        # TODO: Implement actual broadcast-ready text correction with another GPT call
        # This prompt should be refined for better results.
        correction_prompt = f"""
        Please correct the following raw audio transcription to make it suitable for a news broadcast.
        Focus on clarity, conciseness, removing filler words (like 'um', 'uh', 'like', 'you know'),
        and correcting any grammatical errors or awkward phrasing.
        Ensure the core meaning is preserved.

        Raw Transcription:
        ---
        {raw_transcription}
        ---

        Corrected Broadcast-Ready Text:
        """
        broadcast_ready_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert editor transforming raw audio transcriptions into polished, broadcast-ready text."},
                {"role": "user", "content": correction_prompt}
            ],
            model="gpt-3.5-turbo" # Or gpt-4 for higher quality
        )
        broadcast_ready_text = broadcast_ready_completion.choices[0].message.content.strip()

        return AudioTranscriptionResponse(
            rawTranscription=raw_transcription,
            broadcastReadyText=broadcast_ready_text
        )

    except APIError as e:
        handle_openai_error(e, "audio transcription (OpenAI API)")
    except Exception as e:
        print(f"Unexpected error during audio transcription: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if tmp_audio_file_path and os.path.exists(tmp_audio_file_path):
            os.remove(tmp_audio_file_path) # Clean up the temporary file
        if audioFile:
            await audioFile.close()


# 2. Script Correction / Assistance Endpoint
@app.post('/api/correct-script', response_model=ScriptCorrectionResponse, responses={500: {"model": ErrorResponse}})
async def correct_script_api(request_data: ScriptCorrectionRequest):
    if not client:
        raise HTTPException(status_code=503, detail="OpenAI client not initialized.")
    
    original_text = request_data.text

    try:
        prompt_text = f"""
        You are an expert script editor for journalists.
        Review the following script segment for grammar, clarity, conciseness, and journalistic style.
        Provide a corrected version. If there are stylistic suggestions you'd make beyond direct correction,
        you can briefly note them after the corrected script under a 'Suggestions:' heading.

        Original Script:
        ---
        {original_text}
        ---

        Corrected Script:
        """

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert script editor for journalists."},
                {"role": "user", "content": prompt_text}
            ],
            model="gpt-3.5-turbo",
        )
        corrected_content = chat_completion.choices[0].message.content.strip()
        
        corrected_text_part = corrected_content
        suggestions_part = None # Use None for optional field
        if "Suggestions:" in corrected_content:
            parts = corrected_content.split("Suggestions:", 1)
            corrected_text_part = parts[0].strip()
            suggestions_part = parts[1].strip()

        return ScriptCorrectionResponse(
            correctedText=corrected_text_part,
            suggestions=suggestions_part
        )
    except APIError as e:
        handle_openai_error(e, "script correction (OpenAI API)")
    except Exception as e:
        print(f"Unexpected error during script correction: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


# 3. AI Script Writer Endpoint
@app.post('/api/generate-script', response_model=AIScriptWriterResponse, responses={500: {"model": ErrorResponse}})
async def generate_script_api(request_data: AIScriptWriterRequest):
    if not client:
        raise HTTPException(status_code=503, detail="OpenAI client not initialized.")
    
    topic = request_data.topic

    try:
        script_prompt = f"""
        You are an AI scriptwriter for a news/informational program called "JournalAIse: Insights Today".
        Generate a foundational script (approximately 200-300 words) for a segment on the following topic: {topic}.
        The script should be engaging, informative, and suitable for a general audience.
        Include:
        - A brief introduction by a host.
        - Key points or questions about the topic.
        - Optionally, a placeholder for an expert's input or a visual element.
        - A brief concluding remark by the host.
        Format the script clearly, perhaps using "HOST:" or "EXPERT:" labels.
        ---
        Topic: {topic}
        ---
        Generated Script:
        """

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an AI scriptwriter for a news program."},
                {"role": "user", "content": script_prompt}
            ],
            model="gpt-3.5-turbo",
        )
        generated_script = chat_completion.choices[0].message.content.strip()

        return AIScriptWriterResponse(script=generated_script)
    except APIError as e:
        handle_openai_error(e, "AI script generation (OpenAI API)")
    except Exception as e:
        print(f"Unexpected error during AI script generation: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# To run locally (for development):
# uvicorn main:app --reload --port 8000
# (Port 8000 is common for FastAPI, adjust if needed)
