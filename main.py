# main.py
import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI, APIError # Ensure APIError is imported for specific error handling
from dotenv import load_dotenv
import httpx # Import httpx

# --- Pydantic Models ---
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
# For local development, React often runs on http://localhost:3000
# For production, use your Vercel deployment URL
origins = [
    os.getenv("FRONTEND_URL_DEV", "http://localhost:3000"),
    os.getenv("FRONTEND_URL_PROD"), # This will be your Vercel URL e.g., https://your-app-name.vercel.app
]
# Filter out None values if FRONTEND_URL_PROD is not set (e.g., during early local dev)
origins = [origin for origin in origins if origin]

if not origins: # Fallback if no env vars are set, for very basic local testing
    origins = ["http://localhost:3000", "http://127.0.0.1:3000"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# Initialize OpenAI client
client = None # Initialize client as None globally
try:
    # Explicitly create an httpx.Client instance.
    # httpx by default respects HTTP_PROXY/HTTPS_PROXY environment variables.
    # If you don't want proxies, or need specific proxy settings:
    # http_client_instance = httpx.Client(proxies=None, trust_env=False) # To disable proxies
    # http_client_instance = httpx.Client(proxies="http://yourproxy:port") # To set a specific proxy
    http_client_instance = httpx.Client() # Default behavior, respects env variables

    client = OpenAI(http_client=http_client_instance)
    print("OpenAI client initialized successfully using explicit httpx.Client.")
except Exception as e:
    print(f"CRITICAL: Error initializing OpenAI client: {e}")
    # client remains None, endpoints will check this and return an error.

# --- Helper Function for Error Handling ---
def handle_openai_error(e: Exception, context: str = "OpenAI API call"):
    print(f"ERROR during {context}: {str(e)}")
    if isinstance(e, APIError): # Use the imported APIError
        # Log more details from APIError if available, e.g., e.status_code, e.message
        raise HTTPException(status_code=e.status_code or 500, detail=e.message or "OpenAI API error")
    # For other types of errors, you might want to return a generic error message
    # or handle them differently based on the error type.
    raise HTTPException(status_code=500, detail=f"An error occurred with the AI service: {str(e)}")

# --- API Endpoints ---

@app.api_route("/", methods=["GET", "HEAD"]) # Explicitly allow GET and HEAD for root
async def home():
    if client is None:
        # This check could be here, or you could let subsequent calls fail if client is None.
        # For a home/health check, it's good to indicate service status.
        print("WARN: OpenAI client not initialized, but home endpoint accessed.")
    return {"message": "JournalAIse FastAPI Backend is running!"}

# 1. Audio Transcription Endpoint
@app.post('/api/transcribe-audio', response_model=AudioTranscriptionResponse, responses={500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
async def transcribe_audio_endpoint(audioFile: UploadFile = File(...)):
    if not client:
        raise HTTPException(status_code=503, detail="OpenAI client not initialized. Please check server logs.")
    if not audioFile:
        raise HTTPException(status_code=400, detail="No audio file provided")
    if audioFile.filename == '': # Check if a file was actually selected
        raise HTTPException(status_code=400, detail="No file selected")

    tmp_audio_file_path = None
    try:
        # Ensure the suffix matches the actual file type or convert appropriately.
        # Forcing .wav might be problematic if the input is .mp3, .m4a etc. Whisper handles many formats.
        # Using original suffix or a generic like .tmp might be safer for NamedTemporaryFile if not converting.
        original_suffix = os.path.splitext(audioFile.filename or ".tmp")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=original_suffix) as tmp_audio_file:
            contents = await audioFile.read()
            tmp_audio_file.write(contents)
            tmp_audio_file_path = tmp_audio_file.name
        
        with open(tmp_audio_file_path, "rb") as file_to_transcribe:
            transcript_response = client.audio.transcriptions.create(
                model="whisper-1",
                file=file_to_transcribe
                # You can add options like language, response_format="verbose_json" for more details
            )
        
        raw_transcription = transcript_response.text

        # Call GPT to make the raw transcription broadcast-ready
        correction_prompt = f"""
        Please correct the following raw audio transcription to make it suitable for a news broadcast.
        Focus on clarity, conciseness, removing filler words (like 'um', 'uh', 'like', 'you know'),
        and correcting any grammatical errors or awkward phrasing.
        Ensure the core meaning is preserved. Return only the corrected text.

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
            model="gpt-4" # Or gpt-4 for higher quality if budget allows
        )
        broadcast_ready_text = broadcast_ready_completion.choices[0].message.content.strip()

        return AudioTranscriptionResponse(
            rawTranscription=raw_transcription,
            broadcastReadyText=broadcast_ready_text
        )

    except APIError as e: # Catch OpenAI specific API errors first
        handle_openai_error(e, "audio transcription (OpenAI API)")
    except Exception as e: # Catch other unexpected errors
        print(f"Unexpected error during audio transcription: {e}") # Log the full error for debugging
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during audio processing: {str(e)}")
    finally:
        if tmp_audio_file_path and os.path.exists(tmp_audio_file_path):
            os.remove(tmp_audio_file_path) # Clean up the temporary file
        if audioFile:
            await audioFile.close()


# 2. Script Correction / Assistance Endpoint
@app.post('/api/correct-script', response_model=ScriptCorrectionResponse, responses={500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
async def correct_script_api(request_data: ScriptCorrectionRequest):
    if not client:
        raise HTTPException(status_code=503, detail="OpenAI client not initialized. Please check server logs.")
    
    original_text = request_data.text
    if not original_text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        prompt_text = f"""
        You are an expert script editor for journalists.
        Review the following script segment for grammar, clarity, conciseness, and journalistic style.
        Provide only the corrected version. If there are crucial stylistic suggestions you'd make beyond direct correction,
        integrate them if possible or briefly note them under a 'Suggestions:' heading on a new line after the corrected script.

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
            model="gpt-4", # Or gpt-4
        )
        corrected_content = chat_completion.choices[0].message.content.strip()
        
        corrected_text_part = corrected_content
        suggestions_part = None 
        if "\nSuggestions:" in corrected_content: # Check for newline before "Suggestions:"
            parts = corrected_content.split("\nSuggestions:", 1)
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
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during script correction: {str(e)}")


# 3. AI Script Writer Endpoint
@app.post('/api/generate-script', response_model=AIScriptWriterResponse, responses={500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
async def generate_script_api(request_data: AIScriptWriterRequest):
    if not client:
        raise HTTPException(status_code=503, detail="OpenAI client not initialized. Please check server logs.")
    
    topic = request_data.topic
    if not topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")

    try:
        script_prompt = f"""
        You are an AI content writer specializing in creating detailed, well-researched, and informative articles.
        Write an extensive article on the following topic. The article should be at least 500-700 words.

        The article must be formatted as follows:
        1. The main title of the article should be the topic itself, in bold.
        2. Immediately under the bolded title, include the byline: "By [Your Name]"

        Ensure the article is well-structured with clear paragraphs. You can use subheadings if it helps organize the content for a long article.
        The tone should be informative and engaging for a general audience interested in learning more about the topic.
        Provide comprehensive coverage of the topic.

        ---
        Topic: {topic}
        ---

        Article:
        """

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an AI scriptwriter for a news program, skilled in creating engaging and informative content."},
                {"role": "user", "content": script_prompt}
            ],
            model="gpt-4", # Or gpt-4 for potentially better structure/creativity
            # temperature=0.7 # Adjust for creativity
        )
        generated_script = chat_completion.choices[0].message.content.strip()

        return AIScriptWriterResponse(script=generated_script)
    except APIError as e:
        handle_openai_error(e, "AI script generation (OpenAI API)")
    except Exception as e:
        print(f"Unexpected error during AI script generation: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during script generation: {str(e)}")

# To run locally (for development):
# uvicorn main:app --reload --port 8000
# Ensure your .env file with OPENAI_API_KEY is present in the same directory.
