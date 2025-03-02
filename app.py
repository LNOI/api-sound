from time import time   
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv
from uuid import uuid4
from huggingsound import SpeechRecognitionModel
from loguru import logger
import os
app = FastAPI()

load_dotenv()


model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")
os.makedirs("./audio", exist_ok=True)

async def transcribe_audio(audio_paths: List[str]):
    start_time = time()
    transcriptions = model.transcribe(audio_paths)
    end_time = time()
    print(f"Time taken: {end_time - start_time} seconds")
    return transcriptions

@app.post("/speech-to-text")
async def audio_to_text(audio_files: List[UploadFile] = File(...)):
    """
    Transcribe an audio file from MinIO and save the result to the database.

    Args:
        audio_file (UploadFile): The audio file to transcribe.
    """
    # 
    try:
        audio_paths = []
        for audio_file in audio_files:
            audio_path = f"./audio/{uuid4()}.wav"
            with open(audio_path, "wb") as f:
                f.write(await audio_file.read())
            audio_paths.append(audio_path)
    
        transcriptions = await transcribe_audio(audio_paths)
        
        return {
            "transcriptions": [
                transcription["transcription"]
                for transcription in transcriptions
            ]
        }
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
