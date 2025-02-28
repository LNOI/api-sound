from huggingsound import SpeechRecognitionModel
import librosa
import soundfile as sf
import time

model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")
start_time = time.time()
file_path = "./audio_16k.wav"
y, sr = librosa.load(file_path, sr=None)  # Giữ nguyên sample rate gốc

if sr != 16000:
    print("Resampling audio to 16000 Hz")
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)
    sf.write("audio_16k.wav", y_resampled, 16000)  # Lưu lại file mới
    
    

audio_paths = ["./audio_16k.wav"]


transcriptions = model.transcribe(audio_paths)
print(transcriptions)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

start_time = time.time()
transcription = "why are you gay?"


