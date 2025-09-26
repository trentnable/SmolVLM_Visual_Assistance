import torch
import whisper
import soundfile as sf
import numpy as np
import librosa
import time

# print("torch = ", torch.__version__)
# print("whisper = ", whisper.__version__)
# print("soundfile = ", soundfile.__version__)
# print("numpy = ", numpy.__version__)
# print("librosa = ", librosa.__version__)


# Config
AUDIO_FILE = "/home/sdesign/Documents/noah/audio/harvard.wav"
MODEL_NAME = "small"
CHUNK_SECONDS = 30

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
print(f"Loading Whisper model '{MODEL_NAME}' on {DEVICE}...")
model = whisper.load_model(MODEL_NAME, device=DEVICE)

start_time = time.time()

# Load and preprocess audio
audio, sr = sf.read(AUDIO_FILE, dtype='float32')

if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    sr = 16000

# Split into chunks
chunk_size = CHUNK_SECONDS * sr
chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]

# Transcribe chunks
transcription = ""
for i, chunk in enumerate(chunks, 1):
    result = model.transcribe(chunk, fp16=False)  # fp16=False for CPU
    transcription += result["text"].strip() + " "

transcription = transcription.strip()
print("\nTranscription:\n")
print(transcription)

end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")