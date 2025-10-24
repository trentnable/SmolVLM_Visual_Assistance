import torch
import whisper
import sounddevice as sd
import numpy as np
import time

# Config
MODEL_NAME = "tiny"
SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
global model
model = whisper.load_model(MODEL_NAME, device=DEVICE)

# def record_audio(duration=5):
#     """Record audio from the default microphone"""
#     print(f"Recording for {duration} seconds... Speak now!")
#     audio = sd.rec(int(duration * SAMPLE_RATE), 
#                    samplerate=SAMPLE_RATE, 
#                    channels=1, 
#                    dtype='float32')
#     sd.wait()  # Wait until recording is finished
#     print("Recording finished")
#     return audio.flatten()

def get_voice_input(duration=5):
    """Record and transcribe voice command"""
    print(f"Listening for {duration} seconds...")
    command = listen_for_command(duration)
    print(f"You said: '{command}'")
    return command

def transcribe_audio(audio):
    audio = (audio * 32768).astype(np.int16)
    audio = audio.astype(np.float32) / 32768.0

    start_time = time.time()
    result = model.transcribe(audio, fp16=(DEVICE == "cuda"))
    end_time = time.time()

    print(f"Transcription time: {end_time - start_time:.2f} seconds")
    return result["text"].strip()

# def record_and_transcribe(duration=5):
#     """Record audio from microphone and transcribe it"""
#     audio = record_audio(duration)
#     return transcribe_audio(audio)

def listen_for_command(duration=5, silence_threshold=0.01):
    """Listen for voice command and transcribe into string"""
    print(f"Listening for command (max {duration} seconds)... Speak now!")
    
    # Record audio
    audio = sd.rec(int(duration * SAMPLE_RATE), 
                   samplerate=SAMPLE_RATE, 
                   channels=1, 
                   dtype='float32')
    sd.wait()
    audio = audio.flatten()
    
    # silence trimming
    mask = np.abs(audio) > silence_threshold
    if np.any(mask):
        indices = np.where(mask)[0]
        audio = audio[indices[0]:indices[-1]+1]
    
    print("Processing speech...")
    return transcribe_audio(audio)