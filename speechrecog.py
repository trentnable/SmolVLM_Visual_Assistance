import torch
import whisper
import sounddevice as sd
import numpy as np
import time

# Config
MODEL_NAME = "small"
SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
global model
model = whisper.load_model(MODEL_NAME, device=DEVICE)

def record_audio(duration=5):
    """
    Record audio from the default microphone.
    
    Args:
        duration: Recording duration in seconds (default 5)
    
    Returns:
        numpy array of audio samples
    """
    print(f"Recording for {duration} seconds... Speak now!")
    audio = sd.rec(int(duration * SAMPLE_RATE), 
                   samplerate=SAMPLE_RATE, 
                   channels=1, 
                   dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording finished!")
    return audio.flatten()

def transcribe_audio(audio):
    """
    Transcribe audio buffer using Whisper.
    
    Args:
        audio: numpy array of audio samples (16kHz, mono, float32)
    
    Returns:
        Transcribed text string
    """
    model = load_model()
    result = model.transcribe(audio, fp16=(DEVICE == "cuda"))
    return result["text"].strip()

def record_and_transcribe(duration=5):
    """
    Record audio from microphone and transcribe it.
    
    Args:
        duration: Recording duration in seconds (default 5)
    
    Returns:
        Transcribed text string
    """
    audio = record_audio(duration)
    return transcribe_audio(audio)

def listen_for_command(duration=5, silence_threshold=0.01):
    """
    Listen for voice command with basic silence detection.
    
    Args:
        duration: Maximum recording duration in seconds
        silence_threshold: Audio level below which is considered silence
    
    Returns:
        Transcribed text string
    """
    print(f"Listening for command (max {duration} seconds)... Speak now!")
    
    # Record audio
    audio = sd.rec(int(duration * SAMPLE_RATE), 
                   samplerate=SAMPLE_RATE, 
                   channels=1, 
                   dtype='float32')
    sd.wait()
    audio = audio.flatten()
    
    # Simple silence trimming (remove very quiet parts at start/end)
    mask = np.abs(audio) > silence_threshold
    if np.any(mask):
        indices = np.where(mask)[0]
        audio = audio[indices[0]:indices[-1]+1]
    
    print("Processing speech...")
    return transcribe_audio(audio)


# Example usage when run directly
if __name__ == "__main__":
    print("\n=== Microphone Speech Recognition Test ===\n")
    
    # Test recording and transcription
    start_time = time.time()
    
    text = record_and_transcribe(duration=5)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n--- Transcription Result ---")
    print(f"Text: {text}")
    print(f"\nElapsed time: {elapsed_time:.4f} seconds")