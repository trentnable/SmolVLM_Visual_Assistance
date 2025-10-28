import sounddevice as sd
import numpy as np
import whisper
import torch

MODEL_NAME = "tiny"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model(MODEL_NAME, device=DEVICE)
SAMPLE_RATE = 16000

# Load Silero VAD model (lightweight voice activity detection)
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                   model='silero_vad',
                                   force_reload=False)
(get_speech_timestamps, _, _, _, _) = utils

def listen_for_keyword_with_vad(
    keyword="hello",
    vad_threshold=0.5,
    record_duration=5,
    silence_threshold=0.01
):
    """
    Uses VAD to detect speech, then checks for keyword with Whisper.
    Much more efficient than continuous transcription.
    """
    print(f"Listening for keyword: '{keyword}'...")
    
    while True:
        # Record a chunk
        audio_chunk = sd.rec(
            int(2 * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        audio_chunk = audio_chunk.flatten()
        
        # Convert to tensor for VAD
        audio_tensor = torch.from_numpy(audio_chunk)
        
        # Check if speech is present (much faster than Whisper)
        speech_timestamps = get_speech_timestamps(
            audio_tensor, 
            vad_model,
            sampling_rate=SAMPLE_RATE,
            threshold=vad_threshold
        )
        
        # Only transcribe if speech detected
        if len(speech_timestamps) > 0:
            text = transcribe_audio(audio_chunk).lower()
            
            if keyword.lower() in text:
                print(f"Keyword '{keyword}' detected! Recording command...")
                
                # Record the full command
                audio = sd.rec(
                    int(record_duration * SAMPLE_RATE),
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype='float32'
                )
                sd.wait()
                audio = audio.flatten()
                
                # Silence trimming
                mask = np.abs(audio) > silence_threshold
                if np.any(mask):
                    indices = np.where(mask)[0]
                    audio = audio[indices[0]:indices[-1]+1]
                
                print("Processing speech...")
                return transcribe_audio(audio)

def transcribe_audio(audio):
    audio = (audio * 32768).astype(np.int16)
    audio = audio.astype(np.float32) / 32768.0
    result = model.transcribe(audio, fp16=(DEVICE == "cuda"))
    return result["text"].strip()