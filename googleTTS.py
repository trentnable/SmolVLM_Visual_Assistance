from gtts import gTTS
import pygame, threading
from io import BytesIO

# Initialize audio playback
pygame.mixer.init()

# stop flag for tts thread
stop_tts_flag = threading.Event()

def speak_text(text):
    """
    Convert text to speech using Google TTS and play it.
    Uses in-memory buffer instead of temp files.
    
    Args:
        text: String to be spoken
    """
    try:
        # clear the thread stop flag
        stop_tts_flag.clear()

        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save to buffer
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)  # Reset buffer position to beginning
        
        # check if the thread should stop before playing the audio
        if stop_tts_flag.is_set():
            audio_buffer.close()
            return

        # Load audio from buffer
        pygame.mixer.music.load(audio_buffer, 'mp3')
        pygame.mixer.music.play()
        
        # Await buffer to finish
        while pygame.mixer.music.get_busy() and not stop_tts_flag.is_set():
            pygame.time.Clock().tick(10)

        # stop the tts if thread flag is set
        if stop_tts_flag.is_set():
            pygame.mixer.music.stop()
        
        # Buffer cleanup
        audio_buffer.close()
        
    except Exception as e:
        print(f"TTS Error: {e}")

def stop_speech():
    stop_tts_flag.set()
    pygame.mixer.music.stop()