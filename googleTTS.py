from gtts import gTTS
import pygame
from io import BytesIO

# Initialize audio playback
pygame.mixer.init()

def speak_text(text):
    """
    Convert text to speech using Google TTS and play it.
    Uses in-memory buffer instead of temp files.
    
    Args:
        text: String to be spoken
    """
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save to buffer
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)  # Reset buffer position to beginning
        
        # Load audio from buffer
        pygame.mixer.music.load(audio_buffer, 'mp3')
        pygame.mixer.music.play()
        
        # Await buffer to finish
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        # Buffer cleanup
        
    except Exception as e:
        print(f"TTS Error: {e}")