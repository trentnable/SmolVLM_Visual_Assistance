from gtts import gTTS
import pygame
import tempfile
import os

# Initialize audio playback
pygame.mixer.init()

def speak_text(text):
    """
    Convert text to speech using Google TTS and play it.
    
    Args:
        text: String to be spoken
    """
    try:
        # Create TTS object
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_file = fp.name
            tts.save(temp_file)
        
        # Load and play audio
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()
        
        #Await playback to finish
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        # Delete temp
        os.unlink(temp_file)
        
    except Exception as e:
        print(f"TTS Error: {e}")