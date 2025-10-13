import time

from objectify import classify_request, mode_select
from vision import fuse_yolo_midas, setup_yolo, setup_midas
from speechrecog import listen_for_command, get_voice_input
from googleTTS import speak_text


def reading_mode():
    """Mode 2: Text reading (placeholder)"""
    print("Reading mode")
    speak_text("Reading mode active")
    time.sleep(2)
