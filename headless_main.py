import time
import threading
import cv2
import keyboard

start_time = time.time()
from objectify import classify_request, mode_select
from vision import fuse_yolo_midas, setup_yolo, setup_midas
from speechrecog import listen_for_command, get_voice_input
from googleTTS import speak_text
from mode1 import detection_loop, build_detection_speech, print_results, object_location, on_key_press, wait_for_mic, cleanup, reset_state
from mode2 import reading_mode


def main():
    
    # Load models
    print("Loading models...")
    yolo_model, _ = setup_yolo("yolo11n.pt")
    midas, transform = setup_midas("MiDaS_small")
    print(f"Setup complete: {time.time() - start_time:.2f}s\n")
    
    try:
        while True:
            # Wait for activation
            wait_for_mic()
            
            # Get voice command
            command = get_voice_input(duration=5)
            
            # Determine mode
            mode = mode_select(command)
            
            if mode == "one":
                object_location(yolo_model, midas, transform, command)
            elif mode == "two":
                reading_mode()
            else:
                print("Mode selection error")
            
            # Reset state
            reset_mode_state()
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    finally:
        cleanup()
        total_time = time.time() - start_time
        print(f"Total runtime: {total_time:.2f}s")


if __name__ == "__main__":
    main()