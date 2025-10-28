import time

start_time = time.time()
from objectify import mode_select
from vision import setup_yolo, setup_midas
from speechrecog import get_voice_input
from googleTTS import speak_text
from mode1 import object_location, wait_for_mic, cleanup, reset_mode_state
from mode2 import reading_mode


def main():
    
    # Load models
    print("Loading models...")
    yolo_model, _ = setup_yolo("yolo11n.pt")
    midas, transform = setup_midas("MiDaS_small")
    print(f"Setup complete: {time.time() - start_time:.2f}s\n")
    
    try:
        while True:
            # Wait for activation with 'm' key (Outside of mode selection)
            wait_for_mic()
            
            # Get voice command to select mode
            command = get_voice_input(duration=5)
            
            # Mode selection
            mode = mode_select(command)
            
            if mode == "one":
                #Loop detection and reprompt until 'c' clears mode              
                object_location(yolo_model, midas, transform, command)
            elif mode == "two":
                reading_mode()
            elif mode == "null":
                print("Unclear mode selection, try again")
                speak_text("Unclear mode selection, try again")
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