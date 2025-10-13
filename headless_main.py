import time
import threading
import warnings
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import keyboard

from objectify import classify_request, mode_select
from vision import fuse_yolo_midas, to_base64_img, setup_yolo, setup_midas
from speechrecog import listen_for_command
from googleTTS import speak_text
from ultralytics import YOLO


# Thread-safe event flags

mic_event = threading.Event()
stop_event = threading.Event()
last_press_time = 0


# Keyboard callback

def my_callback(event):
    global last_press_time
    # debounce: ignore if key pressed too quickly
    if time.time() - last_press_time < 0.3:
        return
    last_press_time = time.time()

    if event.name == 'm':
        # If not recording, start mic
        if not mic_event.is_set() and not stop_event.is_set():
            mic_event.set()
            print("m pressed → microphone activated")
        # If currently recording or detecting, stop
        elif not stop_event.is_set():
            stop_event.set()
            print("m pressed → stop signal sent")

# Register callback
keyboard.on_press(my_callback)



# Main application

def main():
    start_time = time.time()

    # Setup models
    print("Loading YOLO model...")
    yolo_model, class_names = setup_yolo("yolo11n.pt")

    print("Loading MiDaS depth model...")
    midas, transform = setup_midas("MiDaS_small")

    setup_time = time.time() - start_time
    print(f"\nSetup Time: {setup_time:.2f} seconds")

    try:
        while True:
            detect = 0
            stop_event.clear()
            mic_event.clear()

            # Wait for microphone activation
            print("Awaiting 'm' press for mic")
            speak_text("Awaiting 'm' press for mic")
            mic_event.wait()  
            mic_event.clear()
            

            # Start listening
            duration = 5
            print(f"Listening ({duration} seconds)")
            command = listen_for_command(duration)
            print(f"User said: {command}")
            keyboard.send('m') #simulates key press (temp fix)

            mode = mode_select(command)

            
            # MODE 1
            
            if mode == "one":
                print("Initializing webcam...")
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                if not cap.isOpened():
                    print("Error: Could not open webcam")
                    return

                print("Classifying request...")
                out1 = int(classify_request(command))
                target_label = yolo_model.names[out1]
                print(f"Helping to locate {target_label}")
                speak_text(f"Helping to locate {target_label}")
                print("\nStarting detection (press 'm' again to stop)...")

                loop_start = time.time()
                stop_event.clear()

                while not stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to capture frame from webcam")
                        break

                    # Run fused YOLO + MiDaS inference
                    (
                        objects,
                        depth_map,
                        annotated_frame,
                        degrees,
                        horizontal,
                        vertical,
                        depth_category
                    ) = fuse_yolo_midas(frame, yolo_model, midas, transform, class_id=out1)

                    cv2.imshow('Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    loop_time = time.time() - loop_start
                    speech_text = ""

                    # Process detections
                    if objects:
                        print(f"Found {len(objects)} object(s):")
                        speech_text += f"Found {len(objects)} object"
                        if len(objects) > 1:
                            speech_text += "s. "
                        else:
                            speech_text += ". "

                        for i, obj in enumerate(objects):
                            print(f"  Object {i+1}: {obj['label']}")
                            print(f"    Position: {obj['position']}")
                            print(f"    Distance: {obj['depth_category']}")
                            if obj['degrees'] is not None:
                                print(f"    Angle: {obj['degrees']:.1f} degrees")

                        print(f"\nDirection: {vertical}, {horizontal}")
                        speech_text += f"Direction is {vertical}. "

                        if degrees is not None:
                            print(f"Angle from center: {degrees:.1f} degrees")
                            speech_text += f", {degrees:.0f} degrees to the {horizontal}. "

                        print(f"Distance: {depth_category}")
                        speech_text += f"Distance is {depth_category}."
                        speak_text(speech_text)
                        detect += 1
                    else:
                        print("No objects detected")
                        speak_text("No objects detected")

                    # Stop conditions
                    if stop_event.is_set():
                        print("Stop signal detected (m pressed again).")
                        break

                    if (loop_time > 30 and detect != 0) or (loop_time > 60 and detect == 0):
                        reason = "timed out" if detect == 0 else "completed"
                        print(f"Detection {reason}, returning to default.")
                        speak_text(f"Object location {reason}, returning to default.")
                        break

                    time.sleep(0.1)

                # Cleanup
                cv2.destroyAllWindows()
                cap.release()
                stop_event.clear()
                mic_event.clear()

            
            # MODE 2: Reading
            
            elif mode == "two":
                print("Helping with reading")
                speak_text("Helping with reading mode")

            else:
                print("General mode selection error")

    except KeyboardInterrupt:
        print("\n\nStopping detection...")

    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nTotal elapsed time: {elapsed_time:.4f} seconds")



# Entry point

if __name__ == "__main__":
    main()
