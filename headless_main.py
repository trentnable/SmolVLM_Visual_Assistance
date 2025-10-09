import time
start_time = time.time()
from objectify import classify_request, mode_select
from vision import fuse_yolo_midas, to_base64_img, setup_yolo, setup_midas
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from ultralytics import YOLO
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
from speechrecog import listen_for_command
from googleTTS import speak_text
import keyboard


def main():

    # Setup webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Setup models
    print("Loading YOLO model...")
    yolo_model, class_names = setup_yolo("yolo11n.pt")

    print("Loading MiDaS depth model...")
    midas, transform = setup_midas("MiDaS_small")
    
    setup_time = time.time() - start_time
    print(f"\nSetup Time: {setup_time}")
    try:
        while True:

            # Initialize variables
            task = False
            detect = 0
            loop_start = time.time()

            # Get mic command
            print("Awaiting 'm' press for mic")
            keyboard.wait('m')

            duration = 5
            print(f"Listening({duration} seconds)")
            command = listen_for_command(duration)
            command = "Help me find my water bottle."
            print(f"User said: {command}")
            
            select = "Mode Selection"
            mode = mode_select(command)


            # Mode_1 (Locate)
            if mode == "one":

                # Classify
                print("Classifying request...")
                out1 = int(classify_request(command))
                print(f"Helping to locate {yolo_model.names[out1]}")
                print("\nStarting detection (press Ctrl+C to stop)...")

                while task == False:
                    # Webcam Capture
                    ret, frame = cap.read()
                    if not ret:
                            print("Failed to capture frame from webcam")
                            break
                        
                    # fuse YOLO and MiDaS
                    objects, depth_map, annotated_frame, degrees, horizontal, vertical, depth_category = fuse_yolo_midas(
                    frame, yolo_model, midas, transform, class_id=out1
                    )
                        
                    cv2.imshow('Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    loop_time = time.time() - loop_start

                    # Build speech text
                    speech_text = ""
                        
                    # Results
                    if objects:
                        print(f"Found {len(objects)} object(s):")
                        speech_text += f"Found {len(objects)} object"
                        if len(objects) > 1:
                            speech_text += "s"
                            speech_text += ". "
                            
                        for i, obj in enumerate(objects):
                            print(f"  Object {i+1}: {obj['label']}")
                            print(f"    Position: {obj['position']}")
                            print(f"    Distance: {obj['depth_category']}")
                            if obj['degrees'] is not None:
                                print(f"    Angle: {obj['degrees']:.1f} degrees")
                            
                            # Direction info
                        print(f"\nDirection: {vertical}, {horizontal}")
                        speech_text += f"Direction is {vertical}. "
                            
                        if degrees is not None:
                            print(f"Angle from center: {degrees:.1f} degrees")
                            speech_text += f", {degrees:.0f} degrees to the {horizontal}. "
                            
                        print(f"Distance: {depth_category}")
                        speech_text += f"Distance is {depth_category}."
                            
                        # Speak all added text
                        speak_text(speech_text)

                        #Mark as detected
                        detect += 1
                            
                    else:
                        print("No objects detected")
                        speak_text("No objects detected")

                        if detect > 0 or loop_time > 30:
                            task = True
                            cv2.destroyAllWindows()
                            cap.release()
                        
                    
                        
                        
                    # while pygame.mixer.music.get_busy():
                    #     pygame.time.Clock().tick(10)
                    #     if cv2.waitKey(1) & 0xFF == ord('q'):
                    #         break
                        
                    # 1 FPS
                    time.sleep(0.1)

            elif mode == "two":
                print("Helping with reading")

            else:
                print("general mode selection error")
 


    except KeyboardInterrupt:
        print("\n\nStopping detection...")
    finally:
                            
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nTotal elapsed time: {elapsed_time:.4f} seconds")
    

if __name__ == "__main__":
    main()