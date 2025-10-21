import time
import threading
import cv2
import keyboard
import math # May still be needed if processing results here

# --- Import necessary functions ---
from vision import fuse_yolo_midas, setup_yolo, setup_midas 
# Assuming classNames might be needed for interpreting results for speech
# If classNames is defined in vision.py or elsewhere, import it:
# from vision import classNames # Or from wherever it's defined

from speechrecog import get_voice_input
from googleTTS import speak_text
from objectify import classify_request, mode_select # Keep command classification
from mode1 import cleanup as mode1_cleanup # Keep cleanup for keyboard unhook

# Define classNames here if not imported (ensure it matches the model)
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ] # Make sure this list is accurate

# --- Main Application ---
def main():
    start_time = time.time()

    # Load models
    print("Loading models...")
    yolo_model, _ = setup_yolo("yolov8n.pt") # Ensure this model file exists
    midas, transform = setup_midas("MiDaS_small")
    print(f"Models loaded: {time.time() - start_time:.2f}s\n")

    # Initialize Webcam
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("--- FATAL ERROR: Could not open camera. Exiting. ---")
        speak_text("Error: Could not open camera.")
        exit()
    print("Camera opened successfully.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Optional: Set width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # Optional: Set height

    print("Starting continuous detection. Press 'q' in the window to quit.")
    print("Press 'm' to activate voice command.")
    
    last_detected_objects = [] # Store details from fuse_yolo_midas for speech

    try:
        while True:
            # --- Read Frame ---
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame. Exiting loop.")
                speak_text("Camera feed lost.")
                break

            # --- Process Frame using function from vision.py ---
            # Call fuse_yolo_midas. We don't need class_id here as we want all objects.
            # Pass None or a specific value if the function requires it. Let's assume None means detect all.
            try:
                # Assuming class_id=None tells it to detect everything
                objects, _, annotated_frame, degrees, horizontal, vertical, depth_category = \
                    fuse_yolo_midas(frame, yolo_model, midas, transform, class_id=None) 
                
                # Store the detected objects list for voice commands
                last_detected_objects = objects if objects else [] 

            except Exception as e:
                print(f"Error during vision processing: {e}")
                annotated_frame = frame # Show raw frame on error
                last_detected_objects = []


            # --- Display Frame ---
            cv2.imshow("Live Object Detection", annotated_frame)

            # --- Check for Quit Key ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit key (q) pressed. Exiting.")
                break
            
            # --- Check for Voice Command Activation Key ---
            if keyboard.is_pressed('m'):
                print("\n'm' key detected. Activating voice command...")
                time.sleep(0.3) # Simple debounce

                command = get_voice_input(duration=5) 
                
                if command and command != "None":
                    # --- Process Command ---
                    # Describe what was *last* seen based on last_detected_objects.
                    # You could add logic here to use classify_request(command) 
                    # to filter last_detected_objects for a specific item if needed.
                    
                    if last_detected_objects:
                        description = f"I currently see {len(last_detected_objects)} object{'s' if len(last_detected_objects) != 1 else ''}. "
                        # Count object types
                        object_labels = [d.get('label', 'unknown') for d in last_detected_objects]
                        counts = {label: object_labels.count(label) for label in set(object_labels)}
                        
                        description_parts = []
                        for label, count in counts.items():
                             description_parts.append(f"{count} {label}{'s' if count > 1 else ''}")
                        
                        description += ", ".join(description_parts) + "."
                        
                        print("Speaking:", description)
                        speak_text(description)
                    else:
                        print("Speaking: I don't see any objects clearly right now.")
                        speak_text("I don't see any objects clearly right now.")
                
                # Wait until 'm' is released
                while keyboard.is_pressed('m'):
                    time.sleep(0.1) 
                print("Ready for 'm' key again.")

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Shutting down...")

    finally:
        # --- Cleanup ---
        print("Releasing camera and closing windows...")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        mode1_cleanup() # Call cleanup from mode1 (e.g., unhook keyboard)
        total_time = time.time() - start_time
        print(f"Total runtime: {total_time:.2f}s")

if __name__ == "__main__":
    main()
