import time
import threading
import keyboard
import cv2

# Assuming these imports are correct and files exist
from objectify import classify_request # Removed mode_select as it's used in headless_main
from vision import fuse_yolo_midas # Removed setup_yolo, setup_midas as they are called in headless_main
from speechrecog import listen_for_command, get_voice_input # Assuming these are used elsewhere or potentially needed later
from googleTTS import speak_text

# Global state (Remains the same)
class AppState:
    def __init__(self):
        self.mic_ready = threading.Event()
        self.stop_requested = threading.Event()
        self.in_detection = False
        self.last_press_time = 0
        self.lock = threading.Lock()

state = AppState()

# on_key_press function (Remains the same)
def on_key_press(event):
    """Handle 'm' key press with debouncing"""
    if event.name != 'm':
        return
    
    current_time = time.time()
    with state.lock:
        if current_time - state.last_press_time < 0.3:
            return
        state.last_press_time = current_time
        
        if state.in_detection:
            state.stop_requested.set()
            print("Stop detection requested via 'm' key.") # Added clarity
        else:
            state.mic_ready.set()
            print("'m' key pressed - Mic ready.") # Added clarity

keyboard.on_press(on_key_press)

# cleanup function (Remove cv2.destroyAllWindows())
def cleanup():
    """Cleanup resources on exit"""
    # cv2.destroyAllWindows() # REMOVED - Handled by headless_main.py
    keyboard.unhook_all()
    print("Keyboard hooks removed.") # Added confirmation

# wait_for_mic function (Remains the same)
def wait_for_mic():
    """Wait for mic button press"""
    print("\nPress 'm' to start listening (Ctrl+C to exit)")
    # speak_text("Press 'm' to start") # Maybe comment out if too verbose
    state.mic_ready.clear()
    state.stop_requested.clear() # Ensure stop is cleared here too
    
    while not state.mic_ready.is_set() and not state.stop_requested.is_set(): # Check stop here too
         # Use wait with a timeout to allow checking stop_requested periodically
         state.mic_ready.wait(timeout=0.1) 
         if keyboard.is_pressed('ctrl') and keyboard.is_pressed('c'): # Manual Ctrl+C check
             print("Ctrl+C detected, stopping wait.")
             raise KeyboardInterrupt 

    if state.stop_requested.is_set():
         print("Stop requested while waiting for mic.")
         return False # Indicate that we shouldn't proceed

    state.mic_ready.clear() # Clear the event after it's been handled
    return True # Indicate that mic is ready


# reset_state function (Remains the same)
def reset_state():
    """Reset state variables for the next cycle"""
    state.in_detection = False
    state.stop_requested.clear() # Explicitly clear stop request here
    print("State reset for next command.") # Added confirmation

# detection_loop function (Remove cap.release() and cv2.destroyAllWindows())
def detection_loop(cap, yolo_model, midas, transform, class_id, target_label):
    """Main detection loop for Mode 1"""

    state.in_detection = True
    state.stop_requested.clear() # Ensure stop is cleared at start
    
    print(f"\nDetecting {target_label}. Press 'm' to stop this detection cycle.") # Updated prompt
    speak_text(f"Detecting {target_label}") # Speak only once at the start

    loop_start = time.time()
    detection_count = 0
    last_speak_time = 0 # Throttle speech
    speak_interval = 5 # Seconds between spoken updates
    
    while not state.stop_requested.is_set():
        # Check if camera is still open (important if shared)
        if not cap.isOpened():
             print("Camera closed unexpectedly.")
             speak_text("Camera connection lost.")
             break

        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame from camera.")
            # Don't break immediately, maybe it's temporary? Add a counter?
            time.sleep(0.5) # Wait a bit before retrying or breaking
            continue # Try reading next frame

        # Run detection
        # Make sure fuse_yolo_midas returns the frame with boxes drawn
        objects, _, annotated_frame, degrees, horizontal, vertical, depth_category = \
            fuse_yolo_midas(frame, yolo_model, midas, transform, class_id=class_id)
        
        # ---> Display the frame with annotations <---
        cv2.imshow('Object Detection Feed', annotated_frame) 

        # ---> Check for 'q' key press to stop THIS loop <---
        if cv2.waitKey(1) & 0xFF == ord('q'):
             print("Quit key (q) pressed. Stopping detection cycle.")
             state.stop_requested.set() # Use the global stop event
             break # Exit the while loop
        
        current_time_loop = time.time()
        # Report findings (throttled)
        if objects:
            detection_count += 1
            # Only speak periodically
            if current_time_loop - last_speak_time > speak_interval:
                speech = build_detection_speech(objects, degrees, horizontal, vertical, depth_category)
                print_results(objects, degrees, horizontal, vertical, depth_category) # Print every time
                speak_text(speech) # Speak only sometimes
                last_speak_time = current_time_loop
        
        # Timeout conditions (remains the same logic)
        elapsed = current_time_loop - loop_start
        if (detection_count > 0 and objects and elapsed > 20) or \
           (detection_count == 0 and elapsed > 60) or \
           (detection_count > 0 and not objects and elapsed > 10): # Shorter timeout if object lost
            status = "completed" if detection_count > 0 else "timed out, object not found"
            print(f"\nDetection {status}.")
            speak_text(f"Detection {status}.")
            state.stop_requested.set() # Ensure loop stops
            break
            
        # time.sleep(0.1) # Removed sleep, waitKey(1) provides a small delay

    # --- Cleanup for this loop ---
    print("Exiting detection loop.")
    state.in_detection = False 
    # cv2.destroyAllWindows() # REMOVED - Handled by headless_main.py
    # cap.release()           # REMOVED - Handled by headless_main.py
    # Important: Destroy the specific window we created if needed, 
    # but destroyAllWindows in main should handle it. Let's rely on main.


# build_detection_speech function (Remains the same)
def build_detection_speech(objects, degrees, horizontal, vertical, depth_category):
    """Build speech output from detection results"""
    count = len(objects)
    
    # Check if any objects were actually detected before formatting
    if not objects:
        return "No target objects found in this frame."

    # Use the details from the first detected object for simplicity, 
    # or modify to aggregate info if needed
    first_obj = objects[0] 
    label = first_obj.get('label', 'object') # Get label safely
    
    speech = f"Found {count} {label}{'s' if count > 1 else ''}. "
    
    # Use overall positioning if available, otherwise default
    pos_vertical = vertical if vertical else "unknown vertical position"
    pos_horizontal = horizontal if horizontal else "unknown horizontal position"
    pos_depth = depth_category if depth_category else "unknown distance"
    pos_degrees = degrees

    speech += f"Direction is roughly {pos_vertical}, "
    
    if pos_degrees is not None:
        speech += f"approximately {int(pos_degrees)} degrees to the {pos_horizontal}. "
    else:
        speech += f"towards the {pos_horizontal}. "
    
    speech += f"Estimated distance is {pos_depth}."
    return speech


# print_results function (Remains the same, added safety checks)
def print_results(objects, degrees, horizontal, vertical, depth_category):
    """Print detection details to console"""
    if not objects:
        print("\nNo target objects found in this frame.")
        return

    print(f"\nFound {len(objects)} object(s)")
    for i, obj in enumerate(objects, 1):
        print(f"  [{i}] {obj.get('label', 'N/A')}") # Use .get for safety
        print(f"      Position: {obj.get('position', 'N/A')}")
        print(f"      Distance: {obj.get('depth_category', 'N/A')}")
        obj_degrees = obj.get('degrees')
        if obj_degrees is not None:
            print(f"      Angle: {obj_degrees:.1f}°")
    
    print(f"\nOverall Position Estimate:")
    print(f"  Vertical: {vertical if vertical else 'N/A'}")
    print(f"  Horizontal: {horizontal if horizontal else 'N/A'}")
    if degrees is not None:
        print(f"  Angle from Center: {degrees:.1f}°")
    print(f"  Distance Category: {depth_category if depth_category else 'N/A'}")


# object_location function (Modified to accept cap and remove internal VideoCapture)
def object_location(cap, yolo_model, midas, transform, command):
    """Mode 1: Object detection and location"""
    
    # cap = cv2.VideoCapture(0) # REMOVED - cap is now passed in
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Keep buffer size setting if needed

    # Check if the passed cap object is valid and opened
    if not cap or not cap.isOpened():
        print("Error: Invalid or closed camera object passed to object_location.")
        speak_text("Camera error in detection mode.")
        return
        
    # Classify target object
    try:
        class_id_str = classify_request(command)
        if class_id_str is None or not class_id_str.isdigit():
             print(f"Error: Could not classify command '{command}' into a valid class ID.")
             speak_text("Sorry, I could not determine what object to look for.")
             return
        class_id = int(class_id_str)
        
        # Validate class_id against model names
        if class_id < 0 or class_id >= len(yolo_model.names):
             print(f"Error: Classified ID {class_id} is out of range for the model.")
             speak_text("Sorry, that object is not in my recognized list.")
             return
             
        target_label = yolo_model.names[class_id]

    except Exception as e:
        print(f"Error during classification: {e}")
        speak_text("There was an error identifying the object.")
        return
    
    print(f"Attempting to locate: {target_label} (Class ID: {class_id})")
    # speak_text(f"Okay, looking for {target_label}") # Moved speak to detection_loop start
    
    # Run detection loop
    detection_loop(cap, yolo_model, midas, transform, class_id, target_label)

    # --- Loop finished or was stopped ---
    # No cleanup needed here anymore (cap.release, destroyAllWindows)
    print(f"Finished object location task for {target_label}.")
