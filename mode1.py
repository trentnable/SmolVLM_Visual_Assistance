import time, threading, keyboard, cv2, math

from objectify import classify_request
from vision import fuse_yolo_midas
from speechrecog import get_voice_input
from googleTTS import speak_text, stop_speech

# Global state
class AppState:
    def __init__(self):
        self.mic_ready = threading.Event()
        self.stop_requested = threading.Event()
        self.cancel_mode = threading.Event()  
        self.in_detection = False
        self.last_press_time = 0
        self.lock = threading.Lock()
        self.mic_active = False

state = AppState()

def on_key_press(event):
    """Handle 'm' key press and 'c' key press (exit mode)"""
    current_time = time.time()
    with state.lock:
        if current_time - state.last_press_time < 0.3:
            return
        state.last_press_time = current_time

        if event.name == 'm':
            if state.in_detection:
                state.stop_requested.set()
                print("Stopping detection")

            state.mic_ready.set()

        elif event.name == 'c':
            if not state.mic_active:
                print("No mode is active")
                speak_text("No mode is active")
                return

            if state.in_detection:
                state.stop_requested.set()
            state.cancel_mode.set()
            print("Exiting Mode, returning to main menu")

keyboard.on_press(on_key_press)

def cleanup():
    """Cleanup resources on exit"""
    cv2.destroyAllWindows()
    keyboard.unhook_all()

def wait_for_mic():
    """Wait for mic button press"""
    print("\nPress 'm' to start (Ctrl+C to exit)")
    speak_text("Press 'm' to start")
    state.mic_ready.clear()
    state.mic_active = False  

    while not state.mic_ready.is_set():
        state.mic_ready.wait(timeout=0.5)
    
    state.mic_ready.clear()
    state.mic_active = True  


def wait_for_mic_in_mode():
    """Wait for mic button press while in Mode 1"""
    
    if state.mic_ready.is_set():
        state.mic_ready.clear()
        return True
    
    print("\nPress 'm' for mic, 'c' to exit location mode")
    speak_text("Press 'm' for mic, or 'c' to exit location mode")
    
    # Wait for mic button or cancel
    while not state.mic_ready.is_set() and not state.cancel_mode.is_set():
        state.mic_ready.wait(timeout=0.5)
    
    state.mic_ready.clear()
    
    
    return not state.cancel_mode.is_set()

def reset_state():
    """Reset state variables between detections"""
    state.in_detection = False
    state.stop_requested.clear()

def reset_mode_state():
    """Reset all state when exiting a mode"""
    state.in_detection = False
    state.stop_requested.clear()
    state.cancel_mode.clear()
    state.mic_ready.clear()

def detection_loop(cap, yolo_model, midas, transform, class_id, class_name_string):
    """Main detection loop for Mode 1 - tracks each class individually"""

    state.in_detection = True
    state.stop_requested.clear()

    print(f"\nDetecting {class_name_string}. Press 'm' for mic, 'c' to exit mode.")

    tts_thread = None
    tracked_objects = {}
    last_detection_time = time.time()
    detection_timeout = 1.5 

    while not state.stop_requested.is_set() and not state.cancel_mode.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Camera error")
            break

        # Run detection
        objects, _, annotated_frame, _, _, _, _, _ = fuse_yolo_midas(
            frame, yolo_model, midas, transform, class_name_string, class_id=class_id
        )

        cv2.imshow('Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        moving_objects = []

        # --- CASE 1: Objects detected ---
        if objects:
            # Reset timer since objects are visible
            last_detection_time = time.time()

            for obj in objects:
                class_name = obj["label"]
                bbox = obj["bbox"]

                # First detection after reset or new object
                if class_name not in tracked_objects:
                    depth, position = initial_change_states(bbox)
                    tracked_objects[class_name] = {
                        "depth": depth,
                        "position": position,
                        "time_of_last_detection": time.time()
                    }
                    moving_objects.append(obj)
                    print(f"First detection of {class_name}")

                else:
                    # Compare position/depth change with previous state
                    prev_depth = tracked_objects[class_name]["depth"]
                    prev_position = tracked_objects[class_name]["position"]

                    significant_change, delta_x, delta_y, delta_depth, depth_initial = change_detection(
                        bbox, prev_depth, prev_position
                    )

                    if significant_change:
                        print(f"{class_name} has moved")
                        moving_objects.append(obj)

                        depth, position = initial_change_states(bbox)
                        tracked_objects[class_name].update({
                            "depth": depth,
                            "position": position,
                            "time_of_last_detection": time.time()
                        })

            # --- Speak when movement is detected ---
            if moving_objects:
                speech = build_detection_speech(moving_objects)

                if tts_thread is not None and tts_thread.is_alive():
                    stop_speech()
                    tts_thread.join(timeout=0.3)

                tts_thread = threading.Thread(target=speak_text, args=[speech], daemon=True)
                tts_thread.start()

        # --- CASE 2: No objects detected ---
        else:
            time_since_last = time.time() - last_detection_time

            # If more than 0.5 s without any detection, stop TTS and reset trackers
            if time_since_last > detection_timeout:
                if tts_thread is not None and tts_thread.is_alive():
                    print(f"No objects detected for {time_since_last:.2f}s → stopping speech")
                    stop_speech()

                # Reset everything so reappearing objects are treated as new
                tracked_objects.clear()
                last_detection_time = time.time()

        time.sleep(0.1)

    state.in_detection = False
    cv2.destroyAllWindows()
    cap.release()

def change_detection(bbox, depth_prev, position_prev):
    """Detects changes in x, y, and depth position between object location events"""
    x1, y1, x2, y2 = bbox

    # Current values
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    depth_initial = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Differences values
    delta_x = abs(mx - position_prev[0])
    delta_y = abs(my - position_prev[1])
    delta_depth = abs(depth_initial - depth_prev)

    # Thresholds
    position_threshold = 150 
    depth_threshold = 0.4 * depth_prev if depth_prev > 0 else 0  

    significant_change = False
    if delta_x > position_threshold and mx > 0:
        print("Change in X")
        significant_change = True
    if delta_y > position_threshold and my > 0:
        print("Change in Y")
        significant_change = True
    if delta_depth > depth_threshold and depth_initial > 0:
        print("Change in depth")
        significant_change = True

    return significant_change, delta_x, delta_y, delta_depth, depth_initial

def initial_change_states(bbox):
    """Compute initial depth and position"""
    x1, y1, x2, y2 = bbox
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    depth_initial = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    position_initial = (mx, my)
    return depth_initial, position_initial


def build_detection_speech(objects):
    parts = []
    for obj in objects:
        label = obj["label"]
        position = obj["position"]
        depth_cat = obj["depth_category"]
        parts.append(f"{label} is {position} and {depth_cat}")
    return ". ".join(parts)

def print_results(objects, degrees, horizontal, vertical, depth_category):
    """Print detection details to console"""
    print(f"\nFound {len(objects)} object(s)")
    for i, obj in enumerate(objects, 1):
        class_name_int = ", ".join(obj["label"]) if isinstance(obj["label"], (list, tuple)) else str(obj["label"])
        print(f"  [{i}] {class_name_int}")
        print(f"      Position: {obj['position']}")
        print(f"      Distance: {obj['depth_category']}")
        if obj['degrees'] is not None:
            print(f"      Angle: {obj['degrees']:.1f}°")
    
    print(f"\nOverall: {vertical}, {horizontal}")
    if degrees is not None:
        print(f"         {degrees:.1f}° from center")
    print(f"         {depth_category}")


def object_location(yolo_model, midas, transform, initial_command):
    """Mode 1: Object detection and location, endless loop until 'c' is pressed"""
    
    command = initial_command
    first_iteration = True
    
    # Mode1 looped
    while not state.cancel_mode.is_set():
        
        if not first_iteration:
            continue_mode = wait_for_mic_in_mode()
            if not continue_mode:
                break
            
            # Get voice command
            command = get_voice_input(duration=5)
        
        first_iteration = False
        
        # Classify object
        class_id = classify_request(command)

        if class_id == "unlisted_object":
            print("Requested object is not supported")
            speak_text("Requested object is not supported")
            continue

        class_name_string = [yolo_model.names[int(i)] for i in class_id]
        
        print(f"Locating {class_name_string}")
        speak_text(f"Locating {class_name_string}")
        
        # Open camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("Camera error")
            speak_text("Camera error")
            continue
        
        # Run detection
        detection_loop(cap, yolo_model, midas, transform, class_id, class_name_string)
        
        # Reset detection
        reset_state()
        
        # Clear detection
        if state.cancel_mode.is_set():
            break
    
    # Clean up for mode exit
    reset_mode_state()
    print("Exited Mode 1")
    speak_text("Exited Mode 1")
