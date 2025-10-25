import time, threading, keyboard, cv2, math

from objectify import classify_request
from vision import fuse_yolo_midas
from speechrecog import get_voice_input
from googleTTS import speak_text, stop_speech
from resource_manager import register_camera, register_tts_thread, register_keyboard_hook

# Global state
class AppState:
    def __init__(self):
        self.mic_ready = threading.Event()
        self.stop_requested = threading.Event()
        self.cancel_mode = threading.Event()  
        self.in_detection = False
        self.last_press_time = 0
        self.lock = threading.Lock()

state = AppState()

def on_key_press(event):
    """Handle 'm' key press (stop detection & trigger mic) and 'c' key press (exit mode)"""
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
            if state.in_detection:
                state.stop_requested.set()
            state.cancel_mode.set()
            print("Exiting Mode 1, returning to main menu")

keyboard.on_press(on_key_press)
register_keyboard_hook()

def wait_for_mic():
    """Wait for mic button press"""
    print("\nPress 'm' to start (Ctrl+C to exit)")
    speak_text("Press 'm' to start")
    state.mic_ready.clear()
    
    while not state.mic_ready.is_set():
        state.mic_ready.wait(timeout=0.5)
    
    state.mic_ready.clear()

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
    
    loop_start = time.time()
    detection_count = 0
    tts_thread = None
    
    # Track state for each class_id
    tracked_objects = {}  
    
    while not state.stop_requested.is_set() and not state.cancel_mode.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Camera error")
            break

        # to skip n frames, do range(n-1); currently skipping 4 frames
        for _ in range(3):
            cap.grab()
        
        # Run detection
        objects, _, annotated_frame, _, _, _, _, _ = fuse_yolo_midas(
            frame, yolo_model, midas, transform, class_name_string, class_id=class_id
        )

        cv2.imshow('Detection', annotated_frame)

        # Check for changes in each tracked class_id
        moving_objects = []
        
        for obj in objects:
            class_name = obj["label"]
            bbox = obj["bbox"]
            
            # First detection of each class_id
            if class_name not in tracked_objects:
                depth, position = initial_change_states(bbox)
                tracked_objects[class_name] = {
                    "depth": depth,
                    "position": position
                }
                moving_objects.append(obj)
                print(f"First detection of {class_name}")
                
            else:
                # Check if each class_id has moved
                prev_depth = tracked_objects[class_name]["depth"]
                prev_position = tracked_objects[class_name]["position"]
                
                significant_change, delta_x, delta_y, delta_depth, depth_initial = change_detection(
                    bbox, prev_depth, prev_position
                )
                
                if significant_change:
                    print(f"{class_name} has moved")
                    moving_objects.append(obj)
                    
                    # Update tracked state for class_id
                    depth, position = initial_change_states(bbox)
                    tracked_objects[class_name]["depth"] = depth
                    tracked_objects[class_name]["position"] = position

        # Announce only objects that moved
        if moving_objects:
            detection_count += 1
            speech = build_detection_speech(moving_objects)

            
            for obj in moving_objects:
                print_results(
                    [obj],
                    obj["degrees"],
                    obj["horizontal"],
                    obj["vertical"],
                    obj["depth_category"]
                )

            # TTS Override
            if tts_thread is not None and tts_thread.is_alive():
                stop_speech()
                tts_thread.join(timeout=0.5)

            tts_thread = threading.Thread(target=speak_text, args=[speech], daemon=True)
            register_tts_thread(tts_thread)     # register for cleanup
            tts_thread.start()

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
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Camera error")
        speak_text("Camera error")
        return

    register_camera(cap)    # register for cleanup
    # only caputre store one frame in the buffer at a time to avoid old frames
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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
