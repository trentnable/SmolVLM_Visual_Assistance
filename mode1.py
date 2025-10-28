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
            # Exit mode completely
            if state.in_detection:
                state.stop_requested.set()
            state.cancel_mode.set()
            print("Exiting Mode 1, returning to main menu")

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

def wait_for_mic_in_mode():
    """Wait for mic button press while in Mode 1"""
    # If mic is already ready (from stopping previous detection), proceed immediately
    if state.mic_ready.is_set():
        state.mic_ready.clear()
        return True
    
    print("\nPress 'm' for new command, 'c' to exit mode")
    speak_text("Press 'm' for new command, or 'c' to exit mode")
    
    # Wait for mic button or cancel
    while not state.mic_ready.is_set() and not state.cancel_mode.is_set():
        state.mic_ready.wait(timeout=0.5)
    
    state.mic_ready.clear()
    
    # Return True if continuing in mode, False if exiting
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
    """Detection loop with flicker tolerance and per-instance tracking."""

    state.in_detection = True
    state.stop_requested.clear()

    print(f"\nDetecting {class_name_string}. Press 'm' for mic, 'c' to exit mode.")

    tts_thread = None
    tracked_objects = {}  # ID dictionary
    last_detection_time = time.time()
    detection_timeout = 0.5  # undetected timer

    max_missing_time = 0.5          # missing object buffer
    min_visible_frames = 6          # object frames before adding to IDs
    movement_confirm_frames = 2     # movement frames
    match_distance_threshold = 60   # tracked object matching
    movement_threshold_px = 30      # significant movement
    next_object_id = 0

    while not state.stop_requested.is_set() and not state.cancel_mode.is_set():
        loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Camera error")
            break

        objects, _, annotated_frame, _, _, _, _, _ = fuse_yolo_midas(
            frame, yolo_model, midas, transform, class_name_string, class_id=class_id
        )

        cv2.imshow('Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        moving_objects = []

        if objects:
            last_detection_time = time.time()

            current_matched_ids = set()

            detections = []
            for obj in objects:
                x1, y1, x2, y2 = obj["bbox"]
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                detections.append({"obj": obj, "cx": cx, "cy": cy, "bbox": obj["bbox"]})

            for det in detections:
                best_id = None
                best_dist = float("inf")
                for obj_id, data in tracked_objects.items():
                    if data["label"] != det["obj"]["label"]:
                        continue

                    # center distance
                    prev_bbox = data["bbox"]
                    pcx = (prev_bbox[0] + prev_bbox[2]) / 2.0
                    pcy = (prev_bbox[1] + prev_bbox[3]) / 2.0
                    dist = math.hypot(det["cx"] - pcx, det["cy"] - pcy)

                    if dist < best_dist:
                        best_dist = dist
                        best_id = obj_id

                # Instance matching
                if best_id is not None and best_dist <= match_distance_threshold:
                    data = tracked_objects[best_id]

                    # movement magnitude
                    prev_cx, prev_cy = data["position"]
                    move_mag = math.hypot(det["cx"] - prev_cx, det["cy"] - prev_cy)

                    # update timestamps
                    data["bbox"] = det["bbox"]
                    data["position"] = (det["cx"], det["cy"])
                    data["depth"], _ = initial_change_states(det["bbox"])  # update depth
                    data["last_seen"] = time.time()
                    data["missing_since"] = None
                    data["visible_frames"] += 1

                    # movement detection
                    if move_mag > movement_threshold_px:
                        data["consecutive_move_frames"] += 1
                    else:
                        data["consecutive_move_frames"] = 0

                    # announce after min_visible_frames
                    if not data["reported"] and data["visible_frames"] >= min_visible_frames:
                        # new object
                        moving_objects.append(det["obj"])
                        data["reported"] = True
                        data["last_reported_position"] = data["position"]

                    elif data["consecutive_move_frames"] >= movement_confirm_frames:
                        # check movement since last
                        last_rep_pos = data.get("last_reported_position", (prev_cx, prev_cy))
                        moved_since_report = math.hypot(data["position"][0] - last_rep_pos[0],
                                                        data["position"][1] - last_rep_pos[1])
                        if moved_since_report > movement_threshold_px:
                            moving_objects.append(det["obj"])
                            data["last_reported_position"] = data["position"]
                            data["consecutive_move_frames"] = 0

                    current_matched_ids.add(best_id)

                else:
                    tracked_objects[next_object_id] = {
                        "label": det["obj"]["label"],
                        "bbox": det["bbox"],
                        "position": (det["cx"], det["cy"]),
                        "depth": initial_change_states(det["bbox"])[0],
                        "first_seen": time.time(),
                        "last_seen": time.time(),
                        "missing_since": None,
                        "visible_frames": 1,          
                        "consecutive_move_frames": 0,
                        "reported": False,
                        "last_reported_position": None
                    }
                    # avoid immediate announcing
                    current_matched_ids.add(next_object_id)
                    print(f"Created tracker ID {next_object_id} for {det['obj']['label']}")
                    next_object_id += 1

            # missing objects identified
            now = time.time()
            for obj_id, data in list(tracked_objects.items()):
                if obj_id not in current_matched_ids:
                    if data["missing_since"] is None:
                        data["missing_since"] = now  

            # remove missing objects
            to_delete = []
            for obj_id, data in list(tracked_objects.items()):
                if data["missing_since"] is not None:
                    if (time.time() - data["missing_since"]) > max_missing_time:
                        to_delete.append(obj_id)

            for obj_id in to_delete:
                del tracked_objects[obj_id]

            # announce moving objects
            if moving_objects:
                speech = build_detection_speech(moving_objects)
                if tts_thread is not None and tts_thread.is_alive():
                    stop_speech()
                    tts_thread.join(timeout=0.3)
                tts_thread = threading.Thread(target=speak_text, args=[speech], daemon=True)
                tts_thread.start()

        else:
            time_since_last = time.time() - last_detection_time
            if time_since_last > detection_timeout:
                if tts_thread is not None and tts_thread.is_alive():
                    print(f"No objects detected for {time_since_last:.2f}s")
                    stop_speech()
                tracked_objects.clear()
                last_detection_time = time.time()

        elapsed = time.time() - loop_start
        sleep_time = max(0.01, 0.1 - elapsed)
        time.sleep(sleep_time)

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
