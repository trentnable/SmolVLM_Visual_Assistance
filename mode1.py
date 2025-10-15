import time
import threading
import keyboard
import cv2
import math

from objectify import classify_request, mode_select
from vision import fuse_yolo_midas, setup_yolo, setup_midas
from speechrecog import listen_for_command, get_voice_input
from googleTTS import speak_text

# Global state
class AppState:
    def __init__(self):
        self.mic_ready = threading.Event()
        self.stop_requested = threading.Event()
        self.in_detection = False
        self.last_press_time = 0
        self.lock = threading.Lock()

state = AppState()

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
            print("Stop detection")
        else:
            state.mic_ready.set()

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
    
    while not state.mic_ready.is_set():
        state.mic_ready.wait(timeout=0.5)
    
    state.mic_ready.clear()

def reset_state():
    """Pass variable values between scripts"""
    state.in_detection = False
    

def detection_loop(cap, yolo_model, midas, transform, class_id, target_label):
    """Main detection loop for Mode 1"""

    state.in_detection = True
    state.stop_requested.clear()
    
    print(f"\nDetecting {target_label}. Press 'm' to stop.")
    
    loop_start = time.time()
    detection_count = 0
    significant_change = False
    depth_current = 0
    position_current = 0, 0
    
    while not state.stop_requested.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Camera error")
            break
        
        # Run detection
        objects, _, annotated_frame, degrees, horizontal, vertical, depth_category, bbox = \
            fuse_yolo_midas(frame, yolo_model, midas, transform, class_id=class_id)
        
        cv2.imshow('Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        significant_change, delta_x, delta_y, delta_depth, depth_current, position_current = change_detection(bbox, depth_current, position_current)

        # Report findings
        if (objects and significant_change) or (objects and detection_count == 0):
            detection_count += 1
            speech = build_detection_speech(objects, degrees, horizontal, vertical, depth_category)
            print_results(objects, degrees, horizontal, vertical, depth_category)
            speak_text(speech)
            depth_initial, position_initial = initial_change_states(bbox)

        time.sleep(1)
    
    state.in_detection = False
    cv2.destroyAllWindows()
    cap.release()

def change_detection(bbox, depth_initial, position_initial):
    """Detects changes in x, y, and depth position between object location events"""
    x1, y1, x2, y2 = bbox

    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    depth_current = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    delta_x = abs(mx - position_initial[0])
    delta_y = abs(my - position_initial[1])
    delta_depth = abs(depth_current - depth_initial)

    # Significant change thresholds
    position_threshold = 0.1 * depth_current  
    depth_threshold = 0.25 * depth_initial 

    if delta_x > position_threshold:
        significant_change = True
        print("Change in X")
    elif delta_y > position_threshold:
        significant_change = True
        print("Change in Y")
    elif delta_depth > depth_threshold:
        significant_change = True
        print("Change in depth")
    else:
        significant_change = False

    position_current = (mx, my)

    return significant_change, delta_x, delta_y, delta_depth, depth_current, position_current


def initial_change_states(bbox):
    """Update initial conditions of for remote changing"""
    x1, y1, x2, y2 = bbox
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    depth_initial = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    position_initial = (mx, my)

    return depth_initial, position_initial


def build_detection_speech(objects, degrees, horizontal, vertical, depth_category):
    """Build speech output from detection results"""
    count = len(objects)
    speech = f"Found {count} object{'s' if count > 1 else ''}. "
    speech += f"Direction is {vertical}, "
    
    if degrees is not None:
        speech += f"{int(degrees)} degrees to the {horizontal}. "
    else:
        speech += f"{horizontal}. "
    
    speech += f"Distance is {depth_category}."
    return speech


def print_results(objects, degrees, horizontal, vertical, depth_category):
    """Print detection details to console"""
    print(f"\nFound {len(objects)} object(s)")
    for i, obj in enumerate(objects, 1):
        print(f"  [{i}] {obj['label']}")
        print(f"      Position: {obj['position']}")
        print(f"      Distance: {obj['depth_category']}")
        if obj['degrees'] is not None:
            print(f"      Angle: {obj['degrees']:.1f}°")
    
    print(f"\nOverall: {vertical}, {horizontal}")
    if degrees is not None:
        print(f"         {degrees:.1f}° from center")
    print(f"         {depth_category}")


def object_location(yolo_model, midas, transform, command):
    """Mode 1: Object detection and location"""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("Camera error")
        speak_text("Camera error")
        return
    
    # Classify target object
    class_id = int(classify_request(command))
    target_label = yolo_model.names[class_id]
    
    print(f"Locating {target_label}")
    speak_text(f"Locating {target_label}")
    
    # Run detection
    detection_loop(cap, yolo_model, midas, transform, class_id, target_label)