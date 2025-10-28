import cv2
import pygame
import keyboard
import torch
import gc
import sys
import threading

class ResourceManager:
    def __init__(self):
        self.camera_captures = []
        self.tts_threads = []
        self.pygame_initialized = False
        self.keyboard_hooked = False
        self.models_loaded = {}
        self.lock = threading.Lock()
    
    def register_camera(self, cap):
        with self.lock:
            if cap not in self.camera_captures:
                self.camera_captures.append(cap)

    def register_tts_thread(self, thread):
        with self.lock: # this releases the lock as soon as the with block is finished
            self.tts_threads.append(thread)

    def register_pygame(self):
        with self.lock:
            self.pygame_initialized = True

    def register_keyboard_hook(self):
        with self.lock:
            self.keyboard_hooked = True
    
    def register_model(self, name, model):
        with self.lock:
            self.models_loaded[name] = model

    def cleanup_tts(self):
        try:
            if self.pygame_initialized:
                from googleTTS import stop_speech
                stop_speech()

                with self.lock:
                    threads = self.tts_threads.copy()

                for thread in threads:
                    if thread and thread.is_alive():
                        thread.join(timeout=1.0)

                self.tts_threads.clear()

                print("TTS thread stopped.")

        except Exception as e:
            print(f"TTS cleanup WARNING: {e}")

    def cleanup_cameras(self):
        try:
            with self.lock:
                cameras = self.camera_captures.copy()

            for cap in cameras:
                try:
                    if cap is not None and cap.isOpened():
                        cap.release()
                except Exception as e:
                    print(f"Camera release WARNING: {e}")

            self.camera_captures.clear()

            print("Camera captures released")

        except Exception as e:
            print(f"Camera cleanup WARNING: {e}")

    def cleanup_opencv_windows(self):
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            print("OpenCV windows destroyed")

        except Exception as e:
            print(f"OpenCV cleanup WARNING: {e}")

    def cleanup_pygame(self):
        try:
            if self.pygame_initialized:
                pygame.mixer.quit()
                self.pygame_initialized = False
                print("Pygame mixer cleaned up")

        except Exception as e:
            print(f"Pygame cleanup WARNING: {e}")

    def cleanup_keyboard(self):
        try:
            if self.keyboard_hooked:
                keyboard.unhook_all()
                self.keyboard_hooked = False
                print("Keyboard hooks removed")

        except Exception as e:
            print(f"Keyboard cleanup WARNING: {e}")

    def cleanup_cuda(self):
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("CUDA cache cleared.")

        except Exception as e:
            print(f"CUDA cleanup WARNING: {e}")

    def cleanup_models(self):
        try:
            with self.lock:
                self.models_loaded.clear()
            print("Model references cleared")

        except Exception as e:
            print(f"Model cleanup WARNING: {e}")

    def cleanup_all(self):
        print("Cleaning up resources...")

        self.cleanup_tts()
        self.cleanup_cameras()
        self.cleanup_opencv_windows()
        self.cleanup_pygame()
        self.cleanup_keyboard()
        self.cleanup_cuda()
        self.cleanup_models()
        
        gc.collect()

        print("Cleanup complete.")

resource_manager = ResourceManager()

def cleanup():
    resource_manager.cleanup_all()

def register_cleanup_handlers(exit_key='esc'):
    import atexit
    import signal
    import keyboard
    import os

    atexit.register(cleanup) # register for normal exit

    def signal_handler(sig):
        print(f"\nReceived signal {sig}")
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler) # if the user presses Ctrl + C

    def emergency_exit():
        print(f"\n{exit_key.upper()} pressed exiting the program...")
        cleanup()
        os._exit(0)
    
    keyboard.add_hotkey(exit_key, emergency_exit)

    print(f"Cleanup handlers registered (Press {exit_key.upper()} to exit)")

def register_camera(cap):
    resource_manager.register_camera(cap)

def register_tts_thread(thread):
    resource_manager.register_tts_thread(thread)

def register_pygame():
    resource_manager.register_pygame()

def register_keyboard_hook():
    resource_manager.register_keyboard_hook()

def register_model(name, model):
    resource_manager.register_model(name, model)