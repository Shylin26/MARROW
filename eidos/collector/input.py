from pynput import keyboard, mouse
import threading
import logging

class InputCollector:
    def __init__(self):
        self.keystrokes = []
        self.mouse_events = []
        self.mouse_x = 0
        self.mouse_y = 0
        self.lock = threading.Lock()
        
        # We tell the listener which function to call for which action
        self.kb_listener = keyboard.Listener(on_press=self._on_key_press)
        self.ms_listener = mouse.Listener(
            on_move=self._on_mouse_move,
            on_click=self._on_mouse_click, # Separate click function
            on_scroll=self._on_mouse_scroll
        )

    def start(self):
        self.kb_listener.start()
        self.ms_listener.start()

    def stop(self):
        self.kb_listener.stop()
        self.ms_listener.stop()

    def _on_key_press(self, key):
        with self.lock:
            try:
                # Store character (like 'a') or special key name (like 'Key.enter')
                k = key.char if hasattr(key, 'char') else str(key)
                self.keystrokes.append(k)
            except Exception as e:
                logging.error(f"Input key error: {e}")

    def _on_mouse_move(self, x, y):
        """Important: Only takes x and y!"""
        with self.lock:
            self.mouse_x = int(x)
            self.mouse_y = int(y)

    def _on_mouse_click(self, x, y, button, pressed):
        """This is where we record the buttons actually being pressed."""
        if pressed:
            with self.lock:
                self.mouse_events.append({
                    "type": "click",
                    "x": int(x),
                    "y": int(y),
                    "button": str(button)
                })

    def _on_mouse_scroll(self, x, y, dx, dy):
        with self.lock:
            self.mouse_events.append({
                "type": "scroll",
                "x": int(x),
                "y": int(y),
                "dy": int(dy),
                "dx": int(dx)
            })

    def get_and_flush(self):
        with self.lock:
            # Note the correct spelling of "keystrokes"
            data = {
                "keystrokes": list(self.keystrokes),
                "mouse_x": self.mouse_x,
                "mouse_y": self.mouse_y,
                "mouse_events": list(self.mouse_events)
            }
            self.keystrokes.clear()
            self.mouse_events.clear()
            return data
