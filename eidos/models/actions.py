import json
import pyautogui

class ActionTokenizer:
    def __init__(self, screen_width=None, screen_height=None):
        if screen_width is None or screen_height is None:
            sw, sh = pyautogui.size()
            self.screen_width = sw
            self.screen_height = sh
        else:
            self.screen_width = screen_width
            self.screen_height = screen_height
        self.MOUSE_GRID_START = 0
        self.MOUSE_EVENT_START = 256
        self.KEYBOARD_START = 300
        self.APP_TOKEN_START = 1000
        self.app_vocab = {}

    def encode_mouse_pos(self, x, y):
        nx = max(0, min(1.0, x / self.screen_width))
        ny = max(0, min(1.0, y / self.screen_height))
        grid_x = int(nx * 15)
        grid_y = int(ny * 15)
        return self.MOUSE_GRID_START + (grid_y * 16) + grid_x

    def encode_keyboard(self, key_str):
        if len(key_str) == 1:
            return self.KEYBOARD_START + ord(key_str)
        special_keys = {
            "Key.enter": 1,
            "Key.space": 2,
            "Key.backspace": 3,
            "Key.shift": 4,
            "Key.tab": 5,
            "Key.cmd": 6,
            "Key.ctrl": 7,
            "Key.esc": 8
        }
        return self.KEYBOARD_START + 200 + special_keys.get(key_str, 0)

    def encode_mouse_event(self, event_type):
        event_map = {"click": 1, "scroll": 2}
        return self.MOUSE_EVENT_START + event_map.get(event_type, 0)

    def encode_app(self, app_name):
        if not app_name: return self.APP_TOKEN_START
        if app_name not in self.app_vocab:
            self.app_vocab[app_name] = self.APP_TOKEN_START + len(self.app_vocab)
        return self.app_vocab[app_name]

    def encode_observation(self, data):
        """Standardized 5-token action packet: [app, mouse_pos, click, key, special]"""
        tokens = [0] * 5
        
        # 1. App
        tokens[0] = self.encode_app(data.get("app_name"))
        
        # 2. Mouse Pos
        tokens[1] = self.encode_mouse_pos(data.get("mouse_x", 0), data.get("mouse_y", 0))
        
        # 3. Click (0: none, 1: left, 2: right)
        events = data.get("mouse_events", [])
        if isinstance(events, str):
            try: events = json.loads(events)
            except: events = []
        if events:
            ev_type = events[0].get("type", "")
            tokens[2] = self.MOUSE_EVENT_START + (1 if ev_type == "click" else 2)
            
        # 4. Keyboard Key (ASCII)
        keys = data.get("keystrokes", [])
        if isinstance(keys, str):
            try: keys = json.loads(keys)
            except: keys = []
        if keys:
            tokens[3] = self.encode_keyboard(keys[0])
            
        return tokens

    def decode_action(self, tokens):
        """Inverts the 5-token packet back into a dictionary for pyautogui."""
        # tokens: [app, mouse_pos, mouse_event, key, special]
        app_token, pos_token, event_token, key_token, _ = tokens
        
        # 1. Decode Mouse Position
        grid_y = pos_token // 16
        grid_x = pos_token % 16
        # Convert grid back to relative movement (or absolute)
        # We'll treat our grid as delta-ish or simple mapping
        # For now let's map it to 16x16 segments of the screen
        target_x = int((grid_x / 15) * self.screen_width)
        target_y = int((grid_y / 15) * self.screen_height)
        
        current_x, current_y = pyautogui.position()
        dx = target_x - current_x
        dy = target_y - current_y
        
        # 2. Decode Click
        click = "none"
        if event_token == self.MOUSE_EVENT_START + 1:
            click = "left"
        elif event_token == self.MOUSE_EVENT_START + 2:
            click = "right"
            
        # 3. Decode Key
        key = "none"
        if key_token >= self.KEYBOARD_START:
            val = key_token - self.KEYBOARD_START
            if val < 200: # Normal ASCII
                key = chr(val)
            else: # Special Keys
                # Reverse the map from encode_keyboard
                special_rev = {1: "enter", 2: "space", 3: "backspace", 4: "shift"} # Simplified
                key = special_rev.get(val - 200, "none")

        return {"dx": dx, "dy": dy, "click": click, "key": key}
