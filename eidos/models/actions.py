import json

class ActionTokenizer:
    def __init__(self, screen_width=1440, screen_height=900):
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
        tokens = []
        tokens.append(self.encode_app(data.get("app_name")))
        tokens.append(self.encode_mouse_pos(data.get("mouse_x"), data.get("mouse_y")))
        keys = data.get("keystrokes", [])
        if isinstance(keys, str):
            try: keys = json.loads(keys)
            except: keys = []
        for k in keys:
            tokens.append(self.encode_keyboard(k))
        events = data.get("mouse_events", [])
        if isinstance(events, str):
            try: events = json.loads(events)
            except: events = []
        for e in events:
            tokens.append(self.encode_mouse_event(e.get("type")))
        return tokens
