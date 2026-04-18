import torch
import sqlite3
import json
import random
from torch.utils.data import Dataset
from eidos.models.tokenizer import VisualTokenizer
from eidos.models.actions import ActionTokenizer
from eidos.models.text import TextTokenizer
from PIL import Image
from torchvision import transforms

class MarrowSequenceDataset(Dataset):
    def __init__(self, db_path, checkpoint_path):
        self.db_path = db_path
        self.action_tokenizer = ActionTokenizer()
        self.text_tokenizer = TextTokenizer()
        self.vis_model = VisualTokenizer()
        self.vis_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.vis_model.eval()
        self.transform = transforms.ToTensor()
        self.sessions = self._load_sessions()
    
    def _load_sessions(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM observations ORDER BY session_id, timestamp")
        all_rows = cursor.fetchall()
        sessions = {}
        for row in all_rows:
            sid = row["session_id"]
            if sid not in sessions:
                sessions[sid] = []
            sessions[sid].append(row)
        valid_sessions = [s for s in sessions.values() if len(s) >= 2]
        conn.close()
        print(f"Loaded {len(valid_sessions)} valid recording sessions.")
        return valid_sessions
    
    def __len__(self):
        return len(self.sessions)
    
    def __getitem__(self, idx):
        session = self.sessions[idx]
        start_idx = torch.randint(0, len(session) - 2 + 1, (1,)).item()
        window = session[start_idx : start_idx + 2]
        full_sequence = []
        for row in window:
            app = row['app_name'] if row['app_name'] else "Desktop"
            if row['mouse_events'] and "click" in row['mouse_events']:
                templates = [f"Click on {app}", f"Interact with {app}", f"Select element in {app}"]
                synthetic_goal = random.choice(templates)
            else:
                synthetic_goal = f"Use {app}"
            text_tokens = self.text_tokenizer.encode(synthetic_goal)

            image = Image.open(row["frame_path"]).convert("RGB")
            img_tensor = self.transform(image).unsqueeze(0)

            with torch.no_grad():
                x_recon, codes, _ = self.vis_model(img_tensor)
                vis_tokens = (codes + 6000).reshape(-1).long().tolist()

            act_tokens = self.action_tokenizer.encode_observation(dict(row))

            full_sequence.extend(text_tokens)
            full_sequence.append(10001)
            full_sequence.extend(vis_tokens)
            full_sequence.append(10000)
            full_sequence.extend(act_tokens)
        return torch.tensor(full_sequence)
