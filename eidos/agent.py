import pyautogui
import torch
import torch.nn.functional as F
import pyautogui
from PIL import Image
from torchvision import transforms
from eidos.models.world import WorldModel
from eidos.models.actions import ActionTokenizer
from eidos.models.tokenizer import VisualTokenizer
from eidos.collector.screen import ScreenCollector
from eidos.models.text import TextTokenizer

class MarrowAgent:
    def __init__(self,world_checkpoint,vis_checkpoint):
        self.device=torch.device("mps" if torch.backends.mps.is_available()else "cpu")
        print(f"Agent initializing on {self.device}...")
        #Initialise tokenizer baby
        self.action_tokenizer=ActionTokenizer()
        self.text_tokenizer = TextTokenizer()
        self.vis_model=VisualTokenizer().to(self.device)
        self.vis_model.load_state_dict(torch.load(vis_checkpoint,map_location=self.device))
        self.vis_model.eval()
        
        # 2. Initialize World Model
        self.world_model = WorldModel().to(self.device)
        self.world_model.load_state_dict(torch.load(world_checkpoint, map_location=self.device))
        self.world_model.eval()
        self.transform=transforms.ToTensor()
        self.screen_collector=ScreenCollector("data/agent_frames")
        pyautogui.FAILSAFE=True
        
        # 3. Memory & Sampling Control
        from collections import deque
        self.vis_history = deque(maxlen=2) 
        self.act_history = deque(maxlen=2)
        self.temperature = 0.7
        self.top_k = 50

    def get_visual_state(self):
        screenshot_path=self.screen_collector.capture("live_view")
        # Save a copy as 'latest' for the user to debug
        import shutil
        shutil.copy(screenshot_path, "data/agent_frames/latest_observation.jpg")
        
        image=Image.open(screenshot_path).convert("RGB")
        img_tensor=self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _,codes,_=self.vis_model(img_tensor)
            vis_tokens=(codes+6000).reshape(-1).long().tolist()
        return vis_tokens
    
    def step(self, instruction=""):
        print("\n--- MARROW STEP ---")
        vis_tokens = self.get_visual_state()
        
        text_tokens = self.text_tokenizer.encode(instruction)
        
        # Build sequence with history: [Goal][Vis_old][Act_old][Goal][Vis_now]
        sequence = []
        if len(self.vis_history) > 0:
            for i in range(len(self.vis_history)):
                sequence.extend(text_tokens)
                sequence.append(10001)
                sequence.extend(self.vis_history[i])
                sequence.append(10000)
                if i < len(self.act_history):
                    sequence.extend(self.act_history[i])
        
        # Add current observation
        sequence.extend(text_tokens)
        sequence.append(10001)
        sequence.extend(vis_tokens)
        sequence.append(10000)
        
        # Update history for next step
        self.vis_history.append(vis_tokens)
            
        print(f"Thinking: '{instruction}' (Continuity: {len(self.act_history)} prev actions)...")
        current_action_tokens = []
        for _ in range(5):
            seq_tensor = torch.tensor([sequence + current_action_tokens], dtype=torch.long).to(self.device)
            if seq_tensor.size(1) > 1024:
                seq_tensor = seq_tensor[:, -1024:]
                
            with torch.no_grad():
                logits = self.world_model(seq_tensor)
                next_token_logits = logits[0, -1, :] / self.temperature
                
                v, _ = torch.topk(next_token_logits, self.top_k)
                next_token_logits[next_token_logits < v[..., -1, None]] = -float('Inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                current_action_tokens.append(next_token)
        
        self.act_history.append(current_action_tokens)
        
        action_obj = self.action_tokenizer.decode_action(current_action_tokens)
        print(f"Action: {action_obj}")
        print(f"Confidence: {probs.max().item():.2%}")
        
        self.execute_action(action_obj)
        import time
        time.sleep(1)
    def execute_action(self,action_obj):
        if action_obj["dx"]!=0 or action_obj["dy"]!=0:
            current_x,current_y=pyautogui.position()
            pyautogui.moveTo(current_x+action_obj["dx"],current_y+action_obj["dy"],duration=0.1)
            
        if action_obj["click"]=="left":
            pyautogui.click(button='left')
        elif action_obj["click"]=="right":
            pyautogui.click(button='right')
            
        key=action_obj.get("key","none")
        if key!="none":
            try:
                pyautogui.press(key)
            except Exception as e:
                print(f"Skipping unknown key:{key}")

if __name__ == "__main__":
    print("Starting MARROW Agent...")
    vis_check = "db/tokenizer_epoch_20.pt"
    world_check = "db/world_model_epoch_10.pt"
    try:
        agent = MarrowAgent(world_checkpoint=world_check, vis_checkpoint=vis_check)
        instruction = input("Give MARROW an instruction: ")
        while True:
            agent.step(instruction=instruction)
    except Exception as e:
        print(f"Agent failed to start: {e}")
