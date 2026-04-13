import mss
from PIL import Image
from pathlib import Path
import logging

class ScreenCollector:
    def __init__(self,output_dir:str):
        self.output_dir=Path(output_dir)
        self.output_dir.mkdir(parents=True,exist_ok=True)
        self.sct=mss.mss()
        self.monitor=self.sct.monitors[1]
    
    def capture(self,filename:str)->str:
        try:
            sct_img=self.sct.grab(self.monitor)
            img=Image.frombytes("RGB",sct_img.size,sct_img.bgra,"raw","BGRX")
            img=img.resize((256,160),Image.Resampling.LANCZOS)
            path=self.output_dir/f"{filename}.jpg"
            img.save(path,"JPEG",quality=85)
            return str(path)
        
        except Exception as e:
            logging.error(f"Screen capture failed:{e}")
            return ""

