import sqlite3
import sqlite3
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os

class FrameDataset(Dataset):
    def __init__(self,db_path):
        self.db_path=db_path
        self.transform=transforms.ToTensor()
        self.frame_paths=self._load_paths()
    def _load_paths(self):
        conn=sqlite3.connect(self.db_path)
        cursor=conn.cursor()
        cursor.execute("SELECT frame_path FROM observations")
        paths=[row[0]for row in cursor.fetchall() if row[0] and os.path.exists(row[0])]
        conn.close()
        return paths
    def __len__(self):
        return len(self.frame_paths)
    
    def __getitem__(self,idx):
        path=self.frame_paths[idx]
        image=Image.open(path).convert("RGB")
        return self.transform(image)