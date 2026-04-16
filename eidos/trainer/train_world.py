import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from eidos.trainer.sequence_dataset import MarrowSequenceDataset
from eidos.models.world import WorldModel
def train_world_model():
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device:{device}")
    db_path="db/marrow.db"
    tokenizer_checkpoint="db/tokenizer_epoch_20.pt" 
    dataset=MarrowSequenceDataset(db_path, tokenizer_checkpoint)
    dataloader=DataLoader(dataset,batch_size=1,shuffle=True)
    model=WorldModel().to(device)
    optimizer=optim.AdamW(model.parameters(),lr=1e-4)
    criterion=nn.CrossEntropyLoss()
    epochs=10
    for epoch in range(epochs):
        model.train()
        total_loss=0
        progress=tqdm(dataloader,desc=f"World Model Epoch {epoch+1}/{epochs}")
        for sequence in progress:
            sequence=sequence.to(device)
            x=sequence[:,:-1]
            y=sequence[:,1:]
            optimizer.zero_grad()
            logits=model(x)
            B,T,C=logits.shape
            logits_flat=logits.view(B*T,C)
            y_flat=y.reshape(-1)
            loss=criterion(logits_flat,y_flat)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
            progress.set_postfix({"loss": loss.item()})
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(dataloader):.4f}")
        torch.save(model.state_dict(), f"db/world_model_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train_world_model()


