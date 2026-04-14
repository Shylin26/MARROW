import torch
from torch.utils.data import DataLoader, random_split
from eidos.models.loss import TokenizerLoss
from eidos.models.tokenizer import VisualTokenizer
from eidos.trainer.dataset import FrameDataset
from tqdm import tqdm

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = FrameDataset(db_path="./db/marrow.db")
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    model = VisualTokenizer().to(device)
    criterion = TokenizerLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            x = batch.to(device)
            x_recon, codes, indices = model(x)
            
            loss_dict = criterion(x_recon, x)
            loss = loss_dict["total_loss"]
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        torch.save(model.state_dict(), f"db/tokenizer_epoch_{epoch+1}.pt")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device)
                x_recon, _, _ = model(x)
                loss_dict = criterion(x_recon, x)
                val_loss += loss_dict["total_loss"].item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

if __name__ == "__main__":
    train()
