import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class AE(nn.Module):
    def __init__(self, dim_in=768, dim_latent=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(dim_in, 512), nn.ReLU(),
            nn.Linear(512, dim_latent)
        )
        self.dec = nn.Sequential(
            nn.Linear(dim_latent, 512), nn.ReLU(),
            nn.Linear(512, dim_in)
        )
    def forward(self,x):
        z = self.enc(x)
        return self.dec(z), z

if __name__ == "__main__":
    X = np.load("all_embeddings.npy").astype(np.float32)
    ds = TensorDataset(torch.from_numpy(X))
    dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=4)

    model = AE(dim_in=X.shape[1], dim_latent=128)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(30):
        for (batch,) in dl:
            recon, z = model(batch)
            loss = loss_fn(recon, batch)
            opt.zero_grad(); loss.backward(); opt.step()
        print("epoch", epoch, "loss", loss.item())

    torch.save(model.enc.state_dict(), "encoder.pt")
