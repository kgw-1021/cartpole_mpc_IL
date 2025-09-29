import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImitationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  
        )
    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":

    df = pd.read_csv('cartpole_mpc_data.csv')

    X = df[['x', 'x_dot', 'theta', 'theta_dot']].values
    y = df['raw_action'].values

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)  # classification

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    #########################################################
    model = ImitationNet().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best = 1

    num_epochs = 1000
    for epoch in range(num_epochs):
        total_loss = 0
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        total_loss /= len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

        if total_loss < best:
            best = total_loss
            best_model = model.state_dict()

        if total_loss < 5e-3:
            print("Early stop")
            break

    torch.save(best_model, 'cartpole_imitation.pth')
