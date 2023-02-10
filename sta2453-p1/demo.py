import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim


class PutNet(nn.Module):
    """
    Example of model Neural Network that could be trained price model put option.
    TODO: modify me!
    """

    def __init__(self) -> None:
        super(PutNet, self).__init__()
        self.l1 = nn.Linear(5, 23)
        self.l2 = nn.Linear(23, 23)
        self.l3 = nn.Linear(23, 11)
        self.out = nn.Linear(11, 1)
        # self.l1 = nn.Linear(5, 13)
        # self.l2 = nn.Linear(13, 26)
        # self.l3 = nn.Linear(26, 52)
        # self.l4 = nn.Linear(52, 26)
        # self.l5 = nn.Linear(26, 13)
        # self.l6 = nn.Linear(13, 5)
        # self.out = nn.Linear(5, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.out(x)
        # x = F.relu(self.l1(x))
        # x = F.relu(self.l2(x))
        # x = F.relu(self.l3(x))
        # x = F.relu(self.l4(x))
        # x = F.relu(self.l5(x))
        # x = F.relu(self.l6(x))
        # x = self.out(x)
        return x


def main():
    """Train the model and save the checkpoint"""

    # Create model
    model = PutNet()

    # Load dataset
    df = pd.read_csv("bs-put-1k.csv")


    # Set up training
    x = torch.Tensor(df[["S", "K", "T", "r", "sigma"]].to_numpy())
    y = torch.Tensor(df[["value"]].to_numpy())

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    # Train for 500 epochs
    for i in range(500):

        # TODO: Modify to account for dataset size
        y_hat = model(x)
        y = y

        # Calculate training loss
        training_loss = criterion(y_hat, y)

        # Take model step
        optimizer.zero_grad()
        training_loss.backward()
        optimizer.step()

        # Check validation loss
        with torch.no_grad():
            # TODO: use model proper validation set
            validation_loss = criterion(model(x), y)

        print(f"Iteration: {i} | Training Loss: {training_loss:.4f} | Validation Loss: {validation_loss:.4f} ")

    torch.save(model.state_dict(), "simple-model.pt")


if __name__ == "__main__":
    main()
