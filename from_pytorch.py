import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

def train_torch():
    batch_size = 64
    lr = 0.1
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(28*28, 64)
            self.drop = nn.Dropout(0.3)
            self.fc2 = nn.Linear(64, 10)

        def forward(self, x):
            x = x.view(-1, 28*28)
            x = torch.sigmoid(self.fc1(x))
            x = self.drop(x)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    model = Net().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.NLLLoss()

    losses, mses = [], []

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

        # Ewaluacja
        model.eval()
        test_loss = 0
        mse_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += loss_fn(output, target).item()

                probs = torch.exp(output)
                one_hot = F.one_hot(target, num_classes=10).float()
                mse_total += F.mse_loss(probs, one_hot, reduction='sum').item()

        test_loss /= len(test_loader)
        mse_avg = mse_total / len(test_loader.dataset)
        losses.append(test_loss)
        mses.append(mse_avg)

    return losses, mses
