# net_torch.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Dane i konfiguracja
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
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Model
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

train_iter = iter(train_loader)

def train_torch_live(epoch):
    global train_iter
    model.train()

    try:
        data, target = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        data, target = next(train_iter)

    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()

    # Ewaluacja
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        total_loss = 0
        total_mse = 0

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = torch.exp(output)
            one_hot = F.one_hot(target, num_classes=10).float()

            total_loss += loss_fn(output, target).item() * data.size(0)
            total_mse += F.mse_loss(probs, one_hot, reduction='sum').item()
            preds = output.argmax(dim=1)
            total_correct += (preds == target).sum().item()
            total_samples += target.size(0)

            loss_val = total_loss / total_samples
            mse_val = total_mse / total_samples
            acc = total_correct / total_samples


        probs = torch.exp(output)
        one_hot = F.one_hot(target, num_classes=10).float()
        mse_val = F.mse_loss(probs, one_hot).item()
        loss_val = loss_fn(output, target).item()
        preds = output.argmax(dim=1)
        acc = (preds == target).float().mean().item()

    return loss_val, mse_val, acc