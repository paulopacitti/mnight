import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(784, 800)
        self.fc2 = nn.Linear(800, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


def setup_dataset(batch_size=16):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    target_transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.LongTensor([x])),
        transforms.Lambda(lambda x: nn.functional.one_hot(x, 10)),
        transforms.Lambda(lambda x: x.squeeze(0).float())
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True,
                                   transform=transform, target_transform=target_transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True,
                                  transform=transform, target_transform=target_transform)

    return train_dataset, test_dataset


def train(model, train_loader, val_loader, optimizer, epochs, device):
    model.to(device)
    model.train()
    train_loss_history = []
    val_loss_history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if i % 100 == 0:
                print('Epoch: {} [{}/{} ({:.0f}%)]\t Training Loss: {:.6f}'.format(
                    epoch, i * len(data), len(train_loader.dataset),
                    100. * i / len(train_loader), loss.item()))

        train_loss /= len(train_loader)
        train_loss_history.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = nn.functional.cross_entropy(output, target)
                val_loss += loss.item()

        # Calculate average validation loss for the epoch
        val_loss /= len(val_loader)
        val_loss_history.append(val_loss)
        print('Epoch: {} Validation Loss: {:.6f}'.format(epoch, val_loss))

    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss History')
    plt.legend()
    plt.show()


def test(model, test_loader, device):
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.cross_entropy(
                output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            target = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def save_model(model, path='./model.pt'):
    torch.save(model.state_dict(), path)


def main():
    parser = argparse.ArgumentParser(
        description='🌙 mnight: MNIST handwritten digits MLP classifier implemented in pytorch in one night')
    subparsers = parser.add_subparsers(dest='command', required=True)
    train_parser = subparsers.add_parser(
        'train', help='train the model')
    train_parser.add_argument('--epochs', type=int,
                              default=2, help='number of epochs')
    train_parser.add_argument('--batch-size', type=int,
                              default=16, help='batch size')
    train_parser.add_argument('--lr', type=float,
                              default=0.01, help='learning rate')
    train_parser.add_argument(
        '--device', type=str, default='cpu', help='device to train the model on')

    test_parser = subparsers.add_parser('test', help='test the model')
    test_parser.add_argument(
        '--device', type=str, default='cpu', help='device to test the model on')
    test_parser.add_argument(
        '--model', type=str, default='./model.pt', help='path to the model')
    test_parser.add_argument(
        '--batch-size', type=int, default=16, help='batch size')

    args = parser.parse_args()

    if args.command == 'train':
        epochs = args.epochs
        batch_size = args.batch_size
        lr = args.lr
        device = args.device

        train_dataset, _ = setup_dataset(batch_size=batch_size)
        train_len = int(len(train_dataset) * 0.8)
        val_len = len(train_dataset) - train_len

        train_loader, val_loader = torch.utils.data.random_split(
            train_dataset, [train_len, val_len])
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            val_loader, batch_size=batch_size, shuffle=True)

        model = Model()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        device = torch.device(device)
        train(model=model, train_loader=train_loader, val_loader=val_loader,
              optimizer=optimizer, epochs=epochs, device=device)

        print('Training complete. Saving model...')
        save_model(model)
        print('Model saved.')
    elif args.command == 'test':
        batch_size = args.batch_size
        device = args.device
        model_path = args.model

        _, test_dataset = setup_dataset()
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True)
        model = Model()
        if os.path.exists(model_path):
            model.load_state_dict(torch.load('./model.pt', weights_only=True))
        else:
            print('Model not found. Testing blank model.')
        test(model=model, test_loader=test_loader, device=device)


if __name__ == '__main__':
    main()
