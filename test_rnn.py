import torch
import torchvision
import torchvision.transforms as transforms
import argparse

from src.rnn import RNNModel

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size',
                        default=28,
                        type=int)
    parser.add_argument('--hidden_size',
                        default=128,
                        type=int)
    parser.add_argument('--num_layers',
                        default=2,
                        type=int)
    parser.add_argument('--num_classes',
                        default=10,
                        type=int)
    parser.add_argument('--sequence_length',
                        default=28,
                        type=int)
    parser.add_argument('--epochs',
                        default=2,
                        type=int)
    parser.add_argument('--batch_size',
                        default=100,
                        type=int)
    parser.add_argument('--lr',
                        default=0.01,
                        type=float)
    args = parser.parse_args()

    return args

def get_mnist(args):
    train_data = torchvision.datasets.MNIST(root='./mnist',
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=False)
    test_data = torchvision.datasets.MNIST(root='./mnist',
                                            train=False,
                                            transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=args.batch_size,
                                              shuffle=False)

    return train_loader, test_loader

def evaluation(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    accuracy = 100 * correct / total

    return accuracy

def train(args, train_loader):
    model = RNNModel(input_size=args.input_size,
                     hidden_size=args.hidden_size,
                     num_layers=args.num_layers,
                     num_classes=args.num_classes)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, args.sequence_length, args.input_size)
            outputs = model(images)
            loss = loss_func(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                accuracy = evaluation(outputs, labels)
                print('Epoch {}/{}, Step {}/{}, Loss: {:.4f}, Accuracy: {:.4f}'
                      .format(epoch+1, args.epochs, i+1, len(train_loader), loss.item(), accuracy))

    return model

def test(args, model, test_loader):
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, args.sequence_length, args.input_size)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy on 1000 test images: {}'.format(100 * correct / total))

def main(args):
    train_loader, test_loader = get_mnist(args)
    model = train(args, train_loader)
    test(args, model, test_loader)

if __name__ == '__main__':
    main(get_args())
    