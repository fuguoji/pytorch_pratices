import os
import torch
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE
import argparse
import time

from src.cnn import CNNModel

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--EPOCHS', 
                        default=1, 
                        type=int)
    parser.add_argument('--BATCH_SIZE',
                        default=50,
                        type=int)
    parser.add_argument('--LR',
                        default=0.001,
                        type=float)
    
    args = parser.parse_args()

    return args

def get_data():
    DOWNLOAD_MNIST = False
    if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
        DOWNLOAD_MNIST = True

    train_data = torchvision.datasets.MNIST(
        root='./mnist/',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=DOWNLOAD_MNIST,
    )

    test_data = torchvision.datasets.MNIST(
        root='./mnist/', 
        train=False,
    )

    return train_data, test_data

def plot_data(data, labels):
    plt.imshow(data[0].numpy(), cmap='gray')
    plt.title('%i' % labels[0])
    plt.show()

def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9))
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize las layer')

def data_proc(train_data, test_data, BATCH_SIZE):
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000] / 255
    test_y = test_data.test_labels[:2000]

    return train_loader, test_x, test_y

def evaluation(test_y, pred_y):
    accuacy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))

    return accuacy

def main(args):
    cnn = CNNModel()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=args.LR)
    loss_func = torch.nn.CrossEntropyLoss()
    train_data, test_data = get_data()
    train_loader, test_x, test_y = data_proc(train_data, test_data, args.BATCH_SIZE)
    plt.ion()
    t1 = time.time()
    for epoch in range(args.EPOCHS):
        for step, (b_x, b_y) in enumerate(train_loader):
            output = cnn(b_x)[0]
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                test_output, last_layer = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuacy = evaluation(test_y, pred_y)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuacy)

                tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                plot_only = 500
                low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                labels = test_y.numpy()[:plot_only]
                plot_with_labels(low_dim_embs, labels)
    t2 = time.time()
    plt.ioff()

    print('training time: %s' % (t2-t1))

if __name__ == '__main__':
    main(get_args())