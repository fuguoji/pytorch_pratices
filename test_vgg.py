import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import argparse

from src.vgg import VGGModel

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content',
                        default='./data/png/content.png',
                        type=str)
    parser.add_argument('--style',
                        default='./data/png/style.png',
                        type=str)
    parser.add_argument('--max_size',
                        default=400,
                        type=int)
    parser.add_argument('--epochs',
                        default=2000,
                        type=int)
    parser.add_argument('--style_weight',
                        default=100,
                        type=float)
    parser.add_argument('--lr',
                        default=0.003,
                        type=float)
    args = parser.parse_args()

    return args

def load_image(path, transform=None, max_size=None, shape=None):
    image = Image.open(path)

    if max_size:
        scale = max_size / max(image.size)
        size = np.array(image.size) * scale
        image = image.resize(size.astype(int), Image.ANTIALIAS)

    if shape:
        image = image.resize(shape, Image.LANCZOS)

    if transform:
        image = transform(image).unsqueeze(0)

    return image

def main(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])
    content = load_image(args.content, transform, args.max_size)
    style = load_image(args.style, transform, shape=[content.size(2), content.size(3)])

    target = content.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([target], lr=args.lr, betas=[0.5, 0.999])
    vgg = VGGModel()

    for epoch in range(args.epochs):
        target_features = vgg(target)
        content_features = vgg(content)
        style_features = vgg(style)

        style_loss = 0
        content_loss = 0
        for f1, f2, f3 in zip(target_features, content_features, style_features):
            content_loss += torch.mean((f1 - f2)**2)

            _, c, h, w = f1.size()
            f1 = f1.view(c, h * w)
            f3 = f3.view(c, h * w)

            f1 = torch.mm(f1, f1.t())
            f3 = torch.mm(f3, f3.t())

            style_loss += torch.mean((f1 - f3)**2) / (c * h * w)
    
    loss = content_loss + config.style_weight * style_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch {}/{}, Content loss: {:.4f}, Style loss: {:.4f}'
          .format(epoch+1, args.epochs, content_loss.item(), style_loss.item()))
    
    if (epoch+1) % 500 == 0:
        denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
        img = target.clone().sequeeze()
        img = denorm(img).clamp_(0, 1)
        save_image(img, 'output-{}.png'.format(epoch+1))

if __name__ == '__main__':
    main(get_args())
