import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F

from torchvision import models, datasets, transforms
#from torchvision.models import vgg16, resnet50, mobilenet_v2, resnet18
from torch import nn


from models import VGG16, MobileNetV2, VisionTransformer, ResNet18


def get_CIFAR10(root="./"):
    input_size = 32
    num_classes = 10
    mean, std = [0.49139968, 0.48215827, 0.44653124],[0.24703233, 0.24348505, 0.26158768]
    normalize = transforms.Normalize((mean), (std))
    
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_dataset = datasets.CIFAR10(
        root + "data/CIFAR10", train=True, transform=train_transform, download=True
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_dataset = datasets.CIFAR10(
        root + "data/CIFAR10", train=False, transform=test_transform, download=True
    )

    return input_size, num_classes, train_dataset, test_dataset


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # self.resnet = models.resnet50(pretrained=False, num_classes=10)

        # self.resnet.conv1 = torch.nn.Conv2d(
            # 3, 64, kernel_size=3, stride=1, padding=1, bias=False
        # )
        # self.resnet.maxpool = torch.nn.Identity()


        self.resnet = mobilenet_v2(pretrained=True)
        self.resnet.classifier[1] = nn.Linear(1280, 10)
        
        
        
    def forward(self, x):
        x = self.resnet(x)
        x = F.log_softmax(x, dim=1)

        return x


def train(model, train_loader, optimizer, epoch):
    model.train()

    total_loss = []

    for data, target in tqdm(train_loader):
        data = data.cuda()
        target = target.cuda()

        optimizer.zero_grad()

        prediction = model(data)
        prediction = F.log_softmax(prediction, dim=1)
        loss = F.nll_loss(prediction, target)

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    avg_loss = sum(total_loss) / len(total_loss)
    print(f"Epoch: {epoch}:")
    print(f"Train Set: Average Loss: {avg_loss:.2f}")


def test(model, test_loader):
    model.eval()

    loss = 0
    correct = 0

    for data, target in test_loader:
        with torch.no_grad():
            data = data.cuda()
            target = target.cuda()

            prediction = model(data)
            prediction = F.log_softmax(prediction, dim=1)
            loss += F.nll_loss(prediction, target, reduction="sum")

            prediction = prediction.max(1)[1]
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    loss /= len(test_loader.dataset)

    percentage_correct = 100.0 * correct / len(test_loader.dataset)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            loss, correct, len(test_loader.dataset), percentage_correct
        )
    )

    return loss, percentage_correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=100, help="number of epochs to train (default: 50)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.05, help="learning rate (default: 0.05)"
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)

    input_size, num_classes, train_dataset, test_dataset = get_CIFAR10()

    kwargs = {"num_workers": 2, "pin_memory": True}

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=5000, shuffle=False, **kwargs
    )

    #model = Model()
    model = MobileNetV2()
    #model = ResNet18()
    
    print(model)
    
    for key in model.state_dict().keys():
        # print(model.state_dict()[key].numel())
        # print(model.state_dict()[key])
        print(key)
    
    model = model.cuda()

    milestones = [25, 50, 80]

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1
    )

    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)
        
        scheduler.step()

    torch.save(model.state_dict(), "cifar_model.pt")


if __name__ == "__main__":
    main()