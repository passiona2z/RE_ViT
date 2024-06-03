import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from transformers import ViTModel, ViTConfig

class VisionTransformer(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.1, pretrained_model_name='google/vit-base-patch16-224'):
        super().__init__()
        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        x = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰 사용
        x = self.dropout(x)
        x = self.classifier(x)
        return x

def train(vit, trainloader, criterion, optimizer, device, clip_value):
    vit.train()
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = vit(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(vit.parameters(), clip_value)

        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(trainloader)

def evaluate(vit, dataloader, criterion, device):
    vit.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = vit(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total, running_loss / len(dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vision Transformer')
    parser.add_argument('--img_size', default=224, type=int, help='image size')  # 논문에서 사용한 기본 이미지 크기
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epochs', default=10, type=int, help='training epoch')  # Set default epochs to 10
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0.1, type=float, help='weight decay')
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
    parser.add_argument('--mode', default='train', type=str, help='train or evaluation')
    parser.add_argument('--pretrained', default=1, type=int, help='use pretrained model')
    parser.add_argument('--clip_value', default=1.0, type=float, help='gradient clipping value')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='dropout rate')  # dropout rate 추가
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # CIFAR-10 데이터셋 로드 및 변환
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model 초기화
    vit = VisionTransformer(num_classes=args.num_classes, dropout_rate=args.dropout_rate).to(device)

    criterion = nn.CrossEntropyLoss()

    if args.mode == 'train':
        optimizer = torch.optim.SGD(vit.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        
        # Initialize the ReduceLROnPlateau scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)

        best_acc = 0
        for epoch in range(args.epochs):
            train_loss = train(vit, trainloader, criterion, optimizer, device, args.clip_value)
            val_acc, val_loss = evaluate(vit, valloader, criterion, device)
            print('[Epoch %d] Train loss: %.3f, Val loss: %.3f, Val acc: %.2f %%' % (epoch + 1, train_loss, val_loss, val_acc))

            # Step the scheduler based on the validation loss
            scheduler.step(val_loss)

            if val_acc > best_acc:
                best_acc = val_acc
                print('Saving the best model with val acc: %.2f %%' % val_acc)
                torch.save(vit.state_dict(), './model7.pth')

    else:
        vit.load_state_dict(torch.load('./model7.pth'))
        test_acc, test_loss = evaluate(vit, valloader, criterion, device)
        print('Test loss: %.3f, Test acc: %.2f %%' % (test_loss, test_acc))
