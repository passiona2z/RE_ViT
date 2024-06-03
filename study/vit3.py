import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from transformers import ViTModel, ViTConfig

class CustomViTModel(nn.Module):
    def __init__(self, pretrained_model_name='google/vit-base-patch16-224', num_classes=10, drop_rate=0.1):
        super().__init__()
        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        self.custom_head = nn.Sequential(
            nn.LayerNorm(self.vit.config.hidden_size),
            nn.Linear(self.vit.config.hidden_size, num_classes)
        )

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        x = outputs.last_hidden_state[:, 0, :]
        x = self.custom_head(x)
        return x

class LinearProjection(nn.Module):
    def __init__(self, patch_vec_size, num_patches, latent_vec_dim, drop_rate):
        super().__init__()
        self.linear_proj = nn.Linear(patch_vec_size, latent_vec_dim)
        self.cls_token = nn.Parameter(torch.randn(1, latent_vec_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, latent_vec_dim))
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.cat([self.cls_token.repeat(batch_size, 1, 1), self.linear_proj(x)], dim=1)
        # x += self.pos_embedding
        x = self.dropout(x)
        return x

class MultiheadedSelfAttention(nn.Module):
    def __init__(self, latent_vec_dim, num_heads, drop_rate):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_heads = num_heads
        self.latent_vec_dim = latent_vec_dim
        self.head_dim = int(latent_vec_dim / num_heads)
        self.query = nn.Linear(latent_vec_dim, latent_vec_dim)
        self.key = nn.Linear(latent_vec_dim, latent_vec_dim)
        self.value = nn.Linear(latent_vec_dim, latent_vec_dim)
        self.scale = torch.sqrt(self.head_dim * torch.ones(1)).to(device)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        batch_size = x.size(0)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 3, 1)  # k.t
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attention = torch.softmax(q @ k / self.scale, dim=-1)
        x = self.dropout(attention) @ v
        x = x.permute(0, 2, 1, 3).reshape(batch_size, -1, self.latent_vec_dim)
        return x, attention

class TFencoderLayer(nn.Module):
    def __init__(self, latent_vec_dim, num_heads, mlp_hidden_dim, drop_rate):
        super().__init__()
        self.ln1 = nn.LayerNorm(latent_vec_dim)
        self.ln2 = nn.LayerNorm(latent_vec_dim)
        self.msa = MultiheadedSelfAttention(latent_vec_dim=latent_vec_dim, num_heads=num_heads, drop_rate=drop_rate)
        self.dropout = nn.Dropout(drop_rate)
        self.mlp = nn.Sequential(
            nn.Linear(latent_vec_dim, mlp_hidden_dim),
            nn.GELU(), nn.Dropout(drop_rate),
            nn.Linear(mlp_hidden_dim, latent_vec_dim),
            nn.Dropout(drop_rate)
        )

    def forward(self, x):
        z = self.ln1(x)
        z, att = self.msa(z)
        z = self.dropout(z)
        x = x + z
        z = self.ln2(x)
        z = self.mlp(z)
        x = x + z
        return x, att

def train(model, trainloader, criterion, optimizer, device, clip_value):
    model.train()
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(trainloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total, running_loss / len(dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vision Transformer')
    parser.add_argument('--img_size', default=224, type=int, help='image size')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epochs', default=20, type=int, help='training epoch')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0.1, type=float, help='weight decay')
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
    parser.add_argument('--clip_value', default=1.0, type=float, help='gradient clipping value')
    parser.add_argument('--patch_size', default=16, type=int, help='patch size')
    parser.add_argument('--latent_vec_dim', default=768, type=int, help='latent vector dimension')
    parser.add_argument('--num_heads', default=12, type=int, help='number of attention heads')
    parser.add_argument('--mlp_hidden_dim', default=3072, type=int, help='hidden dimension of MLP')
    parser.add_argument('--drop_rate', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--num_layers', default=12, type=int, help='number of transformer layers')
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

    # pretrained ViT 모델 불러오기
    vit = CustomViTModel(pretrained_model_name='google/vit-base-patch16-224', num_classes=args.num_classes, drop_rate=args.drop_rate).to(device)

    criterion = nn.CrossEntropyLoss()

    if args.mode == 'train':
        optimizer = torch.optim.SGD(vit.parameters(), lr=args.lr, momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        best_acc = 0
        for epoch in range(args.epochs):
            train_loss = train(vit, trainloader, criterion, optimizer, device, args.clip_value)
            scheduler.step()  # Update the scheduler for each epoch
            val_acc, val_loss = evaluate(vit, valloader, criterion, device)
            print('[Epoch %d] Train loss: %.3f, Val loss: %.3f, Val acc: %.2f%%' % (epoch + 1, train_loss, val_loss, val_acc))
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(vit.state_dict(), 'model3.pth')
        print('Best validation accuracy: %.2f%%' % best_acc)
    elif args.mode == 'evaluate':
        vit.load_state_dict(torch.load('model3.pth'))
        val_acc, val_loss = evaluate(vit, valloader, criterion, device)
        print('Validation accuracy: %.2f%%' % val_acc)
