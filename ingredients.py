import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
from datasets import load_dataset

# Chuẩn bị transform ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
])

# Hàm tiền xử lý ảnh trong dataset
def preprocess(batch):
    batch['pixel_values'] = [transform(image.convert('RGB')) for image in batch['image']]
    return batch

def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch])
    return {'pixel_values': pixel_values, 'labels': labels}



# Tải dataset
dataset = load_dataset("scuccorese/food-ingredients-dataset")

# Tạo map label từ 'ingredient'
ingredients = dataset['train'].unique('ingredient')
label2id = {label: idx for idx, label in enumerate(sorted(ingredients))}
id2label = {idx: label for label, idx in label2id.items()}

def label_example(example):
    example['labels'] = label2id[example['ingredient']]
    return example

dataset = dataset.map(label_example)

# Áp dụng transform
dataset = dataset.with_transform(preprocess)

# Chia dataset train thành train + validation
split_data = dataset['train'].train_test_split(test_size=0.2)
train_dataset = split_data['train']
val_dataset = split_data['test']

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)

# Chuẩn bị mô hình
num_classes = len(label2id)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# Hàm train 1 epoch
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    for batch in dataloader:
        inputs = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

# Hàm đánh giá trên validation set
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

# Huấn luyện và đánh giá
num_epochs = 10
best_val_acc = 0.0
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
    # Lưu model tốt nhất
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_ingredient_classifier.pth')
        print(f'New best model saved with accuracy: {best_val_acc:.4f}')
