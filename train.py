import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import accuracy_score, recall_score, f1_score
import matplotlib.pyplot as plt

def train(model, device, dataset, optimizer, epoch):
    model.train()
    correct = 0
    total = 0
    for x, y in tqdm(dataset):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()
        total += y.size(0)
    print(f"第 {epoch} 次训练的准确率：{100. * correct / total:.2f}%")

def validate(model, device, dataset):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in dataset:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = output.max(1, keepdim=True)[1]
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"测试集上的准确率：{accuracy*100:.2f}%")
    print(f"测试集上的召回率：{recall*100:.2f}%")
    print(f"测试集上的F1分数：{f1*100:.2f}%")
    return accuracy, recall, f1

def plot_metrics(metrics, epochs):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for i, metric in enumerate(['Accuracy', 'Recall', 'F1 Score']):
        ax[i].plot(range(1, epochs + 1), metrics[metric], marker='o')
        ax[i].set_title(f'{metric} over Epochs')
        ax[i].set_xlabel('Epoch')
        ax[i].set_ylabel(metric)
        ax[i].grid(True)
    plt.tight_layout()
    plt.savefig('metrics.png')
    plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.0001
epochs = 100
batch_size = 24
train_root = "img"

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(244, scale=(0.6, 1.0), ratio=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5),
    transforms.ColorJitter(contrast=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4848, 0.4435, 0.4023], std=[0.2744, 0.2688, 0.2757])
])

all_data = torchvision.datasets.ImageFolder(root=train_root, transform=train_transform)
class_counts = Counter(all_data.targets)
weights = [1.0 / class_counts[class_idx] for class_idx in all_data.targets]
sampler = WeightedRandomSampler(weights, len(all_data), replacement=True)
train_data = torch.utils.data.DataLoader(all_data, batch_size=batch_size, sampler=sampler)
valid_data = torch.utils.data.DataLoader(all_data, batch_size=batch_size)

model = torchvision.models.resnet50(pretrained=True)
model.fc = nn.Sequential(nn.Linear(2048, 12))
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

best_accuracy = 0.0
best_model = None
metrics = {'Accuracy': [], 'Recall': [], 'F1 Score': []}

for epoch in range(1, epochs + 1):
    train(model, device, train_data, optimizer, epoch)
    accuracy, recall, f1 = validate(model, device, valid_data)
    metrics['Accuracy'].append(accuracy)
    metrics['Recall'].append(recall)
    metrics['F1 Score'].append(f1)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model.state_dict()

torch.save(best_model, f"model/best_model_{best_accuracy:.2f}.pth")
plot_metrics(metrics, epochs)
