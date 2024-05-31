from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
from torch.utils.data import WeightedRandomSampler, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

loss_fn = nn.CrossEntropyLoss()  # 损失函数在外部定义一次


def train(model1, device, dataset, optimizer1, epoch1):
    model1.train()  # 设置模型为训练模式
    correct = 0
    all_len = 0
    for x, y in tqdm(dataset):
        x, y = x.to(device), y.to(device)  # 将数据移动到设备上进行计算
        optimizer1.zero_grad()  # 梯度清零
        output = model1(x)  # 模型前向传播
        pred = output.max(1, keepdim=True)[1]  # 获取预测结果
        correct += pred.eq(y.view_as(pred)).sum().item()  # 统计预测正确的数量
        all_len += len(x)  # 统计样本数量
        loss = loss_fn(output, y)  # 计算损失
        loss.backward()  # 反向传播，计算梯度
        optimizer1.step()  # 更新模型参数
    print(f"第 {epoch1} 次训练的Train准确率：{100. * correct / all_len:.2f}%")  # 打印训练准确率


def vaild(model, device, dataset):
    model.eval()  # 设置模型为评估模式
    correct = 0
    test_loss = 0
    all_len = 0
    with torch.no_grad():
        for x, target in dataset:
            x, target = x.to(device), target.to(device)  # 将数据移动到设备上进行计算
            output = model(x)  # 模型前向传播
            loss = loss_fn(output, target)  # 计算损失
            test_loss += loss.item()  # 累计测试损失
            pred = output.argmax(dim=1, keepdim=True)  # 获取预测结果
            correct += pred.eq(target.view_as(pred)).sum().item()  # 统计预测正确的数量
            all_len += len(x)  # 统计样本数量
    print(f"Test 准确率：{100. * correct / all_len:.2f}%")  # 打印测试准确率
    return 100. * correct / all_len  # 返回测试准确率


if __name__ == '__main__':

    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')

    LR = 0.0001  # 学习率
    EPOCH = 100  # 训练轮数
    BATCH_SIZE = 24  # 批量大小

    train_root = r"img"  # 训练数据根目录

    # 数据加载及处理
    train_transform = transforms.Compose([
        transforms.Resize(256),  # 调整图像大小为256x256
        transforms.RandomResizedCrop(244, scale=(0.6, 1.0), ratio=(0.8, 1.0)),  # 随机裁剪图像为244x244
        transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
        transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0),  # 改变图像的亮度
        transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0),  # 改变图像的对比度
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize(mean=[0.4848, 0.4435, 0.4023], std=[0.2744, 0.2688, 0.2757])  # 对图像进行标准化
    ])

    # 图像读取转换
    all_data = torchvision.datasets.ImageFolder(
        root=train_root,
        transform=train_transform
    )

    dic = all_data.class_to_idx  # 类别映射表
    print(dic)  # 打印dic,记录数据集顺序。

    # 计算每个类别的样本数量
    class_counts = Counter(all_data.targets)

    # 计算每个类别的样本权重
    weights = [1.0 / class_counts[class_idx] for class_idx in all_data.targets]

    # 创建一个权重采样器
    sampler = WeightedRandomSampler(weights, len(all_data), replacement=True)

    # 按80-20划分训练集和验证集
    train_size = int(0.8 * len(all_data))
    valid_size = len(all_data) - train_size
    train_dataset, valid_dataset = random_split(all_data, [train_size, valid_size])

    # 使用采样器对训练集进行加权采样
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 加载预训练的 ResNet-50 模型，并替换掉最后一层全连接层（fc），使其适应当前任务（共12个类别）。
    model_1 = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
    model_1.fc = nn.Sequential(nn.Linear(2048, 12))

    # 加载已训练好的模型参数, 可选。
    # model_1.load_state_dict(torch.load(r'F:\pythonProject\pytorch\cat_resnet\model\best_model_train100.00.pth'))
    # model_1.train()

    # 设置模型为训练模式
    model_1.to(DEVICE)
    # 使用Adam优化器
    optimizer = optim.Adam(model_1.parameters(), lr=0.001, weight_decay=1e-4)

    # 设置初始的最高准确率为 90.0，并初始化最优模型。
    max_accuracy = 90.0
    # 最优模型全局变量
    best_model = None

    for epoch in range(1, EPOCH + 1):
        train(model_1, DEVICE, train_loader, optimizer, epoch)
        accu = vaild(model_1, DEVICE, valid_loader)
        # 保存准确率最高的模型
        if accu > max_accuracy:
            max_accuracy = accu
            best_model = model_1.state_dict()  # 或者使用 torch.save() 保存整个模型

    # 打印最高准确率
    print("最高成功率： ", max_accuracy)

    # 保存最优模型
    torch.save(best_model, fr"model\best_model_train{max_accuracy:.2f}.pth")
