from sys import argv, exit
from os.path import join, abspath, dirname
from torch import device, load, cuda, unsqueeze, max
from torch.nn import Sequential, Linear
from torchvision.models import resnet50
from PIL.Image import open
from PyQt6 import QtCore
from PyQt6.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("猫的分类")  # 设置窗口标题

        self.central_widget = QWidget()  # 创建一个中心部件
        self.setCentralWidget(self.central_widget)  # 将中心部件设置为窗口的中心部件

        self.font = QFont()
        self.font.setBold(True)  # 设置字体为加粗
        self.font.setPointSize(14)  # 设置字体大小为14

        self.image_label = QLabel(self.central_widget)  # 创建一个标签用于显示图片
        self.image_label.setFixedSize(self.central_widget.size())  # 设置标签大小与窗口大小一致
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)  # 设置标签文本居中对齐

        self.result_label = QLabel(self.central_widget)  # 创建一个标签用于显示分类结果
        self.result_label.setWordWrap(True)  # 设置标签文本自动换行
        self.result_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)  # 设置标签文本居中对齐

        self.upload_button = QPushButton("上传图片", self.central_widget)  # 创建一个按钮用于上传图片
        self.upload_button.setStyleSheet("background-color: Grey;")  # 设置按钮的背景颜色为灰色
        self.upload_button.clicked.connect(self.load_image)  # 将按钮的点击事件与加载图片的方法连接起来

        layout = QVBoxLayout(self.central_widget)  # 创建一个垂直布局管理器
        layout.addWidget(self.image_label)  # 将图片标签添加到布局中
        layout.addWidget(self.result_label)  # 将结果标签添加到布局中
        layout.addWidget(self.upload_button)  # 将上传按钮添加到布局中

        self.dic1 = {'伯曼猫': 0, '俄罗斯蓝猫': 1, '埃及猫': 2, '孟买猫': 3, '孟加拉豹猫': 4, '布偶猫': 5, '无毛猫': 6, '波斯猫': 7, '缅因猫': 8, '英国短毛猫': 9, '逞罗猫': 10, '阿比西亚猫': 11}
        # 创建字典，将类别索引与猫的品种名称对应起来和训练时打印的标签要一至

        if cuda.is_available():
            self.DEVICE = device('cuda')  # 如果可用的话，将设备设置为CUDA
        else:
            self.DEVICE = device('cpu')  # 否则将设备设置为CPU

        self.model = resnet50()  # 创建一个ResNet-50模型
        # 加载权重并替换原模型的fc层
        self.model.fc = Sequential(Linear(2048, 12))

        # 加载保存的模型参数
        self.model.load_state_dict(load(r"model/best_model_train100.00.pth"))
        # 将模型移动到 GPU 上（如果可用的话）
        self.model.to(self.DEVICE)
        # 设置模型为评估模式
        self.model.eval()

        self.transform = Compose([
            Resize((224, 224)),  # 将图片调整大小为 224x224 像素
            ToTensor(),  # 将图片转换为张量
            Normalize(mean=[0.4848, 0.4435, 0.4023], std=[0.2744, 0.2688, 0.2757])  # 对图片进行标准化处理
        ])  # 创建一个图片转换器，用于将上传的图片进行预处理

    def load_image(self):
        # 创建一个文件对话框
        file_dialog = QFileDialog(self)
        # 设置文件过滤器，仅显示图片文件
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
        # 如果对话框返回结果为 Accepted，表示用户选择了文件
        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            # 获取用户选择的文件路径
            selected_file = file_dialog.selectedFiles()[0]
            # 使用选中的文件创建 QPixmap 对象
            pixmap = QPixmap.fromImage(QImage(selected_file))
            # 将 QPixmap 对象缩放到指定大小并显示在 image_label 上
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size()))

            # 打开选中的文件并转换为 RGB 格式的图像
            image = open(selected_file).convert("RGB")
            # 使用预处理函数对图像进行处理
            img = self.transform(image)
            # 将预处理后的图像转换为指定设备上的张量
            img = img.to(self.DEVICE)
            # 在第一维上添加一个维度，将图像扩展为一个批次
            image = unsqueeze(img, 0)
            # 将图像输入模型进行推理
            outputs = self.model(image)
            # 从模型输出中获取预测标签
            _, predicted_labels = max(outputs, dim=1)
            # 获取预测结果的标签索引
            want = predicted_labels.item()

            # 设置结果标签的字体
            self.result_label.setFont(self.font)
            # 在结果标签上显示预测品种
            self.result_label.setText(f"品种: {self.dic1[want]}")


if __name__ == "__main__":
    # 创建应用程序对象
    app = QApplication(argv)
    # 设置应用程序窗口的图标
    # app.setWindowIcon(QIcon(''))
    # 创建主窗口对象
    window = MainWindow()
    # 显示主窗口
    window.show()
    # 进入应用程序的主循环，等待事件触发
    exit(app.exec())
