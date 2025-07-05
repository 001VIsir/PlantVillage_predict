import datetime
import os
import time
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# 设置设备 - 如果有CUDA支持的GPU则使用，否则使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 设置中文字体支持 - 防止图表标题显示为方框
def setup_chinese_font():
    """配置中文字体支持"""
    try:
        # 在Windows系统上
        if os.name == 'nt':
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
            plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
        # 在Linux和macOS系统上
        else:
            # 尝试找到常见的中文字体
            chinese_fonts = [
                'Noto Sans CJK SC',
                'Droid Sans Fallback',
                'WenQuanYi Micro Hei',
                'Microsoft YaHei'
            ]

            # 设置第一个找到的可用中文字体
            for font in matplotlib.font_manager.findSystemFonts():
                for font_name in chinese_fonts:
                    if font_name.lower() in font.lower():
                        plt.rcParams['font.sans-serif'] = [font_name]
                        plt.rcParams['axes.unicode_minus'] = False
                        return
    except:
        # 如果找不到中文字体，使用默认设置
        print("警告: 无法加载中文字体，图表标题可能无法正确显示中文")

# 调用字体设置函数
setup_chinese_font()

# 数据集路径 - 指定PlantVillage数据集的位置
dataset_path = "Plant_leaf_diseases_dataset_with_augmentation/Plant_leave_diseases_dataset_with_augmentation"

# 定义模型训练和图像处理的全局参数
IMG_HEIGHT = 256  # 图像高度
IMG_WIDTH = 256   # 图像宽度
BATCH_SIZE = 128   # 批处理大小
EPOCHS = 15       # 训练轮次
LEARNING_RATE = 0.001  # 降低初始学习率


def load_and_preprocess_data():
    """
    加载和预处理数据

    功能:
    1. 定义数据增强和预处理转换
    2. 加载PlantVillage数据集
    3. 将数据集分为训练集和验证集
    4. 创建数据加载器

    返回:
    - train_loader: 训练数据加载器
    - val_loader: 验证数据加载器
    - num_classes: 类别数量
    - class_names: 类别名称列表
    """
    print("正在加载和预处理数据...")

    # 数据增强与预处理 - 针对训练集
    # 调整数据增强强度，避免过度正则化
    train_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),  # 调整图像大小
        transforms.RandomRotation(10),               # 减少随机旋转角度
        transforms.RandomHorizontalFlip(),           # 随机水平翻转
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # 降低调整强度
        transforms.ToTensor(),                       # 将图像转换为张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 使用ImageNet均值和标准差进行标准化
    ])

    # 验证集只需基本预处理 - 不需要数据增强
    val_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),  # 调整图像大小
        transforms.ToTensor(),                       # 将图像转换为张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 使用ImageNet均值和标准差进行标准化
    ])

    # 创建完整数据集 - 使用ImageFolder自动从文件夹结构中加载数据和标签
    full_dataset = ImageFolder(root=dataset_path, transform=train_transform)

    # 分割训练集和验证集 - 80%用于训练，20%用于验证
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # 更新验证集的转换 - 确保验证集使用val_transform而非train_transform
    val_dataset.dataset = ImageFolder(root=dataset_path, transform=val_transform)

    # 创建数据加载器 - 用于批量加载数据
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,         # 随机打乱训练数据
        num_workers=4,        # 使用4个工作线程加载数据
        pin_memory=True       # 将数据加载到CUDA固定内存中，加速GPU训练
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,        # 验证集不需要打乱
        num_workers=4,        # 使用4个工作线程加载数据
        pin_memory=True       # 将数据加载到CUDA固定内存中，加速GPU训练
    )

    # 获取类别信息 - 从数据集中提取类别名称和数量
    class_names = full_dataset.classes  # 类别名称列表
    num_classes = len(class_names)      # 类别数量

    print(f"共有 {num_classes} 种植物病虫害类别")

    # 打印样本数信息
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")

    return train_loader, val_loader, num_classes, class_names


def build_model(num_classes):
    """
    构建深度神经网络模型

    功能:
    1. 加载预训练的MobileNetV2模型
    2. 部分解冻特征提取层
    3. 修改分类器以适应植物病虫害分类

    参数:
    - num_classes: 输出类别的数量

    返回:
    - model: 配置好的PyTorch模型
    """
    print("正在构建模型...")

    # 使用预训练的MobileNetV2 - 轻量级但高效的CNN架构
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')  # 加载在ImageNet上预训练的权重

    # 解冻最后几层特征提取层 - 增加模型学习能力
    # 冻结大部分层，但解冻最后几层
    total_layers = len(list(model.features))
    print(f"模型共有 {total_layers} 个特征层")

    # 解冻最后10%的层
    for param in model.parameters():
        param.requires_grad = False

    # 解冻最后部分层
    for i in range(int(total_layers * 0.9), total_layers):
        for param in model.features[i].parameters():
            param.requires_grad = True

    # 替换分类器 - 使用更简单的结构
    in_features = model.classifier[1].in_features  # 获取特征维度

    # 使用更简单的分类头
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),                 # 添加dropout防止过拟合
        nn.Linear(in_features, num_classes)  # 直接连接到输出层
    )

    # 将模型移动到指定设备(GPU或CPU)
    model = model.to(device)

    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型可训练参数: {trainable_params}/{total_params} ({trainable_params/total_params*100:.2f}%)")

    return model


def train_model(model, train_loader, val_loader, num_classes):
    """
    训练模型

    功能:
    1. 设置损失函数、优化器和学习率调度器
    2. 实现训练和验证循环
    3. 添加早停机制
    4. 保存最佳模型
    5. 记录训练过程

    参数:
    - model: 待训练的模型
    - train_loader: 训练数据加载器
    - val_loader: 验证数据加载器
    - num_classes: 类别数量

    返回:
    - history: 训练历史记录
    - model: 训练后的模型
    """
    print("开始训练模型...")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，适用于多分类问题
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)  # AdamW优化器，结合Adam和权重衰减

    # 使用CosineAnnealingLR学习率调度器 - 更好的学习率衰减
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*EPOCHS, eta_min=1e-6)

    # 用于早停的变量 - 当验证损失不再下降时停止训练
    best_val_loss = float('inf')  # 初始最佳验证损失设为无穷大
    best_val_acc = 0.0
    patience = 8  # 增加耐心值
    patience_counter = 0  # 记录验证损失未改善的轮数
    best_model_wts = None  # 存储最佳模型权重

    # 记录训练历史 - 用于后续可视化
    history = {
        'train_loss': [],  # 训练损失
        'train_acc': [],   # 训练准确率
        'val_loss': [],    # 验证损失
        'val_acc': []      # 验证准确率
    }

    # 计算总训练步数用于显示总进度
    total_steps = len(train_loader) * EPOCHS
    steps_done = 0

    # 记录训练开始时间 - 用于估计剩余时间
    start_time = time.time()

    # 创建保存训练日志的目录
    log_dir = "training_logs"
    os.makedirs(log_dir, exist_ok=True)  # 创建目录，如果已存在则不报错
    log_file = os.path.join(log_dir, f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    # 训练循环 - 迭代每个epoch
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()  # 记录当前epoch开始时间
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print('-' * 50)

        # 训练阶段
        model.train()  # 设置模型为训练模式（启用dropout和batchnorm）
        running_loss = 0.0  # 记录累计损失
        running_corrects = 0  # 记录累计正确预测数

        # 添加进度条 - 显示训练进度
        progress_bar = tqdm(train_loader, desc=f"训练 Epoch {epoch + 1}/{EPOCHS}",
                            ncols=100, colour="green")

        batch_count = 0
        for inputs, labels in progress_bar:
            batch_count += 1
            steps_done += 1

            # 将数据移到指定设备(GPU或CPU)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 清零梯度 - 防止梯度累积
            optimizer.zero_grad()

            # 前向传播 - 计算预测值
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # 计算损失

            # 反向传播和优化 - 计算梯度并更新权重
            loss.backward()
            optimizer.step()

            # 更新学习率
            scheduler.step()

            # 统计 - 计算批次损失和准确率
            _, preds = torch.max(outputs, 1)  # 获取每个样本的预测类别
            batch_loss = loss.item() * inputs.size(0)  # 计算批次总损失
            batch_corrects = torch.sum(preds == labels).item()  # 计算批次正确预测数
            batch_acc = batch_corrects / inputs.size(0)  # 计算批次准确率

            running_loss += batch_loss
            running_corrects += batch_corrects

            # 更新进度条信息 - 显示当前批次损失、准确率和总体进度
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{batch_acc:.4f}",
                '总进度': f"{steps_done}/{total_steps} ({steps_done / total_steps * 100:.1f}%)",
                'lr': f"{scheduler.get_last_lr()[0]:.6f}"  # 显示当前学习率
            })

            # 每10个batch输出一次详细信息到控制台和日志
            if batch_count % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                log_msg = f"Epoch {epoch + 1}/{EPOCHS}, Batch {batch_count}/{len(train_loader)}, "
                log_msg += f"Loss: {loss.item():.4f}, Acc: {batch_acc:.4f}, LR: {current_lr:.6f}"

                # 将信息写入日志文件
                with open(log_file, "a") as f:
                    f.write(log_msg + "\n")

        # 计算整个epoch的训练损失和准确率
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        # 计算本轮训练用时
        epoch_time = time.time() - epoch_start_time

        print(f"训练集 Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} 用时: {epoch_time:.1f}秒")

        # 验证阶段
        model.eval()  # 设置模型为评估模式（禁用dropout和使用batchnorm的评估模式）
        val_loss = 0.0
        val_corrects = 0
        total_samples = 0

        # 验证集进度条
        val_progress_bar = tqdm(val_loader, desc=f"验证 Epoch {epoch + 1}/{EPOCHS}",
                                ncols=100, colour="blue")

        # 在验证阶段禁用梯度计算，提高效率并节省内存
        with torch.no_grad():
            for inputs, labels in val_progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                batch_size = inputs.size(0)
                total_samples += batch_size

                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # 计算准确率
                _, preds = torch.max(outputs, 1)
                batch_corrects = torch.sum(preds == labels).item()

                val_loss += loss.item() * batch_size
                val_corrects += batch_corrects
                batch_acc = batch_corrects / batch_size

                # 更新验证进度条信息
                val_progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{batch_acc:.4f}"
                })

        # 计算整个验证集的损失和准确率
        val_loss /= total_samples
        val_acc = val_corrects / total_samples
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 获取当前学习率
        current_lr = scheduler.get_last_lr()[0]

        # 计算剩余时间估计
        time_elapsed = time.time() - start_time
        time_per_epoch = time_elapsed / (epoch + 1)
        estimated_remaining = time_per_epoch * (EPOCHS - epoch - 1)

        # 将秒转换为时分秒格式，便于阅读
        remaining_hrs = int(estimated_remaining // 3600)
        remaining_mins = int((estimated_remaining % 3600) // 60)
        remaining_secs = int(estimated_remaining % 60)

        # 打印验证结果和时间信息
        print(f"验证集 Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        print(f"当前学习率: {current_lr:.6f}")
        print(f"已用时间: {time_elapsed / 60:.1f}分钟")
        print(f"预计剩余时间: {remaining_hrs}小时 {remaining_mins}分钟 {remaining_secs}秒")

        # 记录到日志文件
        log_summary = f"\nEpoch {epoch + 1}/{EPOCHS} 总结:\n"
        log_summary += f"训练集 Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n"
        log_summary += f"验证集 Loss: {val_loss:.4f} Acc: {val_acc:.4f}\n"
        log_summary += f"当前学习率: {current_lr:.6f}\n"
        log_summary += f"已用时间: {time_elapsed / 60:.1f}分钟\n"
        log_summary += f"预计剩余时间: {remaining_hrs}小时 {remaining_mins}分钟 {remaining_secs}秒\n"
        log_summary += "-" * 50 + "\n"

        with open(log_file, "a") as f:
            f.write(log_summary)

        # 早停检查 - 如果验证损失改善，重置计数器并保存最佳模型
        # 同时检查验证准确率是否提高
        if val_loss < best_val_loss or val_acc > best_val_acc:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if val_acc > best_val_acc:
                best_val_acc = val_acc

            patience_counter = 0
            best_model_wts = model.state_dict().copy()  # 保存当前最佳模型权重
            # 保存最佳模型到文件
            torch.save(model.state_dict(), 'best_plant_disease_model.pth')
            print(f"模型已保存: best_plant_disease_model.pth")
        else:
            patience_counter += 1  # 验证损失未改善，计数器加1

        # 如果连续多轮验证损失未改善，提前停止训练
        if patience_counter >= patience:
            print(f"早停: 验证损失连续{patience}轮未改善")
            break

        print()

    # 计算总训练时间并格式化
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print(f"训练完成! 总用时: {hours}小时 {minutes}分钟 {seconds}秒")

    # 将训练完成信息记录到日志
    with open(log_file, "a") as f:
        f.write(f"\n训练完成! 总用时: {hours}小时 {minutes}分钟 {seconds}秒\n")

    # 加载最佳模型权重 - 确保返回的是最佳性能的模型
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    return history, model


def evaluate_model(history, model, val_loader, class_names):
    """
    评估模型性能

    功能:
    1. 绘制训练历史曲线
    2. 在验证集上进行预测
    3. 生成分类报告
    4. 绘制混淆矩阵

    参数:
    - history: 训练历史记录
    - model: 训练好的模型
    - val_loader: 验证数据加载器
    - class_names: 类别名称列表
    """
    print("正在评估模型性能...")

    # 创建图表保存目录
    plots_dir = "model_evaluation_plots"
    os.makedirs(plots_dir, exist_ok=True)

    # 设置图表字体大小
    plt.rcParams.update({'font.size': 12})

    # 绘制训练历史 - 可视化训练过程
    plt.figure(figsize=(14, 6))

    # 绘制准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], linewidth=2)
    plt.plot(history['val_acc'], linewidth=2)
    plt.title('模型准确率', fontsize=14)
    plt.ylabel('准确率', fontsize=12)
    plt.xlabel('轮次', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(['训练集', '验证集'], loc='lower right', fontsize=12)
    plt.tight_layout()

    # 绘制损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], linewidth=2)
    plt.plot(history['val_loss'], linewidth=2)
    plt.title('模型损失', fontsize=14)
    plt.ylabel('损失', fontsize=12)
    plt.xlabel('轮次', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(['训练集', '验证集'], loc='upper right', fontsize=12)
    plt.tight_layout()

    # 保存图像到文件
    history_path = os.path.join(plots_dir, 'training_history.png')
    plt.savefig(history_path, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"训练历史图已保存至: {history_path}")

    # 在验证集上进行预测
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    # 收集所有预测结果和真实标签
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="评估模型", ncols=100, colour="magenta"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())  # 将预测结果添加到列表
            all_labels.extend(labels.numpy())      # 将真实标签添加到列表
            all_probs.extend(probs.cpu().numpy())  # 存储预测概率

    # 打印分类报告 - 包括精确率、召回率、F1分数等详细指标
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("分类报告:")
    print(report)

    # 保存分类报告到文件
    report_path = os.path.join(plots_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"分类报告已保存至: {report_path}")

    # 绘制混淆矩阵 - 可视化模型在各类别上的性能
    plt.figure(figsize=(18, 14))
    cm = confusion_matrix(all_labels, all_preds)

    # 调整字体大小和标签旋转角度以防重叠
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 8}  # 调整注释字体大小
    )
    plt.title('混淆矩阵', fontsize=14)
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.xticks(rotation=90)  # 旋转x轴标签以防止重叠
    plt.yticks(rotation=0)
    plt.tight_layout()

    # 保存混淆矩阵图像
    cm_path = os.path.join(plots_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"混淆矩阵已保存至: {cm_path}")

    return all_probs, all_preds, all_labels


def predict_image(model, image_path, class_names):
    """
    预测单张图像的植物病虫害类别

    参数:
    - model: 训练好的模型
    - image_path: 待预测图像的路径
    - class_names: 类别名称列表

    返回:
    - 预测的类别名称
    - 预测的置信度
    """
    # 加载图像 - 读取图像并确保转换为RGB格式
    img = Image.open(image_path).convert('RGB')

    # 预处理 - 与验证集使用相同的预处理步骤
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),  # 调整大小
        transforms.ToTensor(),                       # 转换为张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)  # 添加批次维度并移至设备

    # 预测 - 使用模型进行推理
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 禁用梯度计算
        outputs = model(img_tensor)
        # 计算softmax以获得概率分布
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        # 获取最高概率的类别和置信度
        predicted_class = torch.argmax(probabilities, 1).item()
        confidence = round(probabilities[0][predicted_class].item() * 100, 2)

    # 显示结果 - 绘制图像并标注预测类别和置信度
    plt.figure(figsize=(9, 7))
    plt.imshow(img)
    plt.title(f"预测类别: {class_names[predicted_class]} \n置信度: {confidence}%", fontsize=14)
    plt.axis('off')  # 隐藏坐标轴
    plt.tight_layout()
    plt.show()

    # 打印top-3预测结果
    top_probs, top_classes = torch.topk(probabilities, 3, dim=1)
    print("\nTop-3预测结果:")
    for i in range(3):
        print(f"{class_names[top_classes[0][i].item()]}: {top_probs[0][i].item()*100:.2f}%")

    return class_names[predicted_class], confidence


def main():
    """
    主函数 - 协调整个训练和评估流程

    功能:
    1. 加载和预处理数据
    2. 构建模型
    3. 训练模型
    4. 评估模型
    5. 保存最终模型
    """
    print("植物病虫害识别系统启动...")

    # 加载和预处理数据
    train_loader, val_loader, num_classes, class_names = load_and_preprocess_data()

    # 构建模型
    model = build_model(num_classes)

    # 打印模型摘要 - 显示模型架构
    print(model)

    # 训练模型
    history, trained_model = train_model(model, train_loader, val_loader, num_classes)

    # 评估模型
    evaluate_model(history, trained_model, val_loader, class_names)

    # 保存最终模型
    torch.save(trained_model.state_dict(), 'plant_disease_model_final.pth')
    print("模型已保存为 'plant_disease_model_final.pth'")

    # 测试单张图像预测
    test_image = "path/to/your/test/image.jpg"  # 替换为您的测试图像路径
    if os.path.exists(test_image):
        print("\n测试单张图像预测...")
        predicted_class, confidence = predict_image(trained_model, test_image, class_names)
        print(f"预测结果: {predicted_class} ({confidence}%)")
    else:
        print(f"测试图像不存在: {test_image}")


if __name__ == "__main__":
    main()