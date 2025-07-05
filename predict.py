import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import argparse
from tqdm import tqdm

# 设置设备 - 如果有CUDA支持的GPU则使用，否则使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 设置图像处理参数
IMG_HEIGHT = 224  # 图像高度
IMG_WIDTH = 224   # 图像宽度

def load_class_names(dataset_path):
    """
    加载类别名称
    
    功能:
    - 从数据集目录中读取类别名称
    
    参数:
    - dataset_path: 数据集路径
    
    返回:
    - class_names: 按字母顺序排序的类别名称列表
    """
    class_names = sorted(os.listdir(dataset_path))  # 获取数据集目录下的所有文件夹名称并排序
    return class_names

def build_model(num_classes):
    """
    构建模型架构 - 与训练时一致
    
    功能:
    - 创建与训练时相同架构的MobileNetV2模型
    
    参数:
    - num_classes: 输出类别的数量
    
    返回:
    - model: 配置好的PyTorch模型
    """
    # 初始化MobileNetV2模型，但不加载预训练权重（因为会加载我们自己的权重）
    model = models.mobilenet_v2(weights=None)
    
    # 修改分类器 - 必须与训练时使用的架构完全一致
    in_features = model.classifier[1].in_features  # 获取特征维度
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),                 # 添加dropout防止过拟合
        nn.Linear(in_features, 512),     # 全连接层降维到512
        nn.ReLU(),                       # ReLU激活函数
        nn.Dropout(0.3),                 # 再次添加dropout进一步防止过拟合
        nn.Linear(512, num_classes)      # 最终分类层，输出对应类别数量的logits
    )
    
    # 将模型移动到指定设备(GPU或CPU)
    return model.to(device)

def predict_image(model, image_path, class_names):
    """
    预测单张图像的植物病虫害类别
    
    功能:
    1. 加载和预处理图像
    2. 使用模型进行预测
    3. 显示结果及置信度
    4. 输出前5个可能的类别及其置信度
    
    参数:
    - model: 训练好的模型
    - image_path: 待预测图像的路径
    - class_names: 类别名称列表
    
    返回:
    - predicted_class: 预测的类别名称
    - confidence: 预测的置信度
    """
    # 检查图像是否存在
    if not os.path.exists(image_path):
        print(f"错误：图片 {image_path} 不存在")
        return None, 0
    
    # 添加预测进度信息
    print("正在进行预测...")
    
    # 创建进度条 - 显示预测过程的各个阶段
    progress = tqdm(total=3, desc="预测过程")
    
    # 加载图像
    progress.set_description("加载图像")  # 更新进度条描述
    try:
        img = Image.open(image_path).convert('RGB')  # 确保图像为RGB格式
    except Exception as e:
        print(f"无法加载图像: {e}")
        return None, 0
    progress.update(1)  # 更新进度条，表示第一步完成
    
    # 预处理图像 - 调整大小、转换为张量并标准化
    progress.set_description("预处理图像")
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),  # 调整大小
        transforms.ToTensor(),                       # 转换为张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)  # 添加批次维度并移至设备
    progress.update(1)  # 更新进度条，表示第二步完成
    
    # 预测 - 使用模型进行推理
    progress.set_description("执行推理")
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 禁用梯度计算，加速推理并节省内存
        outputs = model(img_tensor)
        # 计算softmax获取概率分布
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # 获取最可能的类别及其置信度
        predicted_class = torch.argmax(probabilities, 1).item()
        confidence = round(probabilities[0][predicted_class].item() * 100, 2)
        
        # 获取前5个最可能的类别及其置信度
        top5_values, top5_indices = torch.topk(probabilities, 5)
        top5_values = top5_values.squeeze().cpu().numpy()  # 转换为numpy数组
        top5_indices = top5_indices.squeeze().cpu().numpy()
    progress.update(1)  # 更新进度条，表示第三步完成
    progress.close()  # 关闭进度条
    
    # 显示结果 - 绘制图像并标注预测类别和置信度
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f"预测类别: {class_names[predicted_class]} \n置信度: {confidence}%")
    plt.axis('off')  # 隐藏坐标轴
    plt.show()
    
    # 打印预测结果
    print(f"预测结果: {class_names[predicted_class]}")
    print(f"置信度: {confidence}%")
    
    # 输出前5个可能的结果及其置信度
    print("\n前5个可能的结果:")
    for i in range(min(5, len(class_names))):
        if i < len(top5_indices):
            idx = top5_indices[i]
            conf = round(top5_values[i] * 100, 2)
            print(f"{i+1}. {class_names[idx]}: {conf}%")
    
    return class_names[predicted_class], confidence

def main():
    """
    主函数 - 处理命令行参数并进行图像预测
    
    功能:
    1. 解析命令行参数
    2. 加载模型和类别名称
    3. 预测指定图像
    """
    # 参数解析 - 定义命令行参数
    parser = argparse.ArgumentParser(description='预测植物病虫害类别')
    parser.add_argument('--image', type=str, required=True, help='图片路径')
    parser.add_argument('--model', type=str, default='plant_disease_model_final.pth', help='模型路径')
    parser.add_argument('--dataset', type=str, 
                       default='Plant_leaf_diseases_dataset_with_augmentation/Plant_leave_diseases_dataset_with_augmentation', 
                       help='数据集路径，用于获取类别名称')
    
    args = parser.parse_args()
    
    # 检查文件是否存在 - 确保输入参数有效
    if not os.path.exists(args.image):
        print(f"错误：图片 {args.image} 不存在")
        return
    
    if not os.path.exists(args.model):
        print(f"错误：模型 {args.model} 不存在")
        return
    
    # 加载类别名称
    if not os.path.exists(args.dataset):
        print(f"警告：数据集路径 {args.dataset} 不存在，将使用默认类别名称")
        class_names = ["未知类别"]  # 如果找不到数据集，则使用默认类别名
    else:
        class_names = load_class_names(args.dataset)
    
    num_classes = len(class_names)
    
    # 构建并加载模型
    print(f"正在加载模型 {args.model}...")
    model = build_model(num_classes)  # 创建模型架构
    model.load_state_dict(torch.load(args.model, map_location=device))  # 加载权重
    model.eval()  # 设置为评估模式
    print("模型加载成功！")
    
    # 预测图像
    predict_image(model, args.image, class_names)

if __name__ == "__main__":
    main()
