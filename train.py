import torchvision
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from VGG import VGG

def main():
    #下载训练集与测试集
    train_data = torchvision.datasets.CIFAR10(root = "dataset", train = True, transform = torchvision.transforms.ToTensor(), download = True)
    test_data = torchvision.datasets.CIFAR10(root = "dataset", train = False, transform = torchvision.transforms.ToTensor(), download = True)

    #加载训练集与测试集
    train_dataloader = DataLoader(train_data, batch_size = 64)
    test_dataloader = DataLoader(test_data, batch_size = 64)

    #设备
    device = torch.device("cuda")

    #创建或者加载模型模型
    VGG_model = VGG()
    VGG_model = VGG_model.to(device)
    # VGG_model = torch.load("VGG16.pth") #直接加载整个模型，无需提前定义
    # VGG_model.load_state_dict(torch.load("VGG16.pth")) #加载参数，需要提前定义模型

    #交叉熵损失函数
    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.to(device)

    #梯度下降
    optimizer = torch.optim.SGD(VGG_model.parameters(), lr = 0.1)

    #配置参数
    epoch = 100
    step = 0

    #训练
    VGG_model.train() #设置为训练模式(有一些层在训练和测试阶段所作的事情不一致，所以需要加上)
    for i in range(epoch):
        print(f"第{i+1}轮开始")
        for data in train_dataloader:
            inputs, labels = data #输入图像以及对应标签
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = VGG_model(inputs)
            loss = loss_function(outputs, labels)
            optimizer.zero_grad() #梯度清零，保证每一个批次的训练都是独立的，避免梯度累加产生影响
            loss.backward() #进行反向传播计算，根据损失值计算出每个可以训练的参数的梯度
            optimizer.step() #依据计算得到的梯度，使用SGD对参数进行更新
            step += 1
            if(step % 100 == 0):
                print(f"训练次数：{step}，Loss：{loss.item()}")

    #保存
    torch.save(VGG_model, f"VGG16.pth")#存整个模型与参数
    #torch.save(VGG_model.state_dict(), f"VGG16.pth")只存参数

    # 测试
    VGG_model.eval() #设置为测试模式
    with torch.no_grad(): 
        """在这个代码块内，所有张量requires_grad被设置成False，即使张量原本需要计算梯度，也不会跟踪计算图，可以节省内存和资源，一般用于测试"""
        total_accuracy = 0
        for data in test_dataloader:
            inputs, labels = data #输入图像以及对应标签
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = VGG_model(inputs)
            accuracy = (outputs.argmax(1) == labels).sum()
            total_accuracy += accuracy
        print(f"测试集上的正确率为：{total_accuracy / len(test_data)}")

if __name__ == "__main__":
    main()