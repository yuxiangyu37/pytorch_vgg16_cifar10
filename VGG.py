import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), #输出64*32*32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, 3, 1, 1), #输出64*32*32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2), #输出64*16*16
            nn.Conv2d(64, 128, 3, 1, 1), #输出128*16*16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, 3, 1, 1), #输出128*16*16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2), #输出128*8*8
            nn.Conv2d(128, 256, 3, 1, 1), #输出256*8*8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, 3, 1, 1), #输出256*8*8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, 3, 1, 1), #输出256*8*8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2), #输出256*4*4
            nn.Conv2d(256, 512, 3, 1, 1), #输出512*4*4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, 3, 1, 1), #输出512*4*4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, 3, 1, 1), #输出512*4*4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2), #输出512*2*2
            nn.Conv2d(512, 512, 3, 1, 1), #输出512*2*2
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, 3, 1, 1), #输出512*2*2
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, 3, 1, 1), #输出512*2*2
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2), #输出512*1*1
            nn.Flatten(),
            nn.Linear(512, 10) #这里简化了原模型，如果想试一下原模型的话可以看着图修改
        )

    def forward(self, input):
        return self.model(input)
    
if __name__ == "__main__":
    """
    用来测试一下模型输出格式
    """
    input = torch.ones(64, 3, 32, 32)
    VGG_model = VGG()
    output = VGG_model(input)
    print(output.shape) #torch.Size([64, 10])