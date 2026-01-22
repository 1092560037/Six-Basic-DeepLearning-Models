import torch
from torch import nn

class MyLeNet5(nn.Module):
    #初始化网络
    def __init__(self, num_classes=5, in_channels=3, input_size=64):
        super().__init__()
        self.act = nn.ReLU()

        #设置网络层
        self.c1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.s2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.MaxPool2d(kernel_size=2, stride=2)                
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5) 

        self.flatten = nn.Flatten()

        # 自动计算 flatten 维度：120*10*10 = 12000
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            feat = self._forward_features(dummy)
            self.flat_dim = feat.view(1, -1).shape[1]

        self.f6 = nn.Linear(self.flat_dim, 84)
        self.output = nn.Linear(84, num_classes)

    def _forward_features(self, x):
        x = self.act(self.c1(x))
        x = self.s2(x)
        x = self.act(self.c3(x))
        x = self.s4(x)
        x = self.c5(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.flatten(x)
        x = self.f6(x)
        x = self.output(x)
        return x

if __name__ == "__main__":
    x = torch.rand([1, 3, 64, 64])
    model = MyLeNet5(num_classes=5, in_channels=3, input_size=64)
    y = model(x)
    print("flat_dim =", model.flat_dim)  
    print("y.shape =", y.shape)          




