import torch.nn as nn
import torch

# 卷积池化层提取图像特征，全连接层进行图像分类，代码中写成两个模块，方便调用
# pytorch 中 Tensor 参数的顺序为 (batch, channel, height, width)
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        #用nn.Sequential()将网络打包成一个模块，精简代码
        #更精确的写法可以用nn.ZeroPad2d((1,2.1,2)) 实现左侧补一列，右侧补两列；上方补一行，下方补两行。
        #但为了方便，统一写成padding = 2 就行，因为在pytorch中如果计算结果为小数，会自动将多余数据舍弃掉（最右侧和最下端的一列/行），转换为整数，结果与精确写法差不多
        # torch中的卷积操作处理 https://blog.csdn.net/qq_37541097/article/details/102926037?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522506441b7acda6db98e44b223f86dc537%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=506441b7acda6db98e44b223f86dc537&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-1-102926037-null-null.nonecase&utm_term=pytorch%E5%8D%B7%E7%A7%AF&spm=1018.2226.3001.4450
        self.features = nn.Sequential( # 卷积层提取图像特征
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True), #inplace：增加计算量，但能够降低内存使用的一种方法
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential( # 全连接层对图像分类
            nn.Dropout(p=0.5), # Dropout 随机失活神经元，默认比例为0.5
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        #初始化权重
        if init_weights:
            self._initialize_weights()

    # 前向传播过程
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    # 网络权重初始化，实际上 pytorch 在构建网络时会自动初始化权重
    def _initialize_weights(self):
        for m in self.modules():
            #判断层的类型：卷积层 or 全连接层
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # 用（何）kaiming_normal_法初始化权重
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0) # 初始化偏重为0
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01) # 正态分布初始化
                nn.init.constant_(m.bias, 0)
