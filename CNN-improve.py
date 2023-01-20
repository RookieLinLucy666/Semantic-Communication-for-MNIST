import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os
import time
from color_mnist import ColoredMNIST
from torchvision import datasets, transforms
from scipy.spatial.distance import pdist, squareform
import numpy as np


'''
Created on Wen Nov 30 14:40:09 2022
@author: StarryHuang

详细注释版
语义通信在MNIST手写数据集上的识别-AWGN信道
使用CNN+linear分类，网络结构是semantic encoder-> channel encoder-> channel-> channel decoder ->semantic decoder
将每一张图片压缩到16个特征，SNR=20dB
'''
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
torch.manual_seed(1)  # 使用随机化种子使神经网络的初始化每次都相同

# 超参数
EPOCH = 15  # 训练整批数据的次数
BATCH_SIZE = 64
LR = 0.001  # 学习率
M=4
DOWNLOAD_MNIST = True  # 表示还没有下载数据集，如果数据集下载好了就写False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 下载mnist，【训练集】
# train_data = torchvision.datasets.MNIST(
#     root='./mnist/',  # 保存或提取的位置  会放在当前文件夹中
#     train=True,  # true说明是用于训练的数据，false说明是用于测试的数据
#     transform=torchvision.transforms.ToTensor(),  # 转换PIL.Image or numpy.ndarray
#     # 此处ToTensor()将shape为(H, W, C)的nump.ndarray或img转为shape为(C, H, W)的tensor，其将每一个数值归一化到[0,1]，其归一化方法比较简单，直接除以255即可
#     download=DOWNLOAD_MNIST,
# )
#
# # 加载Mnist训练集, Torch中的DataLoader是用来打乱、分配、预处理
# train_loader = Data.DataLoader(
#     dataset=train_data,
#     batch_size=BATCH_SIZE,
#     shuffle=True,  # 是否打乱数据，一般都打乱，，常用于进行多批次的模型训练
#     drop_last=False  # 设置为True表示当数据集size不能整除batch_size时，则删除最后一个batch_size，否则就不删除
# )

train_loader = torch.utils.data.DataLoader(
    ColoredMNIST(root='./data', env='all_train',
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                 ])),
    batch_size=BATCH_SIZE, shuffle=False,)

# # 下载【测试集】并加载，返回值为一个二元组（data，target）
# test_data = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=torchvision.transforms.ToTensor(),download=DOWNLOAD_MNIST,)
# test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

test_loader = torch.utils.data.DataLoader(
    ColoredMNIST(root='./data', env='test',
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                 ])),
    batch_size=BATCH_SIZE, shuffle=False,)

def pairwise_distances(x):
    #x should be two dimensional
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

def calculate_gram_mat(x, sigma):
    dist = pairwise_distances(x)
    #dist = dist/torch.max(dist)
    return torch.exp(-dist /sigma)

def renyi_entropy(x,sigma):
    alpha = 1.01
    x = x.view(64,-1)
    k = calculate_gram_mat(x,sigma)
    k = k/torch.trace(k)
    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eig_pow = eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))
    return entropy

class RenyiEntropyLoss(nn.Module):
    def __init__(self):
        super(RenyiEntropyLoss, self).__init__()

    def forward(self, x):
        with torch.no_grad():
            x_numpy = x.cpu().detach().numpy()
            k = squareform(pdist(x_numpy, 'euclidean'))       # Calculate Euclidiean distance between all samples.
            sigma = np.mean(np.mean(np.sort(k[:, :10], 1)))
        H = renyi_entropy(x,sigma=sigma**2)
        return H

class MyGate(nn.Module):
    r"""Applies a linear gating transformation to the incoming data: :math:`y = x * sigmoid(w) `
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, out_features: int, use_tanh: bool = True) -> None:
        super(MyGate, self).__init__()
        self.in_features = out_features
        self.out_features = out_features
        self.weight = nn.parameter.Parameter(torch.Tensor(out_features))
        self.mask = nn.parameter.Parameter(torch.ones_like(self.weight))
        self.mask.requires_grad = False
        if use_tanh:
            self.gate_fn = torch.tanh
        else:
            self.gate_fn = torch.sigmoid
        self.reset_parameters()

        def hook_fn(module, input, output):
            if False:
                print(f"input={(input[-1]>0.5).long().sum(axis=1)}")
            if type(output) in [tuple,list]:
                print(f"output{output[0].shape[0]}={(output[0]!=0).long().sum(axis=1)}")
            else:
                print(f"output{output.shape[0]}={(output>0.5).long().sum(axis=1)}")

        self.hook_fn = hook_fn
        if False:
            self.register_backward_hook(self.hook_fn)
            self.register_forward_hook(self.hook_fn)

    def reset_parameters(self) -> None:
        #         init.uniform_(self.weight, -10, 10)
        #         init.uniform_(self.weight, .1, 2)
        nn.init.constant_(self.weight, 10.)


    def update_mask(self):
        def get_mask_(a,m,th=2,th_a=.33):
            b = 1 - (-th_a < a ).float()*(a < th_a).float()
            if sum(b*m)<th:
                return m
            else:
                return b*m
        values = self.gate_fn(self.weight).detach()
        self.mask *=  get_mask_(values,self.mask)
        return self.mask

    def get_mask(self):
        return self.mask

    def get_gates(self):
        return self.gate_fn(self.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.mask * self.gate_fn(self.weight) * input

    def get_loss(self):
        return 1e0*torch.sqrt(((torch.abs(self.gate_fn(self.weight)) - torch.ones_like(self.weight))**2).sum())

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

# # 训练中进行测试
# test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255
# # torch.unsqueeze(a) 是用来对数据维度进行扩充，这样shape就从(2000,28,28)->(2000,1,28,28)
# # 图像的pixel本来是0到255之间，除以255对图像进行归一化使取值范围在(0,1)
# test_y = test_data.targets[:2000]

# 用class类来建立CNN模型
class CNN(nn.Module):  # 我们建立的CNN继承nn.Module这个模块
    def __init__(self):
        super(CNN, self).__init__()
        # semantic encoding
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.gate3 = MyGate(7)
        self.SE = nn.Sequential(self.layer1, self.layer2, self.gate3)

        # self.SE = nn.Sequential(
        #     # 卷积con2d参数设置：
        #     # in_channels=1,  # 输入图片通道数，因为minist数据集是灰度图像只有一个通道
        #     # out_channels=8, # 通道数，
        #     # kernel_size=3,  # 卷积核的大小
        #     # stride=1,  # 步长
        #     # padding=1,  # 想要con2d输出的图片长宽不变，就进行补零操作 padding = (kernel_size-1)/2
        #     nn.Conv2d(1, 8, 3, 1, 1),  # 输出图像大小(8,28,28)
        #     nn.BatchNorm2d(8), # 对所有batch的同一个channel上的数据进行归一化
        #     nn.ReLU(),# 激活函数，非线性操作
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 池化，[(H+2*p-k)/s +1]向下取整，# 输出图像大小(8,14,14)
        #     # nn.MaxPool2d(kernel_size=(14,28),stride=(14,14),padding=0),  # 高和宽也可以不同，根据需求来设定
        #     # nn.AvgPool2d(kernel_size=(5,7), stride=(3, 7), padding=0),  # 14-42 比较一下
        #     nn.Conv2d(8, 8, 3, 1, 1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 输出图像大小(8,7,7)
        # )

        # channel encoding，M是信道编码后每个像素的symbol数，可变
        self.CE = nn.Sequential(
            nn.Conv2d(8, M, 3, 1, 1),
            nn.BatchNorm2d(M),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=2)  # 最后提取出来的特征比较重要，用Average
        )  # 输入图像大小(M,2,2)，特征一共 M* 2 * 2个

        # 加噪，AWGN==============

        # channel decoding
        self.CD = nn.Sequential(
            nn.Conv2d(M, M, 3, 1, 1),
            nn.BatchNorm2d(M),
            nn.ReLU())

        # semantic decoding，建立全卷积连接层分类
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),  # 有一定概率失活，使模型具有一定的泛化性
            nn.Linear(16, 10),  # 输出是10个类。第一个参数要注意：要和上面的输出个数一致 M * h * w
            nn.Softmax()  # 柔性函数，总概率加起来为1
        )

    # 下面定义x的传播路线
    def forward(self, x):
        x = self.SE(x)
        encoder = self.CE(x)
        x = self.CE(x)
        x = AWGN_channel(x, 20)  # SNR可以修改
        x = self.CD(x)
        # print('x:', x[0, 0, :, :].size()) # 可以检查网络的输出当前x的height， weight情况。x是四维[batch，通道，height， weight]

        # 把每一个批次的每一个输入都拉成一个维度
        # 因为pytorch里特征的形式是[bs,channel,h,w]，所以x.size(0)就是batchsize
        x = x.view(x.size(0), -1)  # view就是把x弄成batch行个tensor,一行为一图片,-1表示一个不确定的数,网络自己算数，这里是=channel*H*C
        output = self.classifier(x)
        return encoder,output

    def update_mask(self):
        self.gate3.update_mask()

# add noise
def AWGN_channel(x, snr):  # used to simulate additive white gaussian noise channel
    [batch_size, channel, length, len_feature] = x.shape
    # torch.sum(torch.square(x))
    x_power = torch.sum(torch.square(x)) / (batch_size * length * len_feature * channel)
    n_power = x_power / (10 ** (snr / 10.0))
    # print('n_power',n_power)
    # print('n_power', n_power.numpy()**(0.5))
    noise = torch.normal(mean=0, std=n_power.detach().numpy()**(0.5), size=[batch_size, channel, length, len_feature])
    # print('noise',noise[1:])
    # print('noise', x[1:])
    return x + noise

def train():
    for epoch in range(EPOCH):
        for batch_idx, (inputs, targets) in enumerate(train_loader):  # 分配batch data
            Z, output = cnn(inputs)  # 先将数据放到cnn中计算output
            loss = loss_func(output, targets)  # 输出和真实标签的loss，二者位置不可颠倒
            with torch.no_grad():
                # print(Z.view(64,-1))
                Z_numpy = Z.view(64,-1)
                k = squareform(pdist(Z_numpy, 'euclidean'))       # Calculate Euclidiean distance between all samples.
                sigma = np.mean(np.mean(np.sort(k[:, :10], 1)))

            IXZ = renyi_entropy(inputs, sigma**2)
            IZY = renyi_entropy(Z, sigma**2)
            cross_entropy_loss = loss_func(output, targets)
            beta = 1e-6
            loss = cross_entropy_loss+beta*IXZ

            optimizer.zero_grad()  # 清除之前学到的梯度的参数
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 应用梯度
            if batch_idx % 100 == 0:  # 100个batch后测试一下
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())
                train_losses.append(loss.item())  # loss.item() = loss.data.numpy() 等价
                train_counter.append((batch_idx * 64) + (epoch * len(train_loader.dataset))) # len(train_loader.dataset)=60000

    # torch.save(cnn, model_path)  # 保存整个模型
    # torch.save(model.state_dict(), "my_model.pth")  # 只保存模型的参数

def test():
    test_ave_acc = []
    # net = torch.load(model_path)
    # print('model:',net)
    cnn.eval()  # 设置模型进入预测模式 evaluation
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):  # 分配batch data
            _,test_output = cnn(inputs)
            # TODO:
            # print(test_output.size())
            pred_y = torch.max(test_output, 1)[1]  # torch.max(a,1) 返回每一行中最大值的那个元素，且返回其索引。
            # data.numpy()使得tensor变成numpy形式， targets.size(0)= batchsize
            accuracy = float((pred_y.data.numpy() == targets.data.numpy()).astype(int).sum()) / float(targets.size(0))
            test_ave_acc.append(accuracy)
            # print('test accuracy: %.4f' % accuracy)
    print('Test ave acc:%.2f' % (100. * sum(test_ave_acc)/len(test_ave_acc)))
    # print('len(test_ave_acc):',len(test_ave_acc))



cnn = CNN() # print(cnn) # 可以查看网络状态
print('Total params: %.2fw' % (sum(p.numel() for p in cnn.parameters())/10000.0)) # 参数总数
# 优化器选择Adam，lr要比较小。SDG的lr会比较大。
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
# 损失函数
loss_func = nn.CrossEntropyLoss()  # =softmax得到（Y^）->log（Y^）->交叉熵公式（-Ylog（Y^）），目标标签是one-hotted

train_losses = []
train_counter = []
test_losses = []
test_counter = []
model_path = './model_AWGN_test——standard.pth'


if __name__ == '__main__':
    # from torch.optim import lr_scheduler  # 学习率调整器，在训练过程中合理变动学习率
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

    start = time.time()
    # train
    train()
    print('training has done！')

    # test
    test()
    print("Time: " + str((time.time() - start)/60) + 'mins')
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.xlabel('number of training examples')
    plt.ylabel('loss')
    plt.show()
