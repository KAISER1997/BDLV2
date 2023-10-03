'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.relu=torch.nn.ReLU()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        x = self.linear(out)
        x=(x-x.mean())/(x.std())
        return x,out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

class ENCODER(torch.nn.Module):

    def __init__(self):
        super(ENCODER, self).__init__()

        self.linear1 = torch.nn.Linear(2,128)
        self.linear_mid=torch.nn.Linear(128,64)
        self.linear2= torch.nn.Linear(64,32)
        self.relu=torch.nn.ReLU()
        self.bn1=torch.nn.BatchNorm1d(12)
        self.embed=nn.Embedding(20, 128)


    def forward(self, y): #BX4
        # y[:,-1:]=y[:,-1:]+10
        # z=y[:,-2:].long()
        # z=self.embed(z) #B X 2 X 128
        # v=torch.cat([z[:,0,:],z[:,1,:]],1)
        # print("HELLO",v.shape)
        
        # x = self.linear2(self.relu(self.linear_mid(self.relu(self.linear1(v))))) # HACK  USED CORRECT THIS

        x = self.linear2(self.relu(self.linear_mid(self.relu(self.linear1(y[:,-2:]))))) #Older version HACK  USED CORRECT THIS
        return x

class REJECTOR(torch.nn.Module):

    def __init__(self,extra_res=False):
        super(REJECTOR, self).__init__()

        self.linear1 = torch.nn.Linear(64,32)
        self.linear2= torch.nn.Linear(32,16)
        self.linear3=torch.nn.Linear(16,1)
        self.relu=torch.nn.ReLU()
        self.extra0=torch.nn.Linear(512,128)
        # self.extra1=torch.nn.Linear(256,128)
        self.extra2=torch.nn.Linear(128,32)
        # self.extra3=torch.nn.Linear(64,10)
        self.bn1=torch.nn.BatchNorm1d(512)
        self.res=ResNet18()
        self.res.load_state_dict(torch.load('orares.pt'))
        self.extra_res=extra_res
    def forward(self, x,context):#BX10 ,BX2
        if self.extra_res:
            x=self.res(x)[1]
        x=torch.cat([self.relu(self.extra2(self.relu(self.relu(self.extra0(x))))),self.relu(context)],1)
        x = self.linear3(self.relu(self.linear2(self.relu(self.linear1(x)))))
        x=(x-x.mean())/(x.std())

        return x
        # return x

    # test()