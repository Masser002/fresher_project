import torchvision
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform=transforms.Compose(
    [transforms.ToTensor(),             #标准化。。。。？？
    transforms.Normalize((0.5,0.5,0.5),std=(0.5,0.5,0.5))
])
trainset=torchvision.datasets.CIFAR10(root='./CIFAR',train=True,
                                      download=True,transform=transform)
textset=torchvision.datasets.CIFAR10(root='./CIFAR',train=False,
                                     download=True,transform=transform)
trainloader=DataLoader(dataset=trainset,batch_size=5,shuffle=True,num_workers=2)

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=10,kernel_size=3,
                        stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=10,out_channels=20,kernel_size=3,
                      stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(20,30,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(30,40,3,1,1),
            nn.ReLU(),
        )
        self.fc1=nn.Linear(in_features=40*4*4,out_features=1000)
        self.fc2=nn.Linear(in_features=1000,out_features=64)
        self.fc3=nn.Linear(in_features=64,out_features=10)
    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)

        x=x.view(x.size(0),-1)      #展开为1维向量x.size(0)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x



cnn=Cnn()      #实例化cnn，优化器，损失函数
optimzer=optim.SGD(cnn.parameters(),lr=0.001,momentum=0.9)
crizerion=nn.CrossEntropyLoss()

#用'__main__'包代码进行多进程读取enumerate(trainloader)
if __name__=='__main__':
    for epoch in range(10):         #训练轮数epoch
        run_loss=0                  #初始化损失
        for i ,data in enumerate(trainloader):
            optimzer.zero_grad()    #优化器参数重置为0
            x,y=data
            out_y=cnn(x)
            loss=crizerion(out_y,y) #计算损失
            loss.backward()         #对损失求导而后更新模型参数
            optimzer.step()
            run_loss+=loss.item()
            if i %2000==0:
                print('Epoch '+str(epoch)+'  run_loss '+str(run_loss/2000))
                run_loss=0

    # 在'__main__'里不支持模型读取
    #通过模型参数保存模型  .state_dict()

    torch.save(cnn.state_dict(),'CIFAR-1.pth')
    print('Module Save ')



