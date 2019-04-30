import torchvision
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image

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

#初始化，通过模型参数加载CNN模型
cnn=Cnn()
cnn.load_state_dict(torch.load('F:\pycharm\初级训练\CNN\CIFAR-1.pth'))

#格式转换为（3，32，32），4通道通过numpy减取一个通道
transform=transforms.Compose(
    [transforms.Resize((32,32)),#Image.resize((32,32))也可以
    transforms.ToTensor(),              #标准化？？看文档吧
    transforms.Normalize((0.5,0.5,0.5),std=(0.5,0.5,0.5))
])
textset=torchvision.datasets.CIFAR10(root='./CIFAR',train=False,
                                     download=True,transform=transform)
                        #并行大小 batch=5
textloader=DataLoader(dataset=textset,batch_size=5,shuffle=True,num_workers=4)


if __name__=='__main__':
    error_sum=0
    for i,data in enumerate(textloader):    #enumerate产生迭代索引和对象
        x,y=data
        pre_y=cnn(x)
        np_prey=torch.argmax(pre_y,1).data.squeeze().numpy()    #通过numpy进行
        np_y=y.numpy()                                          #错误统计
        error_sum+=sum(np_prey!=np_y)
        if (i+1)%500==0:
            print('error-rate : '+str(error_sum/(i*5+1)))

    image_dog = Image.open('F:\pycharm\初级训练\CNN\dog1.jpg')
    tsfm_image = transform(image_dog)
    tsfm_image=tsfm_image.unsqueeze(0)          #CNN输入对象为4D，通过unsqueeze(0)增加一个维度
    out=cnn(tsfm_image)
    pre=torch.argmax(out,1)             #将10维的输出argmax为分类标签
    print(pre)




