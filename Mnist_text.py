import torch.nn as nn
import torch.tensor
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import torchvision
import torch.optim as optim

transform=transforms.Compose(
    [transforms.ToTensor()
    ])


textset=torchvision.datasets.MNIST(root='./data',train=False,
                                         transform=transform,download=True
                                         )
textloader=data.DataLoader(dataset=textset,batch_size=2,shuffle=True,num_workers=2)
class Cnn(nn.Module):
    def __init__(self):
        super(Cnn,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

        )
        self.out=nn.Linear(in_features=32*7*7,out_features=10)
    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0),-1)
        output=self.out(x)

        return output

cnn=Cnn()
cnn.load_state_dict(torch.load('F:\pycharm\初级训练\CNN\dict.pth'))
if __name__=='__main__':
    error_num=0
    for i,data in enumerate(textset):
        x,y=data
        out=cnn(x.unsqueeze(0))
        pre_y=torch.argmax(out,1)[0].data.squeeze()
        if pre_y!=y:
            error_num=error_num+1
        if (i+1) %1000==0:
            print(error_num/1000)

            error_num=0

