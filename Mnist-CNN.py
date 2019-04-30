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
trainset=torchvision.datasets.MNIST(root='./data',train=True,
                                          transform=transform,
                                            download=True)
textset=torchvision.datasets.MNIST(root='./data',train=False,
                                         transform=transform,download=True
                                         )

trainloader=data.DataLoader(dataset=trainset,batch_size=4,shuffle=True,num_workers=2)
textloader=data.DataLoader(dataset=textset,batch_size=4,shuffle=True,num_workers=2)


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
optimzer=optim.Adam(cnn.parameters(),lr=0.001)
criterion=nn.CrossEntropyLoss()
if __name__=='__main__':
    for epoch in range(1):
        run_loss=0.0
        for i,(input,label) in enumerate(trainloader):
            optimzer.zero_grad()
            output=cnn(input)
            loss=criterion(output,label)
            loss.backward()
            optimzer.step()
        torch.save(cnn.state_dict(),'dict.pth')




