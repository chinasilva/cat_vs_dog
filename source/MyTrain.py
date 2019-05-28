import torch
import torch.nn as nn
from torch.utils import data

from MyData import MyData
from MyNet import MyNet
from MyUtils import MyUtils

class MyTrain():
    def __init__(self,path,epoch,batchSize):
        self.path=path
        self.epoch=epoch
        self.batchSize=batchSize
        self.myNet=MyNet()
        self.myData=MyData(self.path)
        self.myUtils=MyUtils()
        self.optimizer=torch.optim.Adam(self.myNet.parameters())
        self.lossFun=nn.MSELoss()
        self.trainData=data.DataLoader(self.myData,batch_size=batchSize,shuffle=True)
        # data.dataloader(self.myData)

    def train(self):
        losslst=[]
        for i in range(self.epoch):
            print("epoch:",i)
            for j,(x,y) in enumerate(self.trainData):
                x=x.view(-1,100*100*3)
                output=self.myNet(x)
                #onehot编码
                ...
                y=self.myUtils.make_one_hot(y,2)

                loss=self.lossFun(y,output)
                losslst.append(loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # print("loss:",loss.data)

                if j%10==0:
                    print("loss:",loss.data)

if __name__ == "__main__":
    path="img"
    epoch=100
    batchSize=60
    myTrain=MyTrain(path,epoch,batchSize)
    myTrain.train()