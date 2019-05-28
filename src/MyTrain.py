# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np

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
        self.device=self.myUtils.deviceFun()
        self.myNet=self.myNet.to(self.device)
        self.optimizer=torch.optim.Adam(self.myNet.parameters())
        self.lossFun=nn.MSELoss()
        self.trainData=data.DataLoader(self.myData,batch_size=batchSize,shuffle=True)
        # data.dataloader(self.myData)

    def train(self):
        losslst=[]
        for i in range(self.epoch):
            print("epoch:",i)
            for j,(x,y) in enumerate(self.trainData):
                x=x.view(-1,100*100*3).to(self.device)
                output=self.myNet(x)

                y=self.myUtils.make_one_hot(y,2).to(self.device)

                loss=self.lossFun(y,output)
                losslst.append(loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # print("loss:",loss.data)

                if j%10==0:
                    print("loss:",loss.item())
                    y=torch.argmax(y,dim=1)
                    output=torch.argmax(output,dim=1)
                    acc= np.mean(np.array(y.cpu()==output.cpu()),dtype=np.float)
                    print("acc:",acc)


        
        save_model = torch.jit.trace(self.myNet,  torch.rand(self.batchSize, 3*100*100).to(self.device))
        save_model.save(r"model/net.pth")


        # 保存加载模型所有信息
        # torch.save(self.myNet, r'model/model.pth')  
        # model = torch.load(r'model/model.pth')

        # # 保存加载模型参数信息
        # torch.save(self.myNet.state_dict(), r'model/params.pth')  
        # model_object.load_state_dict(torch.load(r'model/params.pth'))

if __name__ == "__main__":
    path="img"
    epoch=1
    batchSize=600
    myTrain=MyTrain(path,epoch,batchSize)
    myTrain.train()