import torch
import torchvision
import torch.nn.functional as F
import numpy as np
from PIL import Image
from MyUtils import MyUtils

myUtils=MyUtils()
device=myUtils.deviceFun() 


def core(net, img_path):

    net.eval()

    img = Image.open(img_path)
    img= torch.Tensor(np.array(img)/255 - 0.5).unsqueeze(0) * -1
    img=img.to(device)

    output = net(img)
    output=torch.argmax(output,dim=1)
    
    return output


if __name__ == "__main__":
    model = torch.load(r'model/model.pth')
    # net = torch.jit.load("models/net.pth")
    result_data = core(model, "img/0.1.jpeg")
    print(result_data)

