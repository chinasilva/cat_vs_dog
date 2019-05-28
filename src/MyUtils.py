import torch

class MyUtils():
    def make_one_hot(self,labels, C=2):
        '''
        Converts an integer label torch.autograd.Variable to a one-hot Variable.
        
        Parameters
        ----------
        labels : torch.autograd.Variable of torch.cuda.LongTensor
            N x 1 x H x W, where N is batch size. 
            Each value is an integer representing correct classification.
        C : integer. 
            number of classes in labels.
        
        Returns
        -------
        target : torch.autograd.Variable of torch.cuda.FloatTensor
            N x C x H x W, where C is class number. One-hot encoded.
        '''
        # one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
        # target = one_hot.scatter_(1, labels.data, 1)
        labels=labels.long()
        one_hot= torch.zeros(labels.size()[0], C)
        target=one_hot.scatter_(1, labels.view(-1,1), 1)
        # print("labels:",labels)
        # print("labels.view:",labels.view(-1,1))
        # target = Variable(target)
            
        return target

    def deviceFun(self):
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        return device