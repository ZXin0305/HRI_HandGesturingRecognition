import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import pandas as pd
import numpy
from torch.utils.data import DataLoader,Dataset,random_split,Subset
from torchsummary import summary
from IPython import embed
import os

# action_label = {'kick':0,'punch':1,'squat':2,'stand':3,'wave':4,
#                 'youchayao':5
# }

#训练次数
epchos = 100
batch_size = 512

#路径
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

#分割数据集
Shape = [10334,37]
dataset_len = Shape[0]
train_size = int(0.8*dataset_len)
test_size = int(0.2*dataset_len)
print(train_size, test_size)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.block = nn.Sequential(
                nn.Linear(36, 300),
                nn.ReLU(),
                nn.Linear(300, 200),
                # nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(200, 100),
                nn.ReLU(),
                nn.Linear(100, 50),
                # nn.Dropout(0.9),
                nn.ReLU(),
                # nn.Linear(200, 150),
                nn.Linear(50, 5)
        )
        self.fc = nn.Linear(5, 36)
        self.ac1 = nn.ReLU()
        self.ac2 = nn.Sigmoid()
        self.drop = nn.Dropout(0.5)
        self.soft = nn.Softmax()

    def forward(self,x):
        out = self.block(x)
        out = self.fc(out)
        out = self.block(out)
        # out = self.soft(out)　　#loss　function为交叉熵时，不用再使用这个softmax层了
        return out

# net = Net().to('cuda')
# summary(net,(1,36))
# embed()
# loss = nn.CrossEntropyLoss()
# net = Net()
# optimizer  = optim.SGD(net.parameters(),lr=0.001,momentum=0.5)

class Data(Dataset):
    def __init__(self,data,flag='train'):
        # super(Dataset,self).__init__()
        self.data = data
        self.data_info = self.getInfo(self.data,flag)


    @staticmethod
    def getInfo(data,flag):
        if flag == 'train':
            data_info = [[] for i in range(train_size)]   #创建二维数组
            size = train_size
        else:
            data_info = [[] for i in range(test_size)]
            size = test_size
        # print(len(data_info))
        # print(data[0][0:37])
        for idx in range(size):
            data_info[idx] = (data[idx][0:36],int(data[idx][36]))   # pose && label
        return data_info


    def __getitem__(self, index):
        coordi,lable = self.data_info[index]
        return coordi,lable

    def __len__(self):
        return len(self.data_info)



if __name__ == '__main__':
    CURRENT_PATH = os.path.dirname(os.path.abspath(__file__)) + '/'

    key_word = '--dataset'
    paser = argparse.ArgumentParser()
    paser.add_argument(key_word,required=False,help = 'display the dataset',default='/home/xuchengjun/ZXin/smap/action/train_data.csv')

    opt = paser.parse_args().dataset

    #加载数据集(关节点的数据集)
    try:
        raw_data = pd.read_csv(opt,header=0)
    except:
        print("dataset not exist")

    dataset = raw_data.values
    # print(dataset)

    # X = dataset[:,0:36].astype(float)
    # Y = dataset[:,36]
    # print(Y)

    #将动作标签进行转换
    # for i in range(len(Y)):
    #     # # print(key)
    #     if Y[i] in action_label.keys():
    #         Y[i] = action_label[Y[i]]
    # print(Y.astype)   #numpy
    # Y = Y.astype(float)  #强制转换类型，满足torch支持的类型

    #对数据进行优化
    # X_pp = []
    # total = []
    # for i in range(len(X)):
    #     X_pp.append(dpp.pose_normalization(X[i]))
    # total = X_pp

    # total = torch.Tensor(X_pp)
    # print(X_pp.shape)
    # Y_pp = torch.from_numpy(Y)

    # for i in range(len(X_pp)):
    #     total[i].append(Y[i])
    # # print(total)
    # total = torch.Tensor(total)    #torch.Size([4416, 27])
    # Shape = total.shape            #得到shape

    total = torch.Tensor(dataset)
    train_set = Subset(total,range(train_size))  #训练集
    test_set = Subset(total,range(train_size,dataset_len)) #测试集
    #数据处理完成＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

    dataset_1 = Data(train_set,'train')
    train_loader = DataLoader(dataset=dataset_1,batch_size = batch_size,shuffle = True)
    # print(len(dataset_1))/
    loss_C = nn.CrossEntropyLoss()
    net = Net()

    # for name, parameters in net.named_parameters():
    #     print(name, ':', parameters.size())

    optimizer = optim.Adagrad(net.parameters(), lr=0.003)    #调整一下学习率
    #训练
    loss = 0
    total_loss = 0
    correct_num = 0
    for epcho in range(epchos):
        epo_loss = 0   #一个epoch中的loss
        for zips in train_loader:
            train_data = zips[0]
            # print(len(train_data))
            label = zips[1]
            optimizer.zero_grad()
            output = net(train_data)
            max_label = output.argmax(dim = 1)
            # print("max_label:",max_label.data)
            correct_num += max_label.eq(label.data).float().sum().item()
            # print(correct_num)
            # print("label.data",label.data)
            loss = loss_C(output,label)
            total_loss += loss
            epo_loss += loss
            # print(loss)
            loss.backward()
            optimizer.step()


        if epcho % 10 == 0 and epcho != 0:
            # print(output.shape)
            # print(label.shape)
            print("train_epcho: {} -- loss:{:0.3f} -- total_loss:{:0.3f}".format(epcho,epo_loss,total_loss))
            # print("train_epcho: {}   Avg_loss: {:.6f}".format(epcho,total_loss / batch_size))
            print('===============================================================')

    #
    # print("total_loss: {:.6f}".format(total_loss))
    # print('==========================')
    # print("avgLoss: {:.6f}".format(total_loss / len(dataset_1)))
    # print(correct_num)
    print("Acc: {:0.2f}%".format((correct_num / (train_size * epchos) * 100)))

    state_dict = net.state_dict()
    # print(state_dict['block.0.bias'])
    torch.save(state_dict,'../model/action.pkl')

    ####test
    # state_dict_test = torch.load('../model/action.pkl')
    # net = Net()
    # net.load_state_dict(state_dict_test)

    # net.eval()
    # dataset_2 = Data(test_set, 'test')   #制作数据
    # test_loader = DataLoader(dataset=dataset_2,batch_size=64,shuffle=False) #取出数据
    # #begin to predict
    # correct_num = 0
    # for zips in test_loader:
    #     test_data = zips[0]
    #     # print(zips[0])
    #     test_label = zips[1]
    #     test_output = net(test_data)
    #     max_label = test_output.argmax(dim=1)
    #     # print(test_output)
    #     correct_num += max_label.eq(test_label.data).float().sum().item()

    # print("the test-acc is {:0.2f}%".format((correct_num / test_size  * 100)))

