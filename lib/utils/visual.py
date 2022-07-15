import matplotlib.pyplot as plt
from pylab import *     
import numpy as np                            #支持中文
import pandas as pd
mpl.rcParams['font.sans-serif'] = ['SimHei']
from IPython import embed

# names = ['1', '2', '3', '5', '8', '10']
# x = range(len(names))
# y = [40,41.4,43.5,44.3,49.8, 52.0]
# y1=[0.065,0.14,0.25,0.62,1.5,2.6]
#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
#pl.xlim(-1, 11)  # 限定横轴的范围
#pl.ylim(-1, 110)  # 限定纵轴的范围
# plt.plot(x, y, marker='o', mec='r', color='r',label=u'pose estimation')
# plt.plot(x, y1, marker='*',ms=10,label=u'pose tracking')
# plt.legend()  # 让图例生效
# plt.xticks(x, names, rotation=45)
# plt.margins(0)
# plt.subplots_adjust(bottom=0.15)
# plt.xlabel(u"Number of People Per Image") #X轴标签
# plt.ylabel("Running Time(unit:ms)") #Y轴标签
# plt.title("") #标题

# plt.show()

# import numpy as np
# from scipy import stats
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# iris_data = load_iris()
# sample_1 = iris_data.data[0,:] # 取出第1行的所有数据
# print(sample_1)
# # 绘制条开图
# p1 = plt.bar(range(1, len(sample_1) + 1),
#              height = sample_1,
#              tick_label = iris_data.feature_names,
#              width = 0.3)
# plt.ylabel('cm')
# plt.title('plt of_first data')
# plt.show()


df=pd.read_csv('/home/xuchengjun/ZXin/smap/human.csv')
# data = np.array(df[['root','lelbow','lwrist','relbow','rwrist','robot_x','robot_y','robot_z']])
data = np.array(df[['root','lelbow','lwrist','relbow','rwrist']])
root = []
lelbow = []
lwrist = []
relbow = []
rwrist = []
time = []
robot_x = []
robot_y = []
robot_z = []
for i in range(data.shape[0]):
    root.append(data[i][0])
    lelbow.append(data[i][1])
    lwrist.append(data[i][2])
    relbow.append(data[i][3])
    rwrist.append(data[i][4])

    time.append(i * 0.001)

plt.figure()
plt.scatter(time,root,s=2,c='red', label = 'root')
plt.scatter(time,rwrist,s=2,c='green', label = 'rwrist')
plt.xlabel('meter')    
plt.ylabel('time')
plt.title('position') 
plt.show()

# df=pd.read_csv('/home/xuchengjun/ZXin/smap/robot.csv')
# data = np.array(df[['robot_x','robot_y','robot_z']])
# robot_x = []
# robot_y = []
# time = []
# for i in range(data.shape[0]):
#     robot_x.append(data[i][0])
#     robot_y.append(data[i][1])
#     time.append(i * 0.02)

# plt.figure()
# plt.scatter(time,robot_x,s=2,c='red', label = 'root_x')
# plt.scatter(time,robot_y,s=2,c='green', label = 'rwrist_y')
# plt.xlabel('time')    
# plt.ylabel('meter')
# plt.title('position') 
# plt.show()

if __name__ == "__main__":
    from tensorboardX import SummaryWriter

    