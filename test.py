import numpy as np
from IPython import embed
from time import time
import torch
import math
import random
import cv2
import ctypes
import os
import yaml
# from pykinect2 import PyKinectRuntime
# from pykinect2 import PyKinectV2
from lib.utils.tools import *

# xx = np.zeros(shape=(2,15,4), dtype=np.float)

# xx[0, 2, 2] = 1
# xx[1, 2, 2] = 0.5

# yy = xx[:,2,2].argsort()
# xx = xx[yy]
# embed()

# xx = np.ones(shape=(75,45))
# xx = xx.tolist()
# # xx.pop(0,20)
# del xx[0:20]
# embed()

# xx = {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0,'10':0,'11':0,'12':0,'13':0,'14':0,'15':0,'16':0,
#         '17':0,'18':0,'19':0,'20':0,'21':0,'22':0,'23':0,'24':0,'25':0,'26':0,'27':0,'28':0,'29':0,'30':0,'31':0,}

# st = time()
# if "0" in xx.keys():
#     et = time()
#     print(f"total {(et - st)}")

# def change_pose(pred_3d_bodys):
#     """[summary]

#     Args:
#         pred_3d_bodys ([type]): [description]
#         not original 

#     Returns:
#         [type]: [description]
#     """
#     pose_3d = []
#     for i in range(0,1):   # 默认都是1个人
#         for j in range(15):
#             pose_3d.append(pred_3d_bodys[i][j][0])  # x
#             pose_3d.append(pred_3d_bodys[i][j][1])  # y
#             pose_3d.append(pred_3d_bodys[i][j][2])  # z
#     return pose_3d
# xx = np.eye(3)
# yy = np.random.rand(1, 15,3)
# yy =yy.transpose(0,2,1)
# zz = xx @ yy
# zz[0,1] += 1
# zz[0,2] += 1
# embed()

# a = [1,2,3]
# b = [1,2,3]
# c = max(b)
# print(c)

# a = [[1,2,3],[1,2,3]]
# a = np.array(a)

# b = [[1,5,3],[0,0,0]]
# b = np.array(b)

# c = np.array([1,2,3])
# # print(sum(c))
# print(c)
# print(c.argmax(0))

# a = torch.tensor([1,2,3])
# a = 0


# xx = (1 / math.sqrt(2 * math.pi)) * math.exp((-1 / 2) * 0.13)
# xx = math.exp((-1 / 2) * 0.13)
# pri

# xx = random.randrange(30,54)
# print(xx)


# xx = np.array([[1,2,3],[1,2,3]])
# yy = np.delete(xx[:,:],1)
# embed()

# xx = np.array([[ -91.24533081,   -9.77925491,  267.06481934,    1.        ],
#        [ -82.04265594,  -31.73023224,  271.43804932,    1.        ],
#        [ -89.02472687,   40.83181763,  284.30203247,    1.        ],
#        [-104.65914917,  -12.89662933,  276.2901001 ,    1.        ],
#        [-109.66155243,    9.82787323,  289.22158813,    1.        ],
#        [ -85.39533997,    6.52565002,  293.66082764,    1.        ],
#        [ -97.42415619,   39.70267487,  290.0920105 ,    1.        ],
#        [-100.76938629,   71.71859741,  304.46191406,    1.        ],
#        [-110.30347443,  106.38193512,  312.63577271,    1.        ],
#        [ -77.77825165,   -7.06853485,  257.95681763,    1.        ],
#        [ -70.11280823,   17.82071495,  265.47302246,    1.        ],
#        [ -69.68502808,   12.14739037,  288.32354736,    1.        ],
#        [ -80.57032013,   41.87945175,  278.51196289,    1.        ],
#        [ -77.47626495,   75.3965683 ,  291.89306641,    1.        ],
#        [ -78.98562622,  109.38594055,  302.17233276,    1.        ]])

# yy = np.array([[ 195.16946411,  240.30853271,  270.02520752,    2.        ,
#          -97.80656433,   -8.6984005 ,  270.02520752, 1427.33996582,
#         1423.13000488,  949.61798096,  548.13201904],
#        [ 222.62481689,  187.98979187,  273.14260864,    2.        ,
#          -85.97328186,  -32.63896561,  273.14260864, 1427.33996582,
#         1423.13000488,  949.61798096,  548.13201904],
#        [ 213.34616089,  348.8364563 ,  293.92376709,    2.        ,
#          -97.53807831,   44.09518433,  293.92376709, 1427.33996582,
#         1423.13000488,  949.61798096,  548.13201904],
#        [ 173.70863342,  233.23698425,  280.52893066,    2.        ,
#         -112.50686646,  -12.4541378 ,  280.52893066, 1427.33996582,
#         1423.13000488,  949.61798096,  548.13201904],
#        [ 176.00830078,  279.80993652,  292.9100647 ,    2.        ,
#         -116.22911072,   10.05602646,  292.9100647 , 1427.33996582,
#         1423.13000488,  949.61798096,  548.13201904],
#        [ 216.99163818,  286.08230591,  300.94900513,    2.        ,
#          -97.40164948,   13.34723759,  300.94900513, 1427.33996582,
#         1423.13000488,  949.61798096,  548.13201904],
#        [ 197.83638   ,  343.52972412,  299.24221802,    2.        ,
#         -107.50037384,   42.39523315,  299.24221802, 1427.33996582,
#         1423.13000488,  949.61798096,  548.13201904],
#        [ 207.59408569,  397.00366211,  315.77627563,    2.        ,
#         -108.87758636,   73.62127686,  315.77627563, 1427.33996582,
#         1423.13000488,  949.61798096,  548.13201904],
#        [ 202.59803772,  455.28015137,  322.55981445,    2.        ,
#         -115.69550323,  108.71553802,  322.55981445, 1427.33996582,
#         1423.13000488,  949.61798096,  548.13201904],
#        [ 215.68159485,  247.08103943,  260.17868042,    2.        ,
#          -84.7667923 ,   -5.39123869,  260.17868042, 1427.33996582,
#         1423.13000488,  949.61798096,  548.13201904],
#        [ 234.49308777,  303.28533936,  262.46777344,    2.        ,
#          -77.00579834,   19.09913254,  262.46777344, 1427.33996582,
#         1423.13000488,  949.61798096,  548.13201904],
#        [ 250.81388855,  287.89248657,  284.92578125,    2.        ,
#          -75.51580811,   13.37642765,  284.92578125, 1427.33996582,
#         1423.13000488,  949.61798096,  548.13201904],
#        [ 229.62341309,  354.34918213,  288.60528564,    2.        ,
#          -87.57569885,   45.7951622 ,  288.60528564, 1427.33996582,
#         1423.13000488,  949.61798096,  548.13201904],
#        [ 252.86502075,  412.93890381,  304.94400024,    2.        ,
#          -81.10929108,   78.66136169,  304.94400024, 1427.33996582,
#         1423.13000488,  949.61798096,  548.13201904],
#        [ 265.61761475,  467.32778931,  317.81060791,    2.        ,
#          -78.63576508,  112.31228638,  317.81060791, 1427.33996582,
#         1423.13000488,  949.61798096,  548.13201904]])


# xx = xx[:,:3]
# yy = yy[:,4:7]
# error = np.linalg.norm(np.abs(xx - yy), axis=1)
# embed()

# def get_last_depth():
#     frame = kinect.get_last_depth_frame()
#     frame = frame.astype(np.uint8)
#     dep_frame = np.reshape(frame,[424,512])
#     return cv2.cvtColor(dep_frame, cv2.COLOR_GRAY)

# kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)

# frame_type ='depth'
# while True:
#     if frame_type == 'rgb':

#     # if kinect.has_new_color_frame():
#         last_frame = get_last_rbg()
#     else:
#     # if kinect.has_new_depth_frame():
#         last_frame = get_last_depth()

#     cv2.imshow('test', last_frame)
#     cv2.waitKey(1)



# =======================================================================
# ll = ctypes.cdll.LoadLibrary
# lib = ll("./lib/CPPlibs/libcppLib.so")
# # python并不会直接读取到.so的源文件，需要使用.argtypes告诉python在c函数中需要什么参数
# # 这样，在后面使用c函数时pytho会自动处理你的参数，从而达到像调用python参数一样
# lib.cdraw_rectangle.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
# # 同时，python也看不到函数返回什么，默认情况下,python认为函数返回了一个c中的int类型
# # 如果函数返回别的类型，就需要用到retype命令
# lib.cdraw_rectangle.restype = ctypes.c_void_p

# x, y

# l_wrist = [1190, 204] 
# l_elbow = [1277, 307]
# l_wrist = [962, 674] 
# l_elbow = [1012, 628]
# # l_wrist = [1366, 192] 
# # l_elbow = [1294, 312]
# depth_val = 100  # mm
# rec_size = 75 / depth_val * 100
# print(rec_size)
# angle = math.atan2(l_wrist[1] - l_elbow[1], l_wrist[0] - l_elbow[0]) * 180 / math.pi
# # print(angle)
# forearm_length = math.sqrt(pow(l_elbow[0] - l_wrist[0], 2.0) + pow(l_elbow[1] - l_wrist[1], 2.0))
# hand_center_x = l_wrist[0] + (l_wrist[0] - l_elbow[0]) / forearm_length * (forearm_length / 2)
# hand_center_y = l_wrist[1] + (l_wrist[1] - l_elbow[1]) / forearm_length * (forearm_length / 2)

# img = cv2.imread("/home/xuchengjun/ZXin/00_00_00000073.jpg")   # 00_08_00015350.jpg  00_00_00000078.jpg
# # img = torch.tensor(img).to('cuda')
# # img = np.array(img.cpu())
# frame_data = np.asarray(img, dtype = np.uint8)
# frame_data = frame_data.ctypes.data_as(ctypes.c_char_p)
# res = lib.cdraw_rectangle(img.shape[0], img.shape[1], frame_data, int(hand_center_x), int(hand_center_y), int(rec_size), angle)   # 后面的参数都是要计算的
#                                                                                                                        # 原始图像的height, width;; 原始图像的数据；； 手的中心（width, height）；； 边界框的信息；； 旋转量 
# print("have cropped hand image ..")

# int_arrPoint = ctypes.c_int * 2
# point_arr = int_arrPoint()
# lib.cadd_twoNum(point_arr)
# print("结果为: ", point_arr[0], point_arr[1])

# int_resPoints = ctypes.c_int * 10
# points_arr = int_resPoints()
# lib.cget_rectangle_points(points_arr)
# print("结果为: ", points_arr[0], points_arr[1])

# 划线
# cv2.line(img, (points_arr[0], points_arr[1]), (points_arr[2], points_arr[3]), (0, 255, 0), 3)
# cv2.line(img, (points_arr[2], points_arr[3]), (points_arr[4], points_arr[5]), (0, 255, 0), 3)
# cv2.line(img, (points_arr[4], points_arr[5]), (points_arr[6], points_arr[7]), (0, 255, 0), 3)
# cv2.line(img, (points_arr[6], points_arr[7]), (points_arr[0], points_arr[1]), (0, 255, 0), 3)
# cv2.circle(img, (int(hand_center_x), int(hand_center_y)), 4, (255, 0, 0), 2)
# cv2.circle(img, (int(hand_center_x) + 45, int(hand_center_y) - 45), 2, (255, 0, 0), 2)

# cv2.circle(img, (l_wrist[0], l_wrist[1]), 6, (255, 0, 0), -1)
# cv2.circle(img, (l_elbow[0], l_elbow[1]), 6, (255, 0, 0), -1)
# cv2.line(img, (l_wrist[0], l_wrist[1]), (l_elbow[0], l_elbow[1]), (0, 200, 255), 1)

# ctypes重载了*, 因此可以使用类型 *n 来表示n个该类型的元素在一起组成一个整体
# int_arr3 = ctypes.c_int * 3
# imgattr_para = int_arr3()
# imgattr_para[0] = 224
# imgattr_para[1] = 224
# imgattr_para[2] = 3
# tmp = ctypes.string_at(res, imgattr_para[0] * imgattr_para[1] * imgattr_para[2])
# nparr = np.frombuffer(tmp, np.uint8)
# img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# # cv2.imshow('source img', img)
# # cv2.imshow("crop img", img_decode)
# # cv2.waitKey(0)
# print('done ..')

# print(1080 * 3 / 4)
# =====================================================================


# =====================================================================

# def yaml_parser(config_base_path, file_name):
#     """
#     YAML file parser.
#     Args:
#         file_name (str): YAML file to be loaded
#         config_base_path (str, optional): Directory path of file
#                                           Default to '../modeling/config'.
        
#     Returns:
#         [dict]: Parsed YAML file as dictionary.
#     """
    
#     cur_dir = os.path.dirname(os.path.abspath(__file__))   # 定位到程序总文件夹的这一级
#     config_base_path = os.path.normpath(os.path.join(cur_dir, config_base_path))
    
#     file_path = os.path.join(config_base_path, file_name + '.yaml')
#     with open(file_path, 'r') as yaml_file:
#         yaml_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
#     return yaml_dict
    
#     print(config_base_path)
    
    
if __name__ == '__main__':
    import os
    # gesture_dict =  yaml_parser("lib", "hand_gesture")
    # cur_dir = os.path.dirname(os.path.abspath(__file__))   # 定位到程序总文件夹的这一级
    # gesture_dict = yaml_parser('config', 'hand_gesture', cur_dir)
    # gesture_dict['LEFTHAND']['ONE'] = 13
    # yaml_storer('config', 'hand_gesture_2', gesture_dict, cur_dir)
    # embed()
    # /home/xuchengjun/ZXin/smap/depth_and_rgb
    # /media/xuchengjun/disk/datasets/SaveImagesfromKinectV2-master/build/062422/062422_videos/frame_depth/33.jpg
    depth_img = cv2.imread("/media/xuchengjun/disk/datasets/SaveImagesfromKinectV2-master/build/062422/062422_videos/depth_frame/133.png")
    # bb, gg, rr = cv2.split(depth_img)
    # bb, gg, rr = cv2.split(depth_img)
    # depth = rr
    # depth_sum = 0.0
    # depth_trans = depth / 255 * 4096.0
    # for i in range(540 - 25, 540 + 25):
    #     for j in range(960 - 25, 960 + 25):
    #         depth_sum += depth_trans[i,j]
    # depth_avg = depth_sum / 2500
    print(f'depth: {depth_avg}')
    embed()
    # img = cv2.imread("/media/xuchengjun/disk/datasets/SaveImagesfromKinectV2-master/build/062422/062422_videos/rgb_frame/120.png")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.circle(img, (960, 540), 10, (255, 0, 0), 2)
    # embed()
    cv2.imshow('img', depth_img)
    cv2.waitKey(0)
    # embed()
    
    
    

    