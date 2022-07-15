import sys

from numpy.lib.polynomial import roots
sys.path.append('/home/xuchengjun/ZXin/smap')
from time import thread_time
import numpy as np
import torch
from collections import OrderedDict
import json
import os
from matplotlib import pyplot as plt
import cv2
import torchvision.transforms as transforms
from IPython import embed
import pandas as pd
from lib.collect_action.collect import *

from lib.utils.post_3d import back_projection
from mayavi import mlab
from exps.stage3_root2.config import cfg
import copy
import math
import yaml

class Norm():
    def __init__(self, means=128, stand=256):
        self.means = means
        self.stand = stand

    def _norm(self,img):
        img = img.astype(np.float32)
        img = (img - 128) / 256
        img = img.transpose((2,0,1))
        img = torch.from_numpy(img).float()
        return img

def transform(img, cfg):
    normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    img_trans = transform(img)
    return img_trans

def load_state(net, checkpoint):
    source_state = checkpoint['state_dict']  # no module
    target_state = net.state_dict()  # module
    new_target_state = OrderedDict()

    # 1.if no cuda using, remove the "module."
    # for k,v in source_state.items(): #k:键名，v:对应的权值参数
    #     name = k[7:]
    #     new_target_state[name] = v
    # net.load_state_dict(new_target_state) 

    # 2.
    for target_key, target_value in target_state.items():
        if target_key in source_state and source_state[target_key].size() == target_state[target_key].size():
            new_target_state[target_key] = source_state[target_key]
        else:
            new_target_state[target_key] = target_state[target_key]
            print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))

    net.load_state_dict(new_target_state)

def load_state_with_no_ck(net, source_state):
    new_source_dict = OrderedDict()
    for k,v in source_state.items(): #k:键名，v:对应的权值参数
        name = k[7:]
        new_source_dict[name] = v
    net.load_state_dict(new_source_dict)

def read_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def write_json(path, result):
    with open(path, 'w') as file:
        json.dump(result, file)
        
def read_csv(path):
    try:
        data = pd.read_csv(path, header=0)
    except:
        print('dataset not exist')
        return 
    
    return data

def ensure_dir(path):
    """
    create directories if *path* does not exist
    """
    if not os.path.isdir(path):
        os.makedirs(path)

def show_map(map_, id=1):
    map_ = map_.detach().cpu()
    map_ = np.array(map_)

    # map_ *= 255

    # show one img once a time 
    plt.subplot(111)
    plt.imshow(map_)
    plt.axis('off')
    # plt.savefig(f'/home/xuchengjun/ZXin/smap/results/heatmap_{id}.jpg')
    plt.show()

    # show multiple imgs
    # img_num = 15
    # for i in range(img_num):
    #     plt.subplot(3,5,i+1)
    #     plt.imshow(map_[i])
    #     plt.axis('off')
    #     # plt.savefig()
    # plt.show()

def save_img_results(img, img_path, coors, scale, i):
    img_result_dir = '/home/xuchengjun/ZXin/human_pose/exps/stage3_root2/2d_img'
    if (coors.shape[0]) > 0:
        coors[:,:,:2] /= scale
        for human_num in range(len(coors)):
            for idx in range(15):
                if int(coors[human_num, idx, 0]) == 0 and \
                    int(coors[human_num, idx, 1]) == 0:
                    continue 
                cv2.circle(img, center=(int(coors[human_num, idx, 0]), int(coors[human_num, idx, 1])), radius=7, color=(255,0,0), thickness=-1)
    
    cv2.imwrite(os.path.join(img_result_dir, f'{i}.jpg'), img)
    print('have saved the result 2d img to -> {}'.format(os.path.join(img_result_dir, f'{i}.jpg')))

def recover_origin_resolution(bodys, scale):
    if len(bodys) > 0:
        bodys[:, :, 0] = (bodys[:, :, 0] - 0) / scale
        bodys[:, :, 1] = (bodys[:, :, 1] - 22) / scale
    return bodys

def draw_lines(img, bodys, eadges, color, thickness=8):
    """
    img:original image
    bodyy:predicted 2d body coors
    eadges:the order to draw the lines
    color:optional ..
    """
    for body in bodys:
        for i in range(len(eadges)):
            start_coor = body[eadges[i][0]][:2]  # [x1,y1]
            end_coor = body[eadges[i][1]][:2]    # [x2,y2]
            if (int(start_coor[0]) == 0 or int(start_coor[1]) == 0) or \
                (int(end_coor[0]) == 0 or int(end_coor[1]) == 0):
                continue
            cv2.line(img, (int(start_coor[0]), int(start_coor[1])), (int(end_coor[0]), int(end_coor[1])), color=color, thickness=thickness)


def draw_cicles(bodys, image, is_gt = False, color = (255,0,0)):

    for i in range(len(bodys)):
        for j in range(15):
            if (int(bodys[i][j][0]) == 0 and int(bodys[i][j][1]) == 0):
                continue
            # if j != 2 and not is_gt:
            #     continue
            if not is_gt:
                cv2.circle(image, center=(int(bodys[i,j,0]), int(bodys[i,j,1])), color=color, radius=8, thickness=-1)
            else:
                cv2.circle(image, center=(int(bodys[i,j,0]), int(bodys[i,j,1])), color=color, radius=8, thickness=-1)

def draw_cicles_once_only_one(bodys, image, color = (255, 0, 0)):
    for i in range(15):
        if (int(bodys[i][0]) == 0 and int(bodys[i][1]) == 0):
            continue
        if np.isnan(bodys[i][0]) or np.isnan(bodys[i][1]):
            continue
        cv2.circle(image, center=(int(bodys[i,0]), int(bodys[i,1])), color=color, radius=8, thickness=-1)    

def draw_lines_once_only_one(img, body, eadges, color, thickness=8): 
    for i in range(len(eadges)):
        start_coor = body[eadges[i][0]][:2]  # [x1,y1]
        end_coor = body[eadges[i][1]][:2]    # [x2,y2]
        if np.isnan(start_coor[0]) or np.isnan(start_coor[1]) or np.isnan(end_coor[0]) or np.isnan(end_coor[1]): 
            continue
        if (int(start_coor[0]) == 0 or int(start_coor[1]) == 0) or \
            (int(end_coor[0]) == 0 or int(end_coor[1]) == 0):
            continue
        # embed()
        cv2.line(img, (int(start_coor[0]), int(start_coor[1])), (int(end_coor[0]), int(end_coor[1])), color=color, thickness=thickness)    
                
def croppad_img(img, cfg):
    scale = dict()                    #创建字典
    crop_x = cfg.dataset.INPUT_SHAPE[1]  # width 自己设定的
    crop_y = cfg.dataset.INPUT_SHAPE[0]  # height 
    scale['scale'] = min(crop_x / img.shape[1], crop_y / img.shape[0])  #返回的是最小值
    img_scale = cv2.resize(img, (0, 0), fx=scale['scale'], fy=scale['scale'])
    
    scale['img_width'] = img.shape[1]
    scale['img_height'] = img.shape[0]
    scale['net_width'] = crop_x
    scale['net_height'] = crop_y
    pad_value = [0,0]  # left, right, up, down

    center = np.array([img.shape[1]//2, img.shape[0]//2], dtype=np.int)
    
    if img_scale.shape[1] < crop_x:    # pad left and right
        margin_l = (crop_x - img_scale.shape[1]) // 2
        margin_r = crop_x - img_scale.shape[1] - margin_l
        pad_l = np.ones((img_scale.shape[0], margin_l, 3), dtype=np.uint8) * 128
        pad_r = np.ones((img_scale.shape[0], margin_r, 3), dtype=np.uint8) * 128
        pad_value[0] = margin_l
        img_scale = np.concatenate((pad_l, img_scale, pad_r), axis=1)        #在1维进行拼接　也就是w
    elif img_scale.shape[0] < crop_y:  # pad up and down
        margin_u = (crop_y - img_scale.shape[0]) // 2
        margin_d = crop_y - img_scale.shape[0] - margin_u
        pad_u = np.ones((margin_u, img_scale.shape[1], 3), dtype=np.uint8) * 128
        pad_d = np.ones((margin_d, img_scale.shape[1], 3), dtype=np.uint8) * 128
        pad_value[1] = margin_u
        img_scale = np.concatenate((pad_u, img_scale, pad_d), axis=0)       #在0维进行拼接　也就是h
    
    return img_scale, scale, pad_value


def reproject(x, d, K):
    X = np.zeros((len(d), 2), np.float)
    X[:, 0] = K[0, 0] * (x[:, 0] / d) + K[0, 2]
    X[:, 1] = K[1, 1] * (x[:, 1] / d) + K[1, 2]
    return X

def change_pose_order(pred_bodys, root_idx=2):
    pred_pnum = len(pred_bodys)
    change_pose = []
    for i in range(pred_pnum):
        X = []
        Y = []
        Z = []
        if pred_bodys[i][root_idx][3] == 0:
            continue
        for j in range(15):
            X.append(pred_bodys[i][j][0])
            Y.append(pred_bodys[i][j][1])
            Z.append(pred_bodys[i][j][2])
        single_person = [X, Y, Z]
        change_pose.append(single_person)
    return change_pose
def back_pose_order(new_pred_bodys):
    pred_pnum = len(new_pred_bodys)
    back_pose = []
    for i in range(pred_pnum):
        single_person = []
        for j in range(15):
            joint = []
            joint.append(new_pred_bodys[i][0][j])  # x
            joint.append(new_pred_bodys[i][1][j])  # y
            joint.append(new_pred_bodys[i][2][j])  # z
            single_person.append(joint)
        back_pose.append(single_person)
        
    return back_pose


def project_to_pixel(pred_bodys, K, root_idx=2):
    """
    bodys: pred (person_num,15,4) --> 4:(X,Y,Z,1) 3d
    cam:相机内参
    """

    new_pred_bodys = change_pose_order(pred_bodys)
    new_pred_bodys = np.array(new_pred_bodys)
    # print(new_pred_bodys.shape)
    for body in new_pred_bodys:
        body[0:2, :] = body[0:2, :] / body[2, :]
        body[0, :] = K[0, 0] * body[0, :] + K[0, 1] * body[1, :] + K[0, 2]  # x
        body[1, :] = K[1, 0] * body[0, :] + K[1, 1] * body[1, :] + K[1, 2]  # y

    back_pose = back_pose_order(new_pred_bodys)
    pixel_bodys = np.array(back_pose)
    return pixel_bodys

def draw_3d_lines(mlab,p1,p2,color=(0,0,1)):
    xs = np.array([p1[0], p2[0]])
    ys = np.array([p1[1], p2[1]])
    zs = np.array([p1[2], p2[2]])
    mlab.plot3d(xs, ys, zs, [1, 2], tube_radius=0.01, color=color)
def draw_3d_sphere(mlab, point3d, color=(0,1,0)):
    mlab.points3d(
          np.array(point3d[0]), np.array(point3d[1]), np.array(point3d[2]),
          scale_factor=0.02, color=color
      )
def show_3d_results(pred_3d_poses, bodys_eadges):

    mlab.figure(1, bgcolor=(1,1,1), size=(960, 540))
    mlab.view(azimuth=180, elevation=0)
    for i in range(len(pred_3d_poses)):
        for j in range(len(bodys_eadges)):
            p1 = pred_3d_poses[i][bodys_eadges[j][0]]
            p2 = pred_3d_poses[i][bodys_eadges[j][1]]
            draw_3d_lines(mlab, p1, p2)
    for i in range(len(pred_3d_poses)):
        for j in range(15):
            draw_3d_sphere(mlab, pred_3d_poses[i][j])
    mlab.show()


def filter_pose(bodys):
    """
    function:remove person who was block, that his neck and root are both occluded 
    gt_bodys: (pnum, 15, 11)
    loop pnum * pnum
    decision: root depth and the joint distance
    """
    new_bodys = []
    need_remove = []
    bodys_len = len(bodys)
    for i in range(bodys_len):
        if i == bodys_len - 1:
            break
        for j in range(i+1, bodys_len):
            first_root_x = bodys[i, 2, 0]  # pixel
            first_root_y = bodys[i, 2, 1]
            second_root_x = bodys[j, 2, 0]
            second_root_y = bodys[j, 2, 1]

            first_neck_x = bodys[i, 0, 0]  # pixel
            first_neck_y = bodys[i, 0, 1]
            second_neck_x = bodys[j, 0, 0]
            second_neck_y = bodys[j, 0, 1]  

            root_depth_first = bodys[i, 2, 2]
            root_depth_second = bodys[j, 2, 2]          

            # cal joint distance
            # root_dis = np.math.sqrt((first_root_x - second_root_x)**2 + (first_root_y - second_root_y)**2)
            # neck_dis = np.math.sqrt((first_neck_x - second_neck_x)**2 + (first_neck_y - second_neck_y)**2)

            root_dis = np.abs(first_root_x - second_root_x)
            neck_dis = np.abs(first_neck_x - second_neck_x)
            # print(root_dis, neck_dis)
            if root_dis < 30 and neck_dis < 30:
                if root_depth_first <= root_depth_second:
                    need_remove.append(j)
                elif root_depth_first > root_depth_second:
                    need_remove.append(i)
            
    new_need_remove = set(need_remove)
    for k in range(bodys_len):
        if k in new_need_remove:
            continue
        else:
            new_bodys.append(bodys[k])
    return new_bodys


def augment_pose(pred_3d_bodys, scale=1, trans=0, angles=0):
    pose_3d = copy.deepcopy(pred_3d_bodys)
    human_num = pred_3d_bodys.shape[0]
    joint_num = pred_3d_bodys.shape[1]

    # camera_theta = 45 * np.pi / 180

    # first
    trans_ = copy.deepcopy(pred_3d_bodys[0, 2, :3])  # --> (X, Y, Z) in camera coordinate system

    """
    20220418
    如果要加上不同的距离的话
    可以加上下面的代码，将trans_改变
    """
    # trans_ = [trans_[0] + trans * (trans_[0] / trans_[2]), trans_[1], trans_[2] + trans]

    pose_3d[:,:,0] -= trans_[0]
    pose_3d[:,:,1] -= trans_[1]
    pose_3d[:,:,2] -= trans_[2] 

    # 坐标转换到自身的坐标系中
    # 在自身坐标系中进行旋转
    # camera_rotMat = np.array([[1, 0, 0],
    #                           [0, np.cos(camera_theta), np.sin(camera_theta)],
    #                           [0, -np.sin(camera_theta), np.cos(camera_theta)]])
    # H_first = np.ones((4,4))
    # H_first[0:3, 0:3] = camera_rotMat.T
    # H_first[3, 0:3] = [0,0,0]
    # H_first[:3, 3] = [0,0,0]
    # pose_3d = pose_3d @ H_first
    
    theta = angles * np.pi / 180
    rotMat = np.array([[np.cos(theta), 0, -np.sin(theta)],
                        [0, 1, 0],
                        [np.sin(theta), 0, np.cos(theta)]])
    
    xyTrans = [trans * np.cos(theta), trans * np.sin(theta), 0]
    xyTrans = np.array(xyTrans)
    K = np.array([[scale, 0, 0, 0], [0, scale, 0, 0],
                   [0, 0, 1, 0], [0, 0, 0, 1]])
    H = np.ones((4, 4))
    H[0:3, 0:3] = rotMat.T
    H[3, :3] = xyTrans  # trans
    # embed()
    P = K @ H
    P[:3,3] = [0,0,0]
    # embed()
    data = np.ones((human_num, joint_num, 4))
    data[:,:,0:3] = pose_3d[:,:,0:3]
    data = data @ P  

    
    # 再转回到相机的坐标系中
    # 不用再转到之前的角度的相机坐标系了。。
       
    # rotMat_2 = np.array([[1, 0, 0],
    #                      [0, np.cos(-camera_theta), np.sin(-camera_theta)],
    #                      [0, -np.sin(-camera_theta), np.cos(-camera_theta)]])
    
    # H_ = np.ones((4,4))
    # H_[0:3, 0:3] = rotMat_2.T
    # H_[3, 0:3] = [0,0,0]
    # H_[:3, 3] = [0,0,0]
    # data = data @ H_  
    data[:,:,0] += trans_[0]
    data[:,:,1] += trans_[1]
    data[:,:,2] += trans_[2] 

    return data

def convert_skeleton_to_image(data_numpy):
    data_numpy = np.squeeze(data_numpy, axis=0)
    data_max = np.max(data_numpy, (1, 2, 3))
    data_min = np.min(data_numpy, (1, 2, 3))
    img_data = np.zeros((
                            data_numpy.shape[1],
                            data_numpy.shape[2],
                            data_numpy.shape[0]   
                        ))
    # # 之前的
    # img_data[:, :, 0] = (data_max[0] - data_numpy[0, :, :, 0]) * (255 / (data_max[0] - data_min[0])) 
    # img_data[:, :, 1] = (data_max[1] - data_numpy[1, :, :, 0]) * (255 / (data_max[1] - data_min[1]))
    # img_data[:, :, 2] = (data_max[2] - data_numpy[2, :, :, 0]) * (255 / (data_max[2] - data_min[2]))    
    # img_data = cv2.resize(img_data, (244, 244))

    img_data[:, :, 0] = data_numpy[0, :, :, 0] * 255
    img_data[:, :, 1] = data_numpy[1, :, :, 0] * 255
    img_data[:, :, 2] = data_numpy[2, :, :, 0] * 255

    # img_data = cv2.resize(img_data, (244, 244))

    return img_data

def get_embedding(pose_data, joint_num=15, coor_num=3):
    data_numpy = np.array(pose_data)
    data_numpy = np.reshape(data_numpy,
                            (1,
                            data_numpy.shape[0],
                            joint_num,
                            coor_num,
                            1))
    data_numpy = np.moveaxis(data_numpy, [1, 2, 3], [2, 3, 1])
    # N, C, T, J, M = data_numpy.shape
    embed_image = convert_skeleton_to_image(data_numpy)
    return embed_image

def camera2world(pose_data, T):
    """
    cam_ex --> (R, t)
    """
    pose_data = pose_data.transpose(1,0)
    world_pose_data = T @ pose_data
    # world_pose_data[0] += cam_ex[1][0]
    # world_pose_data[1] += cam_ex[1][1]
    # world_pose_data[2] += cam_ex[1][2]
    world_pose_data = world_pose_data.transpose(1,0)
    return world_pose_data

def soft_max(num_list):
    new_num_list = np.array(num_list.cpu())[0]
    total = 0e-6
    exp_list = []
    soft_max_list = []
    if len(num_list) > 0:
        # embed()

        exp_list = np.exp(new_num_list)
        total = sum(exp_list)
        for i in range(len(num_list)):
            soft_max_list.append(np.exp(new_num_list[i]) / total)

    return soft_max_list

def cal_pose_changes(pose_list, cal_frames, point_num):
    pose_list = np.array(pose_list)
    # differ = abs(np.array(pose_list)[int(cal_frames):] - np.array(pose_list)[int(cal_frames)])  #第一步是这样的
    # tmp = sum(differ) / (54 - cal_frames)

    # change_val = (sum(tmp[:,0]) + sum(tmp[:,1]) + sum(tmp[:,2])) / point_num   #所有帧的L2-distance
    # change_val = sum(tmp)
    angle_list = []
    for i in range(len(pose_list)-cal_frames): 
        current_pose = pose_list[cal_frames+i]
        relbow2rshoulder = current_pose[9] - current_pose[10]
        relbow2rwrist = current_pose[11] - current_pose[10]
        cos_theta = relbow2rshoulder.dot(relbow2rwrist)/(np.linalg.norm(relbow2rshoulder) * np.linalg.norm(relbow2rwrist))
        angle = np.arccos(cos_theta)
        angle_list.append(angle * 180 / np.pi)
        
    return np.array(angle_list)

def yaml_parser(config_base_path, file_name, cur_dir):
    """
    YAML file parser.
    Args:
        file_name (str): YAML file to be loaded
        config_base_path (str, optional): Directory path of file
                                          Default to '../modeling/config'.
        
    Returns:
        [dict]: Parsed YAML file as dictionary.
    """

    config_base_path = os.path.normpath(os.path.join(cur_dir, config_base_path))
    file_path = os.path.join(config_base_path, file_name + '.yaml')
    with open(file_path, 'r') as yaml_file:
        yaml_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return yaml_dict

def yaml_storer(config_base_path, file_name, yaml_dict, cur_dir):

    config_base_path = os.path.normpath(os.path.join(cur_dir, config_base_path))
    file_path = os.path.join(config_base_path, file_name + '.yaml')
    with open(file_path, 'w') as yaml_file:
        # yaml_file.write(yaml.dump(yaml_dict, stream, OrderedDumper, allow_unicode=True, **kwds))
        yaml.safe_dump(yaml_dict, yaml_file, encoding='utf-8', allow_unicode=True, sort_keys=False)

if __name__ == '__main__':

    pose = np.array([[[  5.26815605, -29.7260952 , 203.01779175,   1.        ],
        [  2.11058378, -38.14060211, 187.13592529,   1.        ],
        [  4.21306372,  14.6570797 , 209.03567505,   1.        ],
        [ 21.9207325 , -31.39882088, 201.90002441,   1.        ],
        [ 28.60899925,  -6.57456779, 208.542099  ,   1.        ],
        [ 29.17471695,  13.57969666, 201.89425659,   1.        ],
        [ 13.42282295,  14.26616192, 209.14651489,   1.        ],
        [ 14.13363647,  52.19490814, 214.57441711,   1.        ],
        [  9.92290878,  87.99411774, 224.11352539,   1.        ],
        [-11.23623085, -28.36738014, 203.87025452,   1.        ],
        [-20.94290543,  -0.87338543, 209.06535339,   1.        ],
        [-22.51371956,  21.79220963, 208.14646912,   1.         ],
        [ -5.01691389,  14.97479153, 209.09313965,   1.        ],
        [ -9.9235878 ,  50.74664307, 214.59390259,   1.        ],
        [-11.64457703,  86.69314575, 225.88101196,   1.        ]]])
    pose = augment_pose(pose / 100,angles=0)
    # norm_pose = vector_pose_normalization(pose[0,:,:])
    # norm_pose = pose_normalization(pose[0,:,:])
    # pose[0] = norm_pose 
    print(pose)
    show_3d_results(pose, cfg.SHOW.BODY_EADGES)
