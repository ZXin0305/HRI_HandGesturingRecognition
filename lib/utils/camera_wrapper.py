import sys
sys.path.append('/home/xuchengjun/ZXin/smap')
import os.path as osp
import glob
import sys
import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from path import Path
from IPython import embed
from lib.utils.tools import read_json, transform, show_map
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
import tf
from autolab_core import RigidTransform
import copy
import message_filters


class CustomDataset(Dataset):
    def __init__(self, cfg, dataset_path, dataset_name='CMU'):
        """
        this is to set the image dataset ..
        """
        # self.calibration_json_file = None
        self.dataset_name = dataset_name
        # if self.dataset_name == 'CMU':
        #     self.calibration_json_file = '/home/zx/panoptic-toolbox/170407_haggling_a1/calibration_170407_haggling_a1.json'
        #     self.cali_file = read_json(self.calibration_json_file)

        self.dataset_path = dataset_path
        #把该路径下的所有图片都加在进来（只是说的图片的路径）
        self.image_list = glob.glob(osp.join(dataset_path, '**/*.jpg'), recursive=True)    #**/*.jpg会在路径下进行迭代  返回的是路径的list
        self.image_list.extend(glob.glob(osp.join(dataset_path, '**/*.png'), recursive=True))  #该方法没有返回值，但会在已存在的列表中添加新的列表内容。
        self.image_list.extend(glob.glob(osp.join(dataset_path, '**/*.jpeg'), recursive=True))
        self.image_list = self.image_list[0:5]

        self.net_input_shape = (cfg.dataset.INPUT_SHAPE[1], cfg.dataset.INPUT_SHAPE[0]) # (width, height)
        #图片预处理
        normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        self.transform = transform

    def get_cam_(self, cali_file, cam_id):
        cameras = {(cam['panel'], cam['node']): cam for cam in cali_file['cameras']}
        cam = cameras[cam_id]
        return cam['K']


    def __getitem__(self, index):
        image_path = self.image_list[index].rstrip()
        image_name = image_path.replace(self.dataset_path, '').lstrip('/')
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        self.image_shape = (image.shape[1], image.shape[0])  # --> (width, heght)

        net_input_image, scale, pad_value = self.aug_croppad(image)
        net_input_image = self.transform(net_input_image)

        # if self.calibration_json_file is not None:
        #     # cam_id
        #     cam_id = image_name.split('/')[-1].split('_')
        #     lnum, rnum = int(cam_id[0]), int(cam_id[1])
        #     cam = self.get_cam_(self.cali_file, (lnum, rnum))
        #     scale['cam'] = cam

        return image, net_input_image, image_name, scale, self.dataset_name

    def __len__(self):
        return len(self.image_list)

    def aug_croppad(self, img):
        scale = dict()                    #创建字典
        crop_x = self.net_input_shape[0]  # width 自己设定的
        crop_y = self.net_input_shape[1]  # height 512
        scale['scale'] = min(crop_x / img.shape[1], crop_y / img.shape[0])  #返回的是最小值
        img_scale = cv2.resize(img, (0, 0), fx=scale['scale'], fy=scale['scale'])
        
        scale['img_width'] = img.shape[1]
        scale['img_height'] = img.shape[0]
        scale['net_width'] = crop_x
        scale['net_height'] = crop_y
        pad_value = [0,0]  # left,up

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
        scale['pad_value'] = pad_value
        return img_scale, scale, pad_value


class VideoReader:
    def __init__(self, file_name, cfg, my_cam):
        self.file_name = file_name
        self.net_input_shape = (cfg.dataset.INPUT_SHAPE[1], cfg.dataset.INPUT_SHAPE[0]) # (width, height)
        #图片预处理
        normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        self.cam = my_cam
        if self.cam:
            print('using myself cam')
            cam_data = read_json("/home/xuchengjun/ZXin/smap/cam_data/myCam.json")
            self.K = np.array(cam_data['kinect_1'])
        else:
            print('using dataset cam')
            cam_data = read_json('/home/xuchengjun/ZXin/smap/cam_data/cam.json')
            self.K = np.array(cam_data['K'])

        self.transform = transform
        
        try:
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, image = self.cap.read()
        if not was_read:
            raise StopIteration
        
        # transfrom the img
        net_input_image, scale = self.aug_croppad(image)
        scale['K'] = self.K
        scale['f_x'] = self.K[0,0]
        scale['f_y'] = self.K[1,1]
        scale['cx'] = self.K[0,2]
        scale['cy'] = self.K[1,2]
        net_input_image = self.transform(net_input_image)
        net_input_image = net_input_image.unsqueeze(0)

        return image, net_input_image, scale

    def aug_croppad(self, img):
        scale = dict()                    #创建字典
        crop_x = self.net_input_shape[0]  # width 自己设定的
        crop_y = self.net_input_shape[1]  # height 512
        scale['scale'] = min(crop_x / img.shape[1], crop_y / img.shape[0])  #返回的是最小值
        img_scale = cv2.resize(img, (0, 0), fx=scale['scale'], fy=scale['scale'])
        
        scale['img_width'] = img.shape[1]
        scale['img_height'] = img.shape[0]
        scale['net_width'] = crop_x
        scale['net_height'] = crop_y
        pad_value = [0,0]  # left,up

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
            
        scale['pad_value'] = pad_value
        
        return img_scale, scale

class VideoReaderWithDepth:
    def __init__(self, rgb_file_name, depth_file_name, cfg, my_cam):
        self.rgb = rgb_file_name
        self.depth = depth_file_name
        self.net_input_shape = (cfg.dataset.INPUT_SHAPE[1], cfg.dataset.INPUT_SHAPE[0]) # (width, height)
        #图片预处理
        normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        self.cam = my_cam
        if self.cam:
            print('using myself cam')
            cam_data = read_json("/home/xuchengjun/ZXin/smap/cam_data/myCam.json")
            self.K = np.array(cam_data['kinect_1'])
        else:
            print('using dataset cam')
            cam_data = read_json('/home/xuchengjun/ZXin/smap/cam_data/cam.json')
            self.K = np.array(cam_data['K'])

        self.transform = transform
        
        # try:
        #     self.file_name = int(file_name)
        # except ValueError:
        #     pass

    def __iter__(self):
        self.cap_rgb = cv2.VideoCapture(self.rgb)
        self.cap_rgb.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap_rgb.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.cap_depth = cv2.VideoCapture(self.depth)
        self.cap_depth.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap_depth.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)       
        if not self.cap_rgb.isOpened() or not self.cap_depth.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.rgb))
        return self

    def __next__(self):
        rgb_is_read, image = self.cap_rgb.read()
        depth_is_read, depth = self.cap_depth.read()
        if not rgb_is_read or not depth_is_read:
            raise StopIteration
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)   # 20220714 不确定， 在从bin文件中提取出来的时候，RGB的编码格式是BGRA8, 所以在这里制作数据集的时候需要进行转换， 以便和实际的对上
                                                          # 但是实际上不能用BGR转换，读取之后直接用RGBA到RGB就可以，显示出来是一样的
        # transfrom the img
        net_input_image, scale = self.aug_croppad(image)
        scale['K'] = self.K
        scale['f_x'] = self.K[0,0]
        scale['f_y'] = self.K[1,1]
        scale['cx'] = self.K[0,2]
        scale['cy'] = self.K[1,2]
        net_input_image = self.transform(net_input_image)
        net_input_image = net_input_image.unsqueeze(0)

        return image, depth, net_input_image, scale

    def aug_croppad(self, img):
        scale = dict()                    #创建字典
        crop_x = self.net_input_shape[0]  # width 自己设定的
        crop_y = self.net_input_shape[1]  # height 512
        scale['scale'] = min(crop_x / img.shape[1], crop_y / img.shape[0])  #返回的是最小值
        img_scale = cv2.resize(img, (0, 0), fx=scale['scale'], fy=scale['scale'])
        
        scale['img_width'] = img.shape[1]
        scale['img_height'] = img.shape[0]
        scale['net_width'] = crop_x
        scale['net_height'] = crop_y
        pad_value = [0,0]  # left,up

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
            
        scale['pad_value'] = pad_value
        
        return img_scale, scale


class CameraReader(object):
    def __init__(self, topic_name1, cfg, my_cam):
        self.color = None
        self.depth = None
        self.color_convert = None
        self.cv_bridge = CvBridge()
        self.color_topic = topic_name1
        self.depth_topic = 'kinectSDK/depth'
        # self.image_sub = rospy.Subscriber(self.color_topic, Image, self.callback)
        # self.image_sub2 = rospy.Subscriber(self.depth_topic, Image, self.callback2)
        self.color_sub = message_filters.Subscriber(self.color_topic, Image)
        self.depth_sub = message_filters.Subscriber(self.depth_topic, Image)
        self.syc = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.syc.registerCallback(self.callback)
        self.net_input_shape = (cfg.dataset.INPUT_SHAPE[1], cfg.dataset.INPUT_SHAPE[0]) # (width, height)
        
        #图片预处理
        normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        self.cam = my_cam
        if self.cam:
            print('using myself cam')
            cam_data = read_json("/home/xuchengjun/ZXin/smap/cam_data/myCam.json")
            self.K = np.array(cam_data['kinect_1'])
        else:
            print('using dataset cam')
            cam_data = read_json('/home/xuchengjun/ZXin/smap/cam_data/cam.json')
            self.K = np.array(cam_data['K'])

        self.transform = transform

    def callback(self, msg_color, msg_depth):
        # rospy.loginfo('Image has received...')
        self.color = self.cv_bridge.imgmsg_to_cv2(msg_color, 'rgba8')
        self.color_convert = cv2.cvtColor(self.color, cv2.COLOR_RGBA2RGB)  # 这里是从 RGBA到RGB是因为在cpp端转化成ROS信息的时候就变成了  'rgba8'

        self.depth = self.cv_bridge.imgmsg_to_cv2(msg_depth, 'mono8')
        # print(self.depth.shape)
    
    # def callback2(self, msg):
    #     self.image2 = self.cv_bridge.imgmsg_to_cv2(msg, 'mono8')
    #     self.depth_show = self.image2 / 255
    #     self.image_convert2 = self.image2 / 255 * 4096.0

    def __iter__(self):
        return self

    def __next__(self):

        # if self.image is None or self.image_convert is None:
        if self.color is None or self.color_convert is None or self.depth is None: 
            raise StopIteration

        # transfrom the img
        net_input_image, scale = self.aug_croppad(self.color_convert)
        scale['K'] = self.K
        scale['f_x'] = self.K[0,0]
        scale['f_y'] = self.K[1,1]
        scale['cx'] = self.K[0,2]
        scale['cy'] = self.K[1,2]
        net_input_image = self.transform(net_input_image)
        net_input_image = net_input_image.unsqueeze(0)

        return self.color_convert, net_input_image, scale, self.depth

    def aug_croppad(self, img):
        scale = dict()                    #创建字典
        crop_x = self.net_input_shape[0]  # width 自己设定的
        crop_y = self.net_input_shape[1]  # height 512
        scale['scale'] = min(crop_x / img.shape[1], crop_y / img.shape[0])  #返回的是最小值
        img_scale = cv2.resize(img, (0, 0), fx=scale['scale'], fy=scale['scale'])
        
        scale['img_width'] = img.shape[1]
        scale['img_height'] = img.shape[0]
        scale['net_width'] = crop_x
        scale['net_height'] = crop_y
        pad_value = [0,0]  # left,up

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
            
        scale['pad_value'] = pad_value
        
        return img_scale, scale

"""
class CameraReader(object):
    def __init__(self, topic_name, cfg, my_cam):
        self.image = None
        self.image2 = None
        self.image_convert = None
        self.image_convert2 = None
        self.depth_show = None
        self.cv_bridge = CvBridge()
        self.topic = topic_name
        self.image_sub = rospy.Subscriber(self.topic, Image, self.callback)
        self.net_input_shape = (cfg.dataset.INPUT_SHAPE[1], cfg.dataset.INPUT_SHAPE[0]) # (width, height)
        
        #图片预处理
        normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        self.cam = my_cam
        if self.cam:
            print('using myself cam')
            cam_data = read_json("/home/xuchengjun/ZXin/smap/cam_data/myCam.json")
            self.K = np.array(cam_data['kinect_1'])
        else:
            print('using dataset cam')
            cam_data = read_json('/home/xuchengjun/ZXin/smap/cam_data/cam.json')
            self.K = np.array(cam_data['K'])

        self.transform = transform

    def callback(self, msg):
        # rospy.loginfo('Image has received...')
        self.image = self.cv_bridge.imgmsg_to_cv2(msg)

    def __iter__(self):
        return self

    def __next__(self):

        if self.image is None: 
            raise StopIteration

        # transfrom the img
        net_input_image, scale = self.aug_croppad(self.image)
        scale['K'] = self.K
        scale['f_x'] = self.K[0,0]
        scale['f_y'] = self.K[1,1]
        scale['cx'] = self.K[0,2]
        scale['cy'] = self.K[1,2]
        net_input_image = self.transform(net_input_image)
        net_input_image = net_input_image.unsqueeze(0)

        return self.image, net_input_image, scale, None

    def aug_croppad(self, img):
        scale = dict()                    #创建字典
        crop_x = self.net_input_shape[0]  # width 自己设定的
        crop_y = self.net_input_shape[1]  # height 512
        scale['scale'] = min(crop_x / img.shape[1], crop_y / img.shape[0])  #返回的是最小值
        img_scale = cv2.resize(img, (0, 0), fx=scale['scale'], fy=scale['scale'])
        
        scale['img_width'] = img.shape[1]
        scale['img_height'] = img.shape[0]
        scale['net_width'] = crop_x
        scale['net_height'] = crop_y
        pad_value = [0,0]  # left,up

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
            
        scale['pad_value'] = pad_value
        
        return img_scale, scale
"""

class CameraInfo:
    def __init__(self):
        self.child_frame = "camera_base_1"
        self.parent_frame = "marker_0"

    def get_tf(self):  # camera to world
        listener = tf.TransformListener()
        rate = rospy.Rate(30)
        rot_matrix = None
        trans = None
        while not rospy.is_shutdown():
            try:
                (t, q) = listener.lookupTransform(self.child_frame, self.parent_frame, rospy.Time(0))
                # embed()
            except:
                rospy.loginfo("wait for camera info ..")
                rate.sleep()
                continue

            #得到相应的旋转矩阵和平移向量
            trans = self.t2trans(t)
            ts = trans
            rot_matrix = self.q2rot(q)


            break

        rospy.loginfo_once("have get camera2world info ..")
        print(rot_matrix)
        print(trans)
        return rot_matrix, ts


    
    def q2rot(self, q): 
        """四元数到旋转矩阵
        
        """
        w, x, y, z= q[0], q[1], q[2], q[3]
        # qq = np.array([w,x,y,z])
        # rotMat = RigidTransform(qq, trans)
        rotMat = np.array(
            [[1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
             [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
             [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2]]
        )
        # rotMat = np.array([[ 0.977503 ,  0.210436 ,-0.0143229],[0.0273426 , -0.193757,  -0.980668],[-0.209143,   0.958214 , -0.195152]])
        return rotMat
    
    def t2trans(self, t):
        """转换成numpy形式的平移向量
        """
        trans = np.zeros((3, 1), dtype=np.float32)
        for i in range(len(t)):
            trans[i] = t[i]
        # trans = np.array([-0.414375,0.68425,2.33348])
        return trans

if __name__ == '__main__':
    rospy.init_node("test_get_camera_info", anonymous=True)
    rot_matrix, trans = CameraInfo().listen_tf() 
    print(rot_matrix, trans)