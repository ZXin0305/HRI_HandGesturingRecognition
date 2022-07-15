import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters

class CameraTopic(object):
    def __init__(self, topic_name, video_path):
        self.image = None
        self.cv_bridge = CvBridge()
        self.topic = topic_name
        self.image_sub = rospy.Subscriber(self.topic, Image, self.callback)
        self.video_path = video_path
        self.fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.out = cv2.VideoWriter(self.video_path, self.fourcc, 20.0,(1920, 1080))

    def callback(self, msg):
        self.image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        if self.image is not None:

            self.out.write(self.image)
            cv2.imshow('img', self.image)
            cv2.waitKey(33)
        else:
            raise StopIteration

if __name__ == "__main__":
    rospy.init_node('Image_sub', anonymous=True)
    rospy.Rate(60).sleep()
    video_path = "/media/xuchengjun/zx/videos/filter.avi"
    camera_topic = "/kinect2_1/hd/image_color"
    cam = CameraTopic(camera_topic, video_path)
    rospy.spin()