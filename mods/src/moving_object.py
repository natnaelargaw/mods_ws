#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function

import roslib
roslib.load_manifest('mods')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from collections import deque
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import os
import numpy as np
from config import *
from model import *
from utility import *

from PIL import Image as im
from numpy import asarray
"""
This class subscribes to any live video stream, /naoqi_driver/camera/front/image_raw topic in this case, and 
provides a moving object detection and segmentation map and set locomotion parameters for the robot pepper focuse
on the moving objects in real-time.

The class is initialized with cv_bridge instance to convert cv image to ros image and various publisher and subscriber instances
along with their call back.

The moving object detection and segmentation model callback function subscribes to /naoqi_driver/camera/front/image_raw topic and perform two major tasks. 
predicting the moving object map using predict() function and localizing the ROI of the robot using localize_torsow().

"""

class moving_object_detection:
  
  def __init__(self):
    self.object_map_pub = rospy.Publisher("mods/object_image",Image, queue_size=10)
    self.localize_pub = rospy.Publisher("mods/object_roi",String,queue_size=10)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/naoqi_driver/camera/front/image_raw", Image, self.saliency_model_callback)

    self.x = Input(batch_shape=(1, None, shape_r, shape_c, 3))
    self.x2 = Input(batch_shape=(1, None, shape_r, shape_c, 3))
    self.x3 = Input(batch_shape=(1, None, shape_r, shape_c, 3))
    self.stateful = True
    self.m = Model(inputs=[self.x, self.x2, self.x3], outputs=transform_saliency([self.x, self.x2, self.x3], self.stateful))
    self.m.load_weights('XYshift.h5')
    self.queue = deque()


  
  """ callback function for the topic /naoqi_driver/camera/front/image_raw  
  important documentation for the incompatability issue of tensorflow and cv_bridge:https://jacqueskaiser.com/posts/2020/03/ros-tensorflow and  https://medium.com/@beta_b0t/how-to-setup-ros-with-python-3-44a69ca36674

  """
  def saliency_model_callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    print("Loading ACL weights")
    # cv.imshow("Frame", cv_image)
    # cv.waitKey(3)

   
    # get video access
    if len(self.queue) != num_frames:
      self.queue.append(cv_image)
    else:
      self.queue.popleft()
      self.queue.append(cv_image)

      print(len(self.queue))

      Xims = np.zeros((1, len(self.queue), shape_r, shape_c, 3))  # change dimensionality
      Xims2 = np.zeros((1, len(self.queue), shape_r, shape_c, 3))  # change dimensionality
      Xims3 = np.zeros((1, len(self.queue), shape_r, shape_c, 3))  # change dimensionality

      [X, X2, X3] = preprocess_images_realtime(self.queue, shape_r, shape_c)
      print(X.shape, "X shape new")
      #   # print("Inside Test Generator ", X.shape)
      Xims[0] = np.copy(X)
      Xims2[0] = np.copy(X2)
      Xims3[0] = np.copy(X3)



      prediction = self.m.predict([Xims,Xims2,Xims3])
      print("Prediction shape: ", prediction.shape)

      for j in range(len(self.queue)):
        orignal_image = self.queue[0]
        print(orignal_image.shape, "Queue shape")
        x, y = divmod(j, len(self.queue))
        print(x, y)


        # cv.imshow("Frame", prediction[0,0,:,:,0] )
        # cv.waitKey(3)

      # self.predict(Xims[0], Xims2[1], Xims3[3], m)
      # print("data predicted")

      # image_data = im.fromarray(prediction[0,0,:,:,0])

      # image_data = self.bridge.imgmsg_to_cv2(prediction[0,0,:,:,0], "bgr8")
      # self.predict(asarray(image_data))
      print(prediction[0,0,:,:,0].shape)
      self.predict(prediction[0,0,:,:,0])

      self.m.reset_states()
      # (rows,cols,channels) = cv_image.shape
      # if cols > 60 and rows > 60 :
      #   cv2.circle(cv_image, ((cols/2)-15,(rows/2)-15), 30, 255)

      #   cv2.imshow("Image window", cv_image)
      #   cv2.waitKey(3)
      #   """ prediction and locomotion command is acquired after this function call """

      #   object_map = self.predict(cv_image)

      #   [X, Y, Z] = self.localize_torsow(object_map)

        
  
  """ Moving object detection and segmentation model """
  def predict(self, image_data):
    """
    load model here  
    predict
    """

    print("Publishable data received")
    predicted_map = image_data
    try:
        self.object_map_pub.publish(self.bridge.cv2_to_imgmsg(predicted_map, "32FC1"))
        print(predicted_map.shape, "Data published")
    except CvBridgeError as e:
        print(e)


  """ calculates x,y,z value of all dof for the robot torsow and publishes both to the
   torsow subscriber and a separate topic called /mods/object_roi 
  """
  def localize_torsow(self, object_map):
    hello_coordinte = "hello [X, Y, Z]%s   this is the coordinate" % rospy.get_time()
    self.localize_pub.publish(hello_coordinte)

    return ["X", "Y", "Z"]


""" entry function for the mods model"""
def main(args):
  mods = moving_object_detection()
  rospy.init_node('dynamic_saliency_prediction', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

""" entry point for the file """
if __name__ == '__main__':
    main(sys.argv)

