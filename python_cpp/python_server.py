  # This file is part of deep_3d_descriptor.
 
  # Copyright (C) 2018 Ayush Dewan (University of Freiburg)
 
  # deep_3d_descriptor is free software: you can redistribute it and/or modify
  # it under the terms of the GNU General Public License as published by
  # the Free Software Foundation, either version 3 of the License, or
  # (at your option) any later version.
 
  # deep_3d_descriptor is distributed in the hope that it will be useful,
  # but WITHOUT ANY WARRANTY; without even the implied warranty of
  # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  # GNU General Public License for more details.
 
  # You should have received a copy of the GNU General Public License
  # along with deep_3d_descriptor.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('gen-py')
from python_cpp import get_descriptors
from python_cpp.ttypes import *
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
import operator
import functools
import argparse
import sys
from sys import argv
import random
import numpy as np
import numpy.matlib
import cv2
import tensorflow as tf
from sys import argv
import multiprocessing
import time
import socket
class InferenceNet:
  def __init__ (self,bneck_size,graph):
    self.graph = []
    self.bneck_1 = tf.placeholder(tf.float32, shape=[None,bneck_size],name="bneck_1")
    self.bneck_2 = tf.placeholder(tf.float32, shape=[None,bneck_size],name="bneck_2")
    self.fc_size = 1024
    self.graph = graph
    self.fc3 = self.metric_model(self.bneck_1,self.bneck_2)  

  def test_prediction(self):
      return(tf.nn.softmax(self.fc3,name="test_prediction"))
    
  def fc_layer(self,bottom,output,name):
    with tf.variable_scope(name):
      input_shape = bottom.shape.as_list()
      input_shape_flattened = functools.reduce(operator.mul,input_shape[1:len(input_shape)], 1)
      weight_name = name + "/W_fc:0"
      W_fc = self.graph.get_tensor_by_name(weight_name)
      bias_name = name + "/bias:0"
      b_fc = self.graph.get_tensor_by_name(bias_name)
      x = tf.reshape(bottom, [-1, input_shape_flattened])
      fc = tf.nn.relu(tf.matmul(x, W_fc) + b_fc,name=name)
      return(fc)
    
  def inference_layer(self,bottom,output,name):
    with tf.variable_scope(name):
      input_shape = bottom.shape.as_list()
      input_shape_flattened = functools.reduce(operator.mul,input_shape[1:len(input_shape)], 1)
      weight_name = name + "/W_fc:0"
      W_fc = self.graph.get_tensor_by_name(weight_name)
      bias_name = name + "/bias:0"
      b_fc = self.graph.get_tensor_by_name(bias_name)
      x = tf.reshape(bottom, [-1, input_shape_flattened])
      fc = tf.nn.bias_add(tf.matmul(x, W_fc), b_fc)
      return(fc)

  def metric_model(self,feature_1,feature_2):
      self.bneck_combined = tf.concat((feature_1,feature_2),axis=1)
      self.fc1 = self.fc_layer(self.bneck_combined,self.fc_size,"fc1")
      self.fc2 = self.fc_layer(self.fc1,self.fc_size/2,"fc2")
      self.fc4 = self.fc_layer(self.fc2,self.fc_size/2,"fc4")
      self.fc5 = self.fc_layer(self.fc4,self.fc_size/4,"fc5")
      self.fc3 = self.inference_layer(self.fc5,2,"fc3")
      return(self.fc3)
  def bias_variable(self,shape):
      initial = tf.constant(0.1, shape=shape,name="bias")
      return tf.Variable(initial)
class GetDescriptorsHandler:
  def __init__(self,model,is_hinge_loss):
    self.log = {}
    self.session = tf.Session()
    model_graph = model + '.ckpt.meta'
    new_saver = tf.train.import_meta_graph(model_graph)
    new_saver.restore(self.session,model+'.ckpt')
    self.graph = tf.get_default_graph()
    self.patch_size = 64
    self.feature_size = 256
    self.is_hinge_loss = int(is_hinge_loss)

    print('model_loaded')
    self.inference_object = InferenceNet(self.feature_size,self.graph)
 
  def match_descriptors(self,feature_1,feature_2):
    print('finding correspondences')
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    feature_1 = np.reshape(feature_1,[-1,self.feature_size])
    feature_2 = np.reshape(feature_2,[-1,self.feature_size])
    for index_left in range(0,feature_1.shape[0]):
      feature_1_element = feature_1[index_left,:]
      store_prediction = []
      store_label = []
      feature_1_rep = np.matlib.repmat(feature_1_element,feature_2.shape[0],1)
      feed_dict = {self.inference_object.bneck_1:feature_1_rep,self.inference_object.bneck_2:feature_2}
      prediction_test = self.session.run(self.inference_object.fc3,feed_dict=feed_dict) 
      index_match = np.argmax(prediction_test,axis=0)[0]
      index_query = index_left
      if(index_left == 0):
        corresponding_points = np.array([index_match])
      else:
        corresponding_points = np.vstack((corresponding_points,[index_match]))
    corresponding_points = np.reshape(corresponding_points,[-1])
    print(corresponding_points.shape)
    return(corresponding_points) 
  def return_descriptors(self,input_patch_vector):

    input_patch_vector = np.array(input_patch_vector)
    number_patches = np.int(input_patch_vector.shape[0]/(self.patch_size*self.patch_size*2))
    input_patches = np.zeros([number_patches,self.patch_size,self.patch_size,2],np.float64)
    patch_size = self.patch_size * self.patch_size
    print("start with patches")
    for patch in range (0,number_patches):
      for channel in range(0,2):
        start_index = (2 * patch) + channel
        input_patches[patch,:,:,channel] = np.reshape(input_patch_vector[start_index * patch_size:(start_index * patch_size) +
          patch_size],[self.patch_size,self.patch_size])

    print("end with patches")
    input_x1 = self.graph.get_tensor_by_name("input_x1:0")
    keep_prob = self.graph.get_tensor_by_name("keep_prob:0")
    is_training = self.graph.get_tensor_by_name("is_training:0")

    if(self.is_hinge_loss == 1):
      bottleneck =  self.graph.get_tensor_by_name("siamese/bottleneck/Relu:0")
    else:
      bottleneck =  self.graph.get_tensor_by_name("siamese/bottleneck:0")
    step_size = 512

    feature = np.zeros((number_patches,self.feature_size),np.float64)
    for index_patch in range(0,number_patches,step_size):
      if (index_patch + step_size) < number_patches:
        step = step_size
      else:
        step = number_patches - index_patch
      input_patches_batch = input_patches[index_patch:index_patch+step,:,:,:]
      input_patches_batch = np.reshape(input_patches_batch,(step,self.patch_size,self.patch_size,2))
      normalize_subtract = np.ones((step,self.patch_size,self.patch_size,2),dtype=np.float)
      normalize_subtract[:,:,:,0] = 4.15
      normalize_subtract[:,:,:,1] = 7.55
      input_patches_batch = np.subtract(input_patches_batch,normalize_subtract)
      feed_dict = {input_x1:input_patches_batch,keep_prob: 1.0,is_training:True}

      feature_1 = self.session.run(bottleneck,feed_dict=feed_dict)
      feature[index_patch:index_patch + step,:] = feature_1

    print(feature.shape)
    return(np.reshape(feature,[-1]))
def main():

  parser = argparse.ArgumentParser()
  parser.add_argument("--model_name")
  parser.add_argument("--use_hinge_loss")
  args = parser.parse_args()
  if args.model_name:
    model = args.model_name
  if args.use_hinge_loss:
    is_hinge_loss = args.use_hinge_loss


  handler = GetDescriptorsHandler(model,is_hinge_loss)
  processor = get_descriptors.Processor(handler)
  transport = TSocket.TServerSocket('localhost',9090)
  tfactory = TTransport.TBufferedTransportFactory()
  pfactory = TBinaryProtocol.TBinaryProtocolFactory()
  import logging
  logging.basicConfig(level=logging.DEBUG)
  server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
  server.serve()

if __name__ == '__main__':
  main()


