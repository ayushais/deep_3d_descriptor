from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import operator
import functools
import argparse
import sys
from sys import argv
import h5py
import numpy as np
import random
import time
import cv2
from scipy import sparse
import matplotlib.pyplot as plt
import tensorflow as tf
from sys import argv

class Siamese:
  def __init__ (self,filename,training_file_path,test_file_path):
    self.batch_size = []
    self.epochs = []
    self.lr = []
    self.bneck_size = []
    self.fc_size = []

    self.eta = []
    self.growth = []
#### file to read parameters listed below
    
    f = open('parameters.txt', 'r')
    filename_save  = "results/" + filename +"_params.txt"
    f_write = open(filename_save, 'w')
    for line in f:
      f_write.write(line)
      line = line.split(" ")
      if(line[0] == "batch_size"):
        self.batch_size = int(line[1])
      if(line[0] == "epochs"):
        self.epochs = int(line[1])
      if(line[0] == "learning_rate"):
        self.base_lr = float(line[1])
      if(line[0] == "bneck_size"):
        self.bneck_size = int(line[1])
      if(line[0] == "fc_size"):
        self.fc_size = int(line[1])
      if(line[0] == "eta"):
        self.eta = float(line[1])
      if(line[0] == "growth"):
        self.growth = float(line[1])
    f_write.close()
    f.close()
      
    self.num_labels = 2  ## matching and non-matching 
    self.image_size = 64 
    self.num_channels = 2
    self.keep_prob = tf.placeholder(tf.float32,name="keep_prob")##for dropout
    self.lr = tf.placeholder(tf.float32,name="lr")##for learning rate
      
### load the file listing positive combinations      
    self.positive_combination = np.loadtxt('positive_combination_train_3d_patches.txt',delimiter=",")
    random.shuffle(self.positive_combination)

### load the file listing positive combinations      
    self.negative_combination = np.loadtxt('negative_combination_train_3d_patches.txt',delimiter=",")
    random.shuffle(self.negative_combination)
      ###load the training hdf5 file (need to get rid of the hardcoded path)
    hdf5_file = h5py.File(training_file_path,  "r")  
    a_group_key_data = hdf5_file.keys()[0]
    data_d = list(hdf5_file[a_group_key_data])
    self.train_data = np.array(data_d)
    self.train_data = np.transpose(self.train_data,(0,2,3,1))
    a_group_key_label = hdf5_file.keys()[1]
    data_l = list(hdf5_file[a_group_key_label])
    self.train_label_data = np.array(data_l)
    hdf5_file.close()


    print('training data loaded')
    print(self.train_data.shape)
  ###calculate iterations required for epochs      
    self.iter_per_epochs = self.negative_combination.shape[0]/self.batch_size
     

###load the testing hdf5 file (need to get rid of the hardcoded path)
    self.positive_combination_test = np.loadtxt('positive_combination_test_3d_patches_test.txt',delimiter=",")
    random.shuffle(self.positive_combination_test)
    self.negative_combination_test = np.loadtxt('negative_combination_test_3d_patches_test.txt',delimiter=",")
    random.shuffle(self.negative_combination_test)
    hdf5_file = h5py.File(test_file_path,  "r")  
    a_group_key_data = hdf5_file.keys()[0]
    data_d_test = list(hdf5_file[a_group_key_data])
    self.test_data = np.array(data_d_test)
    self.test_data = np.transpose(self.test_data,(0,2,3,1))
    a_group_key_label = hdf5_file.keys()[1]
    data_l_test = list(hdf5_file[a_group_key_label])
    self.test_label_data = np.array(data_l_test)
    
    hdf5_file.close()
  
    print('testing data loaded')
  
    print(self.test_data.shape)

###placeholders for bneck for both streams            
    self.bneck_1 = tf.placeholder(tf.float32, shape=[None,None],name="bneck_1")
    self.bneck_2 = tf.placeholder(tf.float32, shape=[None,None],name="bneck_2")
    self.is_training = tf.placeholder(tf.bool,name="is_training")
### placeholder for input for first stream 
    self.x1 = tf.placeholder(
    tf.float32, shape=[None, self.image_size, self.image_size, self.num_channels],name="input_x1")

### placeholder for input for second stream 
    self.x2 = tf.placeholder(
    tf.float32, shape=[None, self.image_size, self.image_size, self.num_channels],name="input_x2")
### place holder for labels used in one iteration
    self.tf_train_labels = tf.placeholder(
    tf.float32, shape=[None, self.num_labels],name="tf_train_labels")
###define the model     
    with tf.variable_scope("siamese") as scope:
      self.bneck_1 = self.dense_model(self.x1)
      scope.reuse_variables()
      self.bneck_2 = self.dense_model(self.x2)
### model for metric learning, need to fix the name        
    self.fc_5 = self.metric_model()
    self.loss_value = self.loss_function_cross_entropy()
    self.train_step = self.optimize_function()

###function for bias initialization   
    def bias_variable(self,shape):
      initial = tf.constant(0.1, shape=shape,name="bias")
      return tf.Variable(initial)
    def avg_pool(self,input, s):
      return(tf.nn.avg_pool(input, [ 1, s, s, 1 ], [1, s, s, 1 ], 'VALID'))

#### function defining convolution layer

    def conv_layer(self,bottom,output,k_size_h,k_size_w,stride_h,stride_v,pad,name,is_activation=True):
      with tf.variable_scope(name):
        W_conv= tf.get_variable("W_conv", shape=[k_size_h,k_size_w,bottom.shape.as_list()[3],output],
         initializer=tf.contrib.layers.xavier_initializer())
        b_conv = self.bias_variable([output])
        conv = tf.nn.conv2d(bottom,W_conv,strides=[1, stride_v, stride_h, 1], padding=pad)
        if(is_activation):
          return(tf.nn.relu(conv + b_conv))
        else:
          return(tf.nn.bias_add(conv, b_conv))
###fully connected
    def fc_layer(self,bottom,output,name,is_activation=True):
      with tf.variable_scope(name):
        input_shape = bottom.shape.as_list()
        input_shape_flattened = functools.reduce(operator.mul,input_shape[1:len(input_shape)], 1)
        W_fc = tf.get_variable("W_fc", shape=[input_shape_flattened,output],
            initializer=tf.contrib.layers.xavier_initializer())
        b_fc = self.bias_variable([output])
        x = tf.reshape(bottom, [-1, input_shape_flattened])
        if(is_activation):
          fc = tf.nn.relu(tf.matmul(x, W_fc) + b_fc,name=name)
        else:
          fc = tf.nn.bias_add(tf.matmul(x, W_fc), b_fc)
        return(fc)
##max pooling
    def add_layer(self, prev_layer, in_features, out_features, name):
      print ('adding layer, prev_layer has shape:', prev_layer.get_shape())
      current_layer = tf.contrib.layers.batch_norm(prev_layer, scale=True, is_training=self.is_training, updates_collections=None)
      current_layer = tf.nn.relu(current_layer)
      current_layer = self.conv_layer(current_layer,out_features, 3,3,1,1,'SAME',name,False)
      current_layer = tf.nn.dropout(current_layer, self.keep_prob)
      return current_layer

    def add_block(self, scope_name, prev_layer, num_layers, in_features, growth):
      features = in_features
      stack = prev_layer
      db_output = []
      db_output = np.array(db_output)
      with tf.variable_scope(scope_name) as scope:
        for idx in xrange(num_layers):
          current_layer = self.add_layer(stack, features, growth, scope_name+ `idx` + 'W')
          if(db_output.shape[0] == 0):
            db_output = current_layer
          else:  
            db_output = tf.concat([db_output,current_layer],axis=3)

          stack = tf.concat([stack, current_layer],axis=3)
          print(stack.shape)
          features += growth
        return stack,features,db_output
    def contrastive_loss(self):
      d = tf.reduce_sum(tf.square(self.bneck_1 - self.bneck_2),1)
      self.d_sqrt = tf.sqrt(d)
      margin = 0.8
      self.loss = tf.multiply(self.tf_train_labels[:,1],tf.maximum(0.,margin - self.d_sqrt)) + \
              tf.multiply(self.tf_train_labels[:,0],self.d_sqrt)
      return(tf.reduce_mean(self.loss))

##loss function with l2 weight regularization
    def loss_function_cross_entropy(self):
      vars_= tf.trainable_variables()
      l2loss_W = tf.add_n([tf.nn.l2_loss(v) for v in vars_ if len(v.get_shape().as_list()) > 1])
      return(tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=self.tf_train_labels, logits=self.fc_5)) + (self.eta * l2loss_W))
    def optimize_function(self):
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
          return(tf.train.AdamOptimizer(self.lr).minimize(self.loss_value))
####metric learning model
    def metric_model(self,feature_1,feature_2):
       ##concatenation of two streams of siamese 
      bneck_combined = tf.concat((feature_1,feature_2),axis=1) 
      fc_1 = self.fc_layer(bneck_combined,self.fc_size,"fc_1")
      fc_2 = self.fc_layer(fc_1,512,"fc_2")
      fc_3 = self.fc_layer(fc_2,self.fc_size/2,"fc_3")
      fc_4 = self.fc_layer(fc_3,self.fc_size/4,"fc_4")
      fc_5 = self.fc_layer(fc_4,2,"fc_5",False)
      return(fc_3)
    def dense_model(self,data):
      conv_0 = self.conv_layer(data,16,3,3,1,1,'SAME',"conv_0")
      conv_1 = self.conv_layer(conv_0,16,3,3,1,1,'SAME',"conv_1")
      transition = self.avg_pool(conv_1,2)
      dense_block,features,db_output = self.add_block("dense_block_1", transition, 2,16,self.growth)
      transition = self.avg_pool(dense_block,2)
      dense_block,features,db_output = self.add_block("dense_block_2", transition,2,features,self.growth)
      transition = self.avg_pool(dense_block,2)
      conv_2 = self.conv_layer(transition,4,1,1,1,1,'SAME',"conv_2")
#      transition = self.add_transition("transition_3", dense_block, features, features)
#      transition = tf.reshape(transition,(tf.shape(transition)[0],int(transition.shape[1]) * int(transition.shape[2]) * int(transition.shape[3])))
#      bottleneck = tf.contrib.layers.fully_connected(transition,256,scope="bottleneck")
#        bottleneck = self.fc_layer(transition,256,"bottleneck")
#         print(self.fc.shape)
      input_shape = conv_2.shape.as_list()
      input_shape_flattened = functools.reduce(operator.mul,input_shape[1:len(input_shape)], 1)
      bottleneck = tf.reshape(conv_2,[tf.shape(conv_2)[0],input_shape_flattened],name="bottleneck")
      return(bottleneck)
    def test_prediction(self):
      return(tf.nn.softmax(self.fc_5,name="test_prediction"))
      
 
###function to load samples from test data     
    def load_test_batch(self,batch_size):

##pick some random positive sample        
      positive_sample = random.sample(self.positive_combination_test,batch_size)
      positive_sample = np.array(positive_sample)

##pick some random negative sample        
      negative_sample = random.sample(self.negative_combination_test,batch_size)
      negative_sample = np.array(negative_sample)
###load the samples from the hdf5 file      
      sample_left = positive_sample[:,0]
      sample_right = positive_sample[:,1]
      image_left_positive = self.test_data[np.uint32(sample_left),:,:,:]
      image_right_positive = self.test_data[np.uint32(sample_right),:,:,:]
      label_left_positive = self.test_label_data[np.uint32(sample_left),:,:,:]
      label_right_positive = self.test_label_data[np.uint32(sample_right),:,:,:]
      label_positive = np.uint8(label_right_positive == label_left_positive)
      label_positive = np.concatenate((label_positive,1 - label_positive))
      label_positive = np.reshape(label_positive,(self.num_labels,batch_size))
      label_positive = np.transpose(label_positive)

      image_left_negative = self.test_data[np.uint32(sample_left),:,:,:]
      image_right_negative = self.test_data[np.uint32(sample_right),:,:,:]
      label_left_negative = self.test_label_data[np.uint32(sample_left),:,:,:]
      label_right_negative = self.test_label_data[np.uint32(sample_right),:,:,:]
      label_negative = np.uint8(label_right_negative == label_left_negative)
      label_negative = np.concatenate((label_negative,1 - label_negative))
      label_negative = np.reshape(label_negative,(self.num_labels,batch_size))
      label_negative = np.transpose(label_negative)
     
     
      label = np.concatenate((label_positive,label_negative))
###concatenate positive samples with negative samples in each stream of the siamese network      
      image_left = np.concatenate((image_left_positive,image_left_negative),axis = 0)
      image_right = np.concatenate((image_right_positive,image_right_negative),axis = 0)
      
      
###subtract mean      
      normalize_subtract = np.ones((2 * batch_size,self.image_size,self.image_size,self.num_channels),dtype=np.float)
      normalize_subtract[:,:,:,0] = 4.15
      normalize_subtract[:,:,:,1] = 7.55

      image_left = np.subtract(image_left,normalize_subtract)
      
      image_right = np.subtract(image_right,normalize_subtract)
      return image_left, image_right, label
      
###load training data      
### input to the network is a batch of equal number of positive and negative examples    
    def load_batch(self,step):

###estimate offset to get the index from which new data has to retreived        
      offset_positive = (step * int(self.batch_size/2)) % (self.positive_combination.shape[0] - int(self.batch_size/2))
      offset_negative = (step * int(self.batch_size/2)) % (self.negative_combination.shape[0] - int(self.batch_size/2))

#### load a batch of positive samples 
      sample_left = self.positive_combination[offset_positive:offset_positive+int(self.batch_size/2),0]
      sample_right = self.positive_combination[offset_positive:offset_positive+int(self.batch_size/2),1]

      image_left_positive = self.train_data[np.uint32(sample_left),:,:,:]
      image_right_positive = self.train_data[np.uint32(sample_right),:,:,:]


      label_left_positive = self.train_label_data[np.uint32(sample_left),:,:,:]
      label_right_positive = self.train_label_data[np.uint32(sample_right),:,:,:]
      label_positive = np.uint8(label_right_positive == label_left_positive)
      label_positive = np.concatenate((label_positive,1 - label_positive))
      label_positive = np.reshape(label_positive,(self.num_labels,int(self.batch_size/2)))
      label_positive = np.transpose(label_positive)

#### load a batch of negative samples 
      sample_left = self.negative_combination[offset_negative:offset_negative+int(self.batch_size/2),0]
      sample_right = self.negative_combination[offset_negative:offset_negative+int(self.batch_size/2),1]

      image_left_negative = self.train_data[np.uint32(sample_left),:,:,:]
      image_right_negative = self.train_data[np.uint32(sample_right),:,:,:]
      label_left_negative = self.train_label_data[np.uint32(sample_left),:,:,:]
      label_right_negative = self.train_label_data[np.uint32(sample_right),:,:,:]
      
      label_negative = np.uint8(label_right_negative == label_left_negative)
      label_negative = np.concatenate((label_negative,1 - label_negative))
      label_negative = np.reshape(label_negative,(self.num_labels,int(self.batch_size/2)))
      label_negative = np.transpose(label_negative)
###concatenate positive and negative labels
      label = np.concatenate((label_positive,label_negative))
###shuffle positive and negative samples in the batch
      index_shuffle = np.arange(0,self.batch_size)
      random.shuffle(index_shuffle)
      label = label[np.uint8(index_shuffle),:]
      image_left = np.concatenate((image_left_positive,image_left_negative),axis = 0)
      image_right = np.concatenate((image_right_positive,image_right_negative),axis = 0)
      image_left = image_left[np.uint8(index_shuffle),:,:,:]
      image_right = image_right[np.uint8(index_shuffle),:,:,:]
###subtract mean      
      normalize_subtract = np.ones((self.batch_size,self.image_size,self.image_size,self.num_channels),dtype=np.float)
      normalize_subtract[:,:,:,0] = 4.15
      normalize_subtract[:,:,:,1] = 7.55

      image_left = np.subtract(image_left,normalize_subtract)
      
      image_right = np.subtract(image_right,normalize_subtract)
      return image_left, image_right, label




def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/predictions.shape[0])

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_name")
  parser.add_argument("--path_to_training_data")
  parser.add_argument("--path_to_testing_data")
  args = parser.parse_args()
  if args.model_name:
    model_name = args.model_name
  if args.path_to_training_data:
    training_data_file_path = args.path_to_training_data
  if args.path_to_testing_data:
    testing_data_file_path = args.path_to_testing_data
  siamese_object = Siamese(model_name,training_data_file_path,testing_data_file_path)
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  vars_ = tf.global_variables()

 #  model_name = '/home/dewan/code/descriptor_learning/python_scripts/models/another_training_before_release_110062.ckpt'
  # new_saver = tf.train.Saver(vars_)
  # new_saver.restore(sess,model_name)

  loss_val_train = []
  loss_val_test = []
  accuracy_train = []
  accuracy_test = []
  ###estimate total number of iterations for training for the given epochs
  total_iteration = int(siamese_object.epochs * siamese_object.iter_per_epochs)

  print(total_iteration)
  model_iteration = int((siamese_object.epochs/2) * siamese_object.iter_per_epochs)
  print(model_iteration)
  saver = tf.train.Saver()

  for num_iteration in range(total_iteration+1):
    image_left, image_right, label = siamese_object.load_batch(num_iteration)
    
    feed_dict = {siamese_object.x1:image_left,siamese_object.x2:image_right,siamese_object.tf_train_labels:label,siamese_object.keep_prob:1.0,siamese_object.lr:siamese_object.base_lr,siamese_object.is_training:True}
    _,l,lr,loss = sess.run([siamese_object.train_step,siamese_object.loss_value,siamese_object.lr,siamese_object.loss_value],feed_dict=feed_dict)

    if(num_iteration % 1000 == 0):
      minibatch_accuracy = accuracy(predictions, label)
      print("loss_value %f,%d" % (l,num_iteration))
      accuracy_train.append(minibatch_accuracy)
      loss_val_train.append(l)
      image_left_test,image_right_test,label_test = siamese_object.load_test_batch(100)
      feed_dict = {siamese_object.x1:image_left_test,siamese_object.keep_prob:1.0,siamese_object.is_training:False}
      feature_1 = sess.run([siamese_object.bneck_1],feed_dict=feed_dict)
      feed_dict = {siamese_object.x2:image_right_test,siamese_object.keep_prob:1.0,siamese_object.is_training:False}
      feature_2 = sess.run([siamese_object.bneck_2],feed_dict=feed_dict)
      feature_1 = np.array(feature_1)
      feature_2 = np.array(feature_2)
      feature_1 = np.reshape(feature_1,(feature_1.shape[1],feature_1.shape[2]))
      feature_2 = np.reshape(feature_2,(feature_2.shape[1],feature_2.shape[2]))
      feed_dict = {siamese_object.bneck_1:feature_1,siamese_object.bneck_2:feature_2,siamese_object.tf_train_labels:label_test,siamese_object.keep_prob:1.0,siamese_object.is_training:False}
      l,test_predictions = sess.run([siamese_object.loss_value,siamese_object.test_prediction()],feed_dict=feed_dict)
      loss_val_test.append(l)
      test_accuracy = accuracy(test_predictions, label_test)
      accuracy_test.append(test_accuracy)
      print("test loss_value %f,%d" % (l,num_iteration))
      print('test accuracy: %.1f%%' % test_accuracy)
    
###fix the model path
      model_name = 'models/' + filename + '_' + str(num_iteration) + '.ckpt'
      saver.save(sess, model_name)


  loss_val_train = np.array(loss_val_train)
  filename_save = "results/" + filename + "_train_loss.txt"
  np.savetxt(filename_save,loss_val_train,fmt="%10.5f")
  
  loss_val_test = np.array(loss_val_test)
  filename_save = "results/" + filename + "_test_loss.txt"
  np.savetxt(filename_save,loss_val_test,fmt="%10.5f")
  
  accuracy_train = np.array(accuracy_train)
  filename_save = "results/" + filename + "_train_accuracy.txt"
  np.savetxt(filename_save,accuracy_train,fmt="%10.5f")

  accuracy_test = np.array(accuracy_test)
  filename_save = "results/" + filename + "_test_accuracy.txt"
  np.savetxt(filename_save,accuracy_test,fmt="%10.5f")
 





###random sample from +ve and -ve combination


if __name__ == '__main__':
  main()


