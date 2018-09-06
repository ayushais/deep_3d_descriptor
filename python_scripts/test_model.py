from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import operator
import functools
import argparse
import sys
from sys import argv
import h5py
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
from sys import argv
import argparse
import os
FLAGS = None
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/predictions.shape[0])
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--path_to_saved_model")
  parser.add_argument("--path_to_testing_data")

  args = parser.parse_args()
  if args.path_to_saved_model:
    saved_model_file_path = args.path_to_saved_model
  else:
    print('please enter the path to saved model')
    exit()
  if args.path_to_testing_data:
    testing_data_file_path = args.path_to_testing_data
  else:
    print('please enter the path to testing data')
    exit()
  path, filename = os.path.split(saved_model_file_path)

  with tf.Session() as sess:
    image_size = 64
    batch_size = 100
    num_labels = 2 
    num_channels = 2
    model_name = saved_model_file_path + '.ckpt.meta'
    saver = tf.train.import_meta_graph(model_name)
    model_name = saved_model_file_path + '.ckpt'

    saver.restore(sess,model_name)
    positive_combination_test = np.loadtxt('positive_combination_testing.txt',delimiter=",")
    negative_combination_test = np.loadtxt('negative_combination_testing.txt',delimiter=",")
    hdf5_file = h5py.File(testing_data_file_path,  "r")  
    a_group_key_data = hdf5_file.keys()[0]
    data_d_test = list(hdf5_file[a_group_key_data])
    test_data = np.array(data_d_test)
    test_data = np.transpose(test_data,(0,2,3,1))
    a_group_key_label = hdf5_file.keys()[1]
    data_l_test = list(hdf5_file[a_group_key_label])
    test_label_data = np.array(data_l_test)
    print(test_label_data.shape)
    print(test_data.shape)
    hdf5_file.close()
    ctr = 0
    labels_combined_positive = []
    predictions_combined_positive  = []
    labels_combined_negative = []
    predictions_combined_negative  = []
    positive_samples = []
    negatives_samples = []
    ###testing for 50000 matching and non-matching image pairs
    for index in range(0,50000,batch_size):
###positive samples
      sample_left = positive_combination_test[index:index+batch_size,0]
      sample_right = positive_combination_test[index:index+batch_size,1]
      image_left_positive = test_data[np.uint32(sample_left),:,:,:]
      image_right_positive = test_data[np.uint32(sample_right),:,:,:]
      label_left_positive = test_label_data[np.uint32(sample_left),:,:,:]
      label_right_positive = test_label_data[np.uint32(sample_right),:,:,:]
      label_positive = np.uint8(label_right_positive == label_left_positive)
      label_positive = np.concatenate((label_positive,1 - label_positive))
      label_positive = np.reshape(label_positive,(num_labels,batch_size))
      label_positive = np.transpose(label_positive)
###negative samples      
      sample_left = negative_combination_test[index:index+batch_size,0]
      sample_right = negative_combination_test[index:index+batch_size,1]
      image_left_negative = test_data[np.uint32(sample_left),:,:,:]
      image_right_negative = test_data[np.uint32(sample_right),:,:,:]
      label_left_negative = test_label_data[np.uint32(sample_left),:,:,:]
      label_right_negative = test_label_data[np.uint32(sample_right),:,:,:]
      label_negative = np.uint8(label_right_negative == label_left_negative)
      label_negative = np.concatenate((label_negative,1 - label_negative))
      label_negative = np.reshape(label_negative,(num_labels,batch_size))
      label_negative = np.transpose(label_negative)
      
      
      
      label = np.concatenate((label_positive,label_negative))
      
      
      labels_combined_positive.append(label_positive)
      labels_combined_negative.append(label_negative)
      image_left = np.concatenate((image_left_positive,image_left_negative),axis = 0)
      image_right = np.concatenate((image_right_positive,image_right_negative),axis = 0)
      image_left[:,:,:,0:2] -= [4.15,7.55] 
      image_right[:,:,:,0:2] -= [4.15,7.55] 
      graph = tf.get_default_graph()
      input_x1 = graph.get_tensor_by_name("input_x1:0")
      input_x2 = graph.get_tensor_by_name("input_x2:0")
      tf_train_labels = graph.get_tensor_by_name("tf_train_labels:0")
      keep_prob = graph.get_tensor_by_name("keep_prob:0")
      is_training = graph.get_tensor_by_name("is_training:0")
      test_prediction = graph.get_tensor_by_name("test_prediction:0")
      feed_dict = {input_x1:image_left,input_x2:image_right,tf_train_labels:label,keep_prob:1.0,is_training:False}
      prediction_test = sess.run(test_prediction,feed_dict=feed_dict)   
      predictions_combined_positive.append(prediction_test[0:batch_size,0])
      predictions_combined_negative.append(prediction_test[batch_size:2*batch_size,0])
      print('test accuracy: %.1f%%' % accuracy(prediction_test, label))

    labels_combined_positive = np.array(labels_combined_positive)
    labels_combined_negative = np.array(labels_combined_negative)
    predictions_combined_positive = np.array(predictions_combined_positive)
    predictions_combined_negative = np.array(predictions_combined_negative)
    
    labels_combined_positive = np.reshape(labels_combined_positive,(labels_combined_positive.shape[0]*batch_size,2))
    labels_combined_negative = np.reshape(labels_combined_negative,(labels_combined_negative.shape[0]*batch_size,2))
    predictions_combined_positive = np.reshape(predictions_combined_positive,(predictions_combined_positive.shape[0]*batch_size,))
    predictions_combined_negative = np.reshape(predictions_combined_negative,(predictions_combined_negative.shape[0]*batch_size,))
    
    total_label = np.concatenate((labels_combined_positive,labels_combined_negative),axis=0)
    total_label = total_label[:,0]
    total_predictions = np.concatenate((predictions_combined_positive,predictions_combined_negative),axis=0)
    zipped_score = zip(total_predictions,total_label)

####sort the testing data according to the softmax score in increasing to decreasing order
###With the decrease in score FPR increases
    zipped_score.sort(key = lambda t: t[0],reverse=True)
    f_prev = -10000
    FP = 0
    TP = 0
    TP_vec = []
    FP_vec = []
    FP_vec = np.array(FP_vec)
    TP_vec = np.array(TP_vec)
    ##iterate over the score and store true_positive_rate and false_positive_rate for different score thresholds
    for score in zipped_score:
      if(score[0]!=f_prev):
        if(FP_vec.shape == 0):##FPR
          FP_vec = (FP/labels_combined_negative.shape[0])
        else:
          FP_vec = np.append(FP_vec,FP/labels_combined_negative.shape[0])
        if(TP_vec.shape == 0):##TPR
          TP_vec = (TP/labels_combined_positive.shape[0])
        else:
          TP_vec = np.append(TP_vec,TP/labels_combined_positive.shape[0])
        f_prev = score[0]
      if(score[1] == 1):##updating true positives
        TP += 1
      if(score[1] == 0):###updating false positives
        FP += 1
    print(TP_vec.shape)
    print(FP_vec.shape)
    TP_vec = np.reshape(TP_vec,[TP_vec.shape[0],1])
    FP_vec = np.reshape(FP_vec,[FP_vec.shape[0],1])
    min_list = abs(TP_vec - 0.95)## for finding the index closest to TPR at 95%
##    print(min(enumerate(min_list), key=itemgetter(1))[0])
    results = np.concatenate((TP_vec,FP_vec),axis = 1)
####storing TPR and FPR for plotting the curve
    print('fpr_score %f'%FP_vec[min(enumerate(min_list), key=itemgetter(1))[0]])
    save_file = path + filename + '.csv'
    np.savetxt(save_file,results,delimiter=',',fmt='%10.5f')
    plt.figure(2)
    plt.plot(FP_vec,TP_vec,'g')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()

if __name__ == '__main__':
  main()


