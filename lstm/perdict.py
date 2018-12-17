import os
import time
import sys
sys.path.insert(0, "D:\\github projects\\linsu07\\rl_dealer\\")
import tensorflow as tf
import numpy as np

from lstm.Action import action
from lstm.env_lstm import get_env
from lstm.evaluate_lstm import evaluation
from lstm.network_lstm import model

tf.app.flags.DEFINE_integer("history_size",60,"")
#goog.us.txt
tf.app.flags.DEFINE_string("data_path","../china_stock/000002.SZ.csv","")
tf.app.flags.DEFINE_integer("epoch_num",60,"")
tf.app.flags.DEFINE_integer("memory_size",100,"")
tf.app.flags.DEFINE_integer("batch_size",50,"")
tf.app.flags.DEFINE_string("model_dir","model","")
tf.app.flags.DEFINE_integer("head_size",3,"")

FLAGS =  tf.app.flags.FLAGS

def main(_):
    FLAGS.agent = model(params=FLAGS)
    FLAGS.act = action()

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        eval = evaluation(FLAGS,sess)
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir )
        if ckpt:
            print('Loading Model...')
            #'model-33.ckpt'
            #saver.restore(sess,ckpt.model_checkpoint_path)
            saver.restore(sess,"model\\model-6.ckpt")

            eval.eval_pic()

if __name__== "__main__":
    tf.app.run(main)