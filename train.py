import os
import time

import tensorflow as tf
import numpy as np

from agent import QNetwork, action
from environment import get_env, Experience

tf.app.flags.DEFINE_integer("history_size",90,"")
tf.app.flags.DEFINE_string("data_path","data/Stocks/goog.us.txt","")
tf.app.flags.DEFINE_integer("epoch_num",50,"")
tf.app.flags.DEFINE_integer("memory_size",200,"")
tf.app.flags.DEFINE_integer("batch_size",50,"")
tf.app.flags.DEFINE_string("model_dir","model","")

FLAGS = tf.app.flags.FLAGS

# 利用主网络参数更新目标网络
def updateTargetGraph(tfVars, tau):
    half_len = len(tfVars)//2
    op_holder = []
    for idx, var in enumerate(tfVars[0: half_len]):
        #op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
        op_holder.append(tfVars[idx+half_len].assign(var.value()))
    return op_holder

def updateTarget(sess):
    op_holder = updateTargetGraph(tf.trainable_variables(),0)
    for op in op_holder:
        sess.run(op)


def main(_):
    main_network = QNetwork(FLAGS.history_size,100,3,True,name = "main")
    main_input = tf.placeholder(tf.float32,[None,FLAGS.history_size+1])
    Q_main,A_main = main_network(main_input)

    object_network =  QNetwork(FLAGS.history_size,100,3,True,name = "object")
    object_input = tf.placeholder(tf.float32,[None,FLAGS.history_size+1])
    Q_object,A_object = object_network(object_input)

    targetQ = tf.placeholder(shape=[None,3], dtype=tf.float32)
    td_error = tf.square(targetQ - Q_main)
    loss = tf.reduce_mean(td_error)
    optimizer = tf.train.AdamOptimizer()
    updateModel = optimizer.minimize(loss)


    environment  = get_env(FLAGS)

    step_max = len(environment.data)-1

    train_freq = 10
    update_q_freq = 20
    gamma = 0.97
    show_log_freq = 5

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    memory = []#Experience(FLAGS.memory_size)
        #创建用于保存模型的目录
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)
    start = time.time()
    act = action()
    with tf.Session() as sess:
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt:
            print('Loading Model...')
            saver.restore(sess,ckpt.model_checkpoint_path)
        total_step = 1
        total_rewards = []
        total_losses = []
        for epoch in range(FLAGS.epoch_num):
            s = environment.reset()

            step = 0
            total_reward = 0
            total_loss = 0
            while step < step_max:
                s = (np.array(s, dtype=np.float32).reshape(1, -1))
                [a] =sess.run([A_main],{main_input:s})
                a = act.get_action(total_step,a[0])
                s_next,r,done = environment.step(a)
                memory.append((s,a,r,s_next,done))
                if len(memory) > FLAGS.memory_size:
                    memory.pop(0)
                if total_step % train_freq == 0 and step>FLAGS.memory_size:
                    shuffled_memory = np.random.permutation(memory)
                    memory_idx = range(len(shuffled_memory))
                    for i in memory_idx[::FLAGS.batch_size]:
                        batch = np.array(shuffled_memory[i:i+FLAGS.batch_size])
                        s_batch = np.array(batch[:, 0].tolist(), dtype=np.float32).reshape(FLAGS.batch_size, -1)
                        #s_batch = tf.constant(s_batch)
                        a_batch = np.array(batch[:, 1].tolist(), dtype=np.int32)
                        reward_batch = np.array(batch[:, 2].tolist(), dtype=np.int32)
                        s_next_batch = np.array(batch[:, 3].tolist(), dtype=np.float32).reshape(FLAGS.batch_size, -1)
                        #s_next_batch = tf.constant((s_next_batch))
                        done_batch = np.array(batch[:, 4].tolist(), dtype=np.bool)
                        target, = sess.run([Q_main],{main_input:s_batch})
                        #target = np.copy.deepcopy(q_main.data)
                        a_main, = sess.run([A_main],{main_input:s_next_batch})
                        q_object, = sess.run([Q_object],{object_input:s_next_batch})
                        for j in range(FLAGS.batch_size):
                            p1 = a_batch[j]
                            p2 = a_main[j]
                            target[j, a_batch[j]] = reward_batch[j]+gamma*q_object[j, a_main[j]]*(not done_batch[j])
                        _,s_loss = sess.run([updateModel,loss],{targetQ:target,main_input:s_batch})
                        total_loss+=s_loss
                if total_step % update_q_freq == 0 and step>FLAGS.memory_size: #更新从Q参数
                    updateTarget(sess)
                # next step
                total_reward += r
                s = s_next
                step += 1
                total_step += 1
            total_rewards.append(total_reward)
            total_losses.append(total_loss)

            if (epoch+1) % show_log_freq == 0:
                log_reward = sum(total_rewards[((epoch+1)-show_log_freq):])/show_log_freq
                log_loss = sum(total_losses[((epoch+1)-show_log_freq):])/show_log_freq
                elapsed_time = time.time()-start
                print('\t'.join(map(str, [epoch+1, act.epsilon, total_step, log_reward, log_loss, elapsed_time])))
                start = time.time()


if __name__== "__main__":
    tf.app.run(main)