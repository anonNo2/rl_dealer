import os
import time
import sys
sys.path.insert(0, "D:\\github projects\\linsu07\\rl_dealer\\")
import tensorflow as tf
import numpy as np

from lstm.agent import action
from lstm.env_lstm import get_env
from lstm.evaluate_lstm import evaluation
from lstm.network_lstm import model

tf.app.flags.DEFINE_integer("history_size",90,"")
#goog.us.txt
tf.app.flags.DEFINE_string("data_path","../china_stock/000002.SZ.csv","")
tf.app.flags.DEFINE_integer("epoch_num",60,"")
tf.app.flags.DEFINE_integer("memory_size",100,"")
tf.app.flags.DEFINE_integer("batch_size",50,"")
tf.app.flags.DEFINE_string("model_dir","model","")

FLAGS =  tf.app.flags.FLAGS

# 利用主网络参数更新目标网络
def updateTargetGraph(tfVars, tau=0.01):
    half_len = len(tfVars)//2
    op_holder = []
    for idx, var in enumerate(tfVars[0: half_len]):
        op_holder.append(tfVars[idx+half_len].assign((var.value()*tau) + ((1-tau)*tfVars[idx+half_len].value())))
        #op_holder.append(tfVars[idx+half_len].assign(var.value()))
    return op_holder

def updateTarget(sess):
    op_holder = updateTargetGraph(tf.trainable_variables(),0.001)
    for op in op_holder:
        sess.run(op)

def run_epch(params,sess,total_step):
    head,history = params.environment.reset()
    step = 0
    total_reward = 0
    total_loss =[]
    while step < params.step_max:
        head_reshape = (np.array(head, dtype=np.float32).reshape(1, -1))
        history_reshape = (np.array(history, dtype=np.float32).reshape(1, params.history_size,5))
        a, =sess.run([params.agent.Q_main],{params.agent.main_head:head_reshape,params.agent.main_history:history_reshape})
        a = params.act.get_action(total_step,a[0],params.environment)
        s_next_head,s_next_history,r,done = params.environment.step(a)
        params.memory.append((head,history,a,r,s_next_head,s_next_history,done))
        while len(params.memory) > params.memory_size:
            params.memory.pop(0)
        if total_step % params.train_freq == 0 and step>params.memory_size+params.history_size:
            shuffled_memory = np.random.permutation(params.memory)
            memory_idx = range(len(shuffled_memory))
            for i in memory_idx[::FLAGS.batch_size]:
                batch = np.array(shuffled_memory[i:i+params.batch_size])
                s_batch_head =np.array( batch[:, 0].tolist(), dtype=np.float32)
                s_batch_history = np.array( batch[:, 1].tolist(), dtype=np.float32)
                a_batch = np.array(batch[:, 2].tolist(), dtype=np.int32)
                reward_batch = np.array(batch[:, 3].tolist(), dtype=np.int32)
                s_next_batch_head = np.array(batch[:, 4].tolist(), dtype=np.float32)#.reshape(params.batch_size, -1)
                s_next_batch_history = np.array(batch[:, 5].tolist(), dtype=np.float32)
                done_batch =1- np.array(batch[:, 6].tolist(), dtype=np.int32)


                a_main, = sess.run([params.agent.A_main],{params.agent.main_head:s_next_batch_head,params.agent.main_history:s_next_batch_history})
                q_object, = sess.run([params.agent.Q_object],{params.agent.object_head:s_next_batch_head,params.agent.object_history:s_next_batch_history})
                q_object = params.gamma*q_object[range(params.batch_size),a_main]*( done_batch)
                doubleQvalue = reward_batch+q_object
                params.agent.main_network.scalar_step+=1
                _,s_loss = sess.run([params.agent.updateModel,params.agent.loss]
                                    ,{params.agent.main_network.targetQ:doubleQvalue
                                    ,params.agent.main_head:s_batch_head
                                    ,params.agent.main_history:s_batch_history
                                    ,params.agent.main_network.action:a_batch
                                    ,params.agent.main_network.step:params.agent.main_network.scalar_step}
                                    )
                total_loss.append(s_loss)
        if total_step % params.update_q_freq == 0 and step>FLAGS.memory_size: #更新从Q参数
            updateTarget(sess)
        # next step
        total_reward += r
        head = s_next_head
        history = s_next_history
        step += 1
        total_step += 1
    return np.average(total_loss),total_reward,total_step,params.environment.profits


def main(_):
    FLAGS.agent = model(params=FLAGS)
    FLAGS.environment  = get_env(FLAGS)
    FLAGS.act = action()

    FLAGS.step_max = FLAGS.environment.data_len()
    FLAGS.train_freq = 80
    FLAGS.update_q_freq = 100
    FLAGS.gamma = 0.97
    FLAGS.show_log_freq = 3
    FLAGS.memory = []#Experience(FLAGS.memory_size)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

        #创建用于保存模型的目录
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)
    start = time.time()

    with tf.Session() as sess:
        sess.run(init)
        eval = evaluation(FLAGS,sess)
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt:
            print('Loading Model...')
            saver.restore(sess,ckpt.model_checkpoint_path)
        total_step = 1
        print('\t'.join(map(str, ["epoch", "epsilon", "total_step", "rewardPerEpoch", "profits", "lossPerBatch", "elapsed_time"])))
        for epoch in range(FLAGS.epoch_num):
            avg_loss_per_batch,total_reward,total_step,profits = run_epch(FLAGS,sess,total_step)
            # total_rewards.append(total_reward)
            # total_losses.append(total_loss)

            if (epoch+1) % FLAGS.show_log_freq == 0:
                # log_reward = sum(total_rewards[((epoch+1)-FLAGS.show_log_freq):])/FLAGS.show_log_freq
                # log_loss = sum(total_losses[((epoch+1)-FLAGS.show_log_freq):])/FLAGS.show_log_freq
                elapsed_time = time.time()-start
                #print('\t'.join(map(str, [epoch+1, FLAGS.act.epsilon, total_step, log_reward, log_loss, elapsed_time])))
                print('\t'.join(map(str, [epoch+1, FLAGS.act.epsilon, total_step, total_reward, profits, avg_loss_per_batch, elapsed_time])))
                start = time.time()

                saver.save(sess,FLAGS.model_dir+'\model-'+str(epoch+1)+'.ckpt')
                eval.eval()




if __name__== "__main__":
    tf.app.run(main)