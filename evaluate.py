import tensorflow as tf

from environment import get_env
import numpy as np

class evaluation():
    def __init__(self,params,sess):
        self.env = get_env(params,tf.estimator.ModeKeys.EVAL)
        self.params = params
        self.sess = sess

    def eval(self):
        params = self.params
        sess = self.sess
        s = self.env.reset()
        # for _ in range(params.history_size//2):
        #     self.env.step()
        # s = self.env.step()[0]
        step = 0
        total_reward = 0
        total_loss =[]
        #print ("eval env length is {}".format(len(self.env.data)))
        memory = []
        reward_list = []
        max_steps = self.env.data_len()
        while step <max_steps:
            s = (np.array(s, dtype=np.float32).reshape(1, -1))
            a, =sess.run([params.agent.A_main],{params.agent.main_input:s})
            a=a[0]
            s_next,r,done = self.env.step(a)
            memory.append((s,a,r,s_next,done))
            # next step
            total_reward += r
            cmd = "stay"
            if a==1:
                cmd = "buy"
            elif a==2:
                cmd = "sell"
            reward_list.append(cmd+":{}".format(r))
            s = s_next
            step += 1
            if len(memory) < 50:
                continue
            shuffled_memory = np.random.permutation(memory)
            memory_idx = range(len(shuffled_memory))
            for i in memory_idx[::params.batch_size]:
                batch = np.array(shuffled_memory[i:i+params.batch_size])
                s_batch = np.array(batch[:, 0].tolist(), dtype=np.float32).reshape(params.batch_size, -1)
                a_batch = np.array(batch[:, 1].tolist(), dtype=np.int32)
                reward_batch = np.array(batch[:, 2].tolist(), dtype=np.int32)
                s_next_batch = np.array(batch[:, 3].tolist(), dtype=np.float32).reshape(params.batch_size, -1)
                done_batch =1- np.array(batch[:, 4].tolist(), dtype=np.int32)

                a_main, q_object= sess.run([params.agent.A_main,params.agent.Q_main],{params.agent.main_input:s_next_batch})
                q_object = params.gamma*q_object[range(params.batch_size),a_main]*( done_batch)
                doubleQvalue = reward_batch+q_object
                s_loss, = sess.run([params.agent.loss]
                                    ,{params.agent.main_network.targetQ:doubleQvalue
                                        ,params.agent.main_input:s_batch
                                        ,params.agent.main_network.action:a_batch}
                                    )
                total_loss.append(s_loss)
                memory.clear()

        print("eval:\t"+'\t'.join(map(str, ["reward:",total_reward,"loss", np.average(total_loss),"porfits",self.env.profits])))
        print ("reward history is {}".format(reward_list))