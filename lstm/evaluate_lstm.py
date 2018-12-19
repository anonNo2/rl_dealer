import tensorflow as tf

from lstm.environment import get_env
import numpy as np
import matplotlib.pyplot as plt

class evaluation():
    def __init__(self,params,sess=None):
        self.env = get_env(params ,mode = tf.estimator.ModeKeys.EVAL)
        self.params = params
        self.sess = sess


    def eval_pic(self):
        params = self.params
        sess = self.sess
        head,history = self.env.reset()
        print(" in eval_pic")
        max_steps = self.env.data_len()
        step = 0
        buy_step_list =[]
        buy_list = []
        sell_step_list = []
        sell_list = []
        stay_step_list = []
        stay_list = []
        total_reward = 0
        while step <max_steps-1:
            if step<params.history_size:
                head,history,_,_ = self.env.step(0)
                step+=1
                continue


            head_reshape = (np.array(head, dtype=np.float32).reshape(1, params.head_size))
            history_reshape = (np.array(history, dtype=np.float32).reshape(1, params.history_size,6))
            a,q =sess.run([params.agent.A_main,params.agent.Q_main],{params.agent.main_head:head_reshape,params.agent.main_history:history_reshape})

            a = a[0]
            q = q[0]
            confidence =np.exp(q[a])/(np.exp(q[0])+np.exp(q[1]))

            is_sell =  a==1 and  self.env.positions==1
            if is_sell and confidence<0.51:
                a = 0 # 不卖了

            is_buy = a==1 and self.env.positions==0
            if is_buy and confidence<0.51:
                a = 0 #不买了

            if self.env.inblock():
                a = 0
                #print ("in block")

            if self.env.isOutRange():
                a = 1
                #print ("in isOutRange")



            is_buy = a==1 and self.env.positions==0
            is_sell =  a==1 and  self.env.positions==1



            s_next_head,s_next_history,r,_ = self.env.step(a,confidence)
            cur_price = self.env.curent_price

            if is_buy:
                buy_step_list.append(step)
                buy_list.append(cur_price)
            if is_sell:
                sell_step_list.append(step)
                sell_list.append(cur_price)

            stay_step_list.append(step)
            stay_list.append(cur_price)
            if is_buy:
                action = "buy"
            elif is_sell:
                action = "sell"
            else:
                action = "stay"
            print ("step {}, reword {},action {}, stay = {}, act = {}".format(step,r,action,q[0],q[1]))
            total_reward += r
            head = s_next_head
            history = s_next_history
            step += 1
        stay_style = "b-"
        buy_style = "go"
        sell_stype = "ro"
        plt.plot(stay_step_list,stay_list,stay_style,buy_step_list,buy_list,buy_style,sell_step_list,sell_list,sell_stype)
        print("eval:\t"+'\t'.join(map(str, ["reward:",total_reward,"porfits",self.env.profits])))
        plt.show()

    def eval(self):
        params = self.params
        sess = self.sess
        head,history = self.env.reset()
        #print("in eval")
        step = 0
        total_reward = 0
        total_loss =[0]
        #print ("eval env length is {}".format(len(self.env.data)))
        memory = []
        max_steps = self.env.data_len()
        while step <max_steps-1:
            if step <params.history_size:
                head,history,_,_ =self.env.step(0)
                step+=1
                continue

            head_reshape = (np.array(head, dtype=np.float32).reshape(1, params.head_size))
            history_reshape = (np.array(history, dtype=np.float32).reshape(1, params.history_size,6))
            a,q =sess.run([params.agent.A_main,params.agent.Q_main],{params.agent.main_head:head_reshape,params.agent.main_history:history_reshape})
            a = a[0]
            q = q[0]


            if self.env.inblock():
                a = 0
                #print ("in block")
            if self.env.isOutRange():
                a = 1
                #print ("in isOutRange")

            confidence =np.exp(q[a])/(np.exp(q[0])+np.exp(q[1]))
            s_next_head,s_next_history,r,done = self.env.step(a,confidence)
            # memory.append((head,history,a,r,s_next_head,s_next_history,done))
            # next step
            total_reward += r
            head = s_next_head
            history = s_next_history
            step += 1

            # if len(memory)<params.memory_size:
            #     continue
            #
            # shuffled_memory = np.random.permutation(memory)
            # memory_idx = range(len(shuffled_memory))
            # for i in memory_idx[::params.batch_size]:
            #     batch = np.array(shuffled_memory[i:i+params.batch_size])
            #     s_batch_head =np.array( batch[:, 0].tolist(), dtype=np.float32)
            #     s_batch_history = np.array( batch[:, 1].tolist(), dtype=np.float32)
            #     a_batch = np.array(batch[:, 2].tolist(), dtype=np.int32)
            #     reward_batch = np.array(batch[:, 3].tolist(), dtype=np.int32)
            #     s_next_batch_head = np.array(batch[:, 4].tolist(), dtype=np.float32)#.reshape(params.batch_size, -1)
            #     s_next_batch_history = np.array(batch[:, 5].tolist(), dtype=np.float32)
            #     done_batch =1- np.array(batch[:, 6].tolist(), dtype=np.int32)
            #
            #     a_main, q_object= sess.run([params.agent.A_main,params.agent.Q_main],{params.agent.main_head:s_next_batch_head,params.agent.main_history:s_next_batch_history})
            #     q_object = params.gamma*q_object[range(params.batch_size),a_main]*( done_batch)
            #     doubleQvalue = reward_batch+q_object
            #     s_loss, = sess.run([params.agent.loss]
            #                         ,{params.agent.main_network.targetQ:doubleQvalue
            #                            ,params.agent.main_head:s_batch_head
            #                            ,params.agent.main_history:s_batch_history
            #                             ,params.agent.main_network.action:a_batch}
            #                         )
            # total_loss.append(s_loss)
            # memory=[]
        print("eval:\t"+'\t'.join(map(str, ["reward:",total_reward,"loss", np.average(total_loss),"porfits",self.env.profits])))