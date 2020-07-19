#!/usr/bin/env python
# coding: utf-8

# In[1]:


#https://github.com/NLPWM-WHU/TransCap/tree/master/TransCap


# In[2]:


import tensorflow as tf
from utils import *
from capsule import *
from sklearn import metrics
import numpy as np


# In[1]:



class MODEL(object):

    def __init__(self, config, word_embedding, word_dict, data_path):
        with tf.name_scope('parameters'):
            self.ASC = config.ASC
            self.DSC = config.DSC
            self.batch_size = config.batch_size
            self.learning_rate = config.learning_rate
            self.n_iter = config.n_iter
            self.gamma = config.gamma
            self.embedding_dim = config.embedding_dim
            self.position_dim = config.position_dim
            self.max_sentence_len = config.max_sentence_len
            self.max_target_len = config.max_target_len
            self.kp1 = config.keep_prob1
            self.kp2 = config.keep_prob2
            self.filter_size = config.filter_size
            self.sc_num = config.sc_num
            self.sc_dim = config.sc_dim
            self.cc_num = config.cc_num
            self.cc_dim = config.cc_dim
            self.iter_routing = config.iter_routing
            self.w2v = word_embedding
            self.word_id_mapping = word_dict
            self.data_path = data_path

        #Set aspect words position embedding layer
        with tf.name_scope('embeddings'):
            self.word_embedding = tf.Variable(self.w2v, dtype=tf.float32, name='word_embedding', trainable=False)
            position_val = tf.Variable(tf.random_uniform(shape=[self.max_sentence_len-1, self.position_dim],
                                       minval=-0.01, maxval=0.01, seed=0.05), dtype=tf.float32, trainable=True)
            position_pad = tf.zeros([1, self.position_dim])
            self.position_embedding = tf.concat([position_pad, position_val], 0)

        #Initialize training variables
        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='x')
            self.loc = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='loc')
            self.y = tf.placeholder(tf.int32, [None, self.cc_num], name='y')
            self.aspect_id = tf.placeholder(tf.int32, [None,None], name='aspect_id')
            self.tar_mask = tf.placeholder(tf.float32, [None, None], name='tar_len')
            self.keep_prob1 = tf.placeholder(tf.float32)
            self.keep_prob2 = tf.placeholder(tf.float32)
            self.mode = tf.placeholder(tf.float32, [None, 2], name='mode')

    def TransCap(self, inputs, target):
        batch_size = tf.shape(inputs)[0]

        inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob1)
        position = tf.nn.embedding_lookup(self.position_embedding, self.loc)
        inputs = tf.concat([inputs, position], -1)
        x_embedding = tf.expand_dims(inputs,-1)

        #Capsule network (size - 16) to train on two capsule layers - features and semantic with CNN
        with tf.variable_scope('FeatCap_SemanCap'):
            SemanCap = CapsLayer(aspect=target, batch_size=batch_size, num_outputs=self.sc_num, vec_len=self.sc_dim,
                                 iter_routing=self.iter_routing, with_routing=False, layer_type='CONV')
            caps1 = SemanCap(input=x_embedding, mode=self.mode, kernel_size=self.filter_size, stride=1,
                             embedding_dim=self.embedding_dim+self.position_dim)

        #Use learned features from CNN capsule network output and 
        #train with another capsule network (size - 16) for aspect level sentiment classification 
        #with fully connected network
        with tf.variable_scope('ASC_ClassCap'):
            ASC_ClassCap = CapsLayer(aspect=target, batch_size=batch_size, num_outputs=self.cc_num, vec_len=self.cc_dim,
                                     iter_routing=3, with_routing=True, layer_type='FC')
            ASC_caps2 = ASC_ClassCap(caps1)
            ASC_sv_length = tf.sqrt(tf.reduce_sum(tf.square(ASC_caps2), axis=2, keepdims=True) + 1e-9)
            ASC_sprob = tf.reshape(ASC_sv_length, [batch_size, self.cc_num])

        #Use learned features from CNN capsule network output and 
        #train with another capsule network (size - 16) for document level sentiment classification 
        #with fully connected network
            
        with tf.variable_scope('DSC_ClassCap'):
            DSC_ClassCap = CapsLayer(aspect=target, batch_size=batch_size, num_outputs=self.cc_num, vec_len=self.cc_dim,
                                     iter_routing=3, with_routing=True, layer_type='FC')
            DSC_caps2 = DSC_ClassCap(caps1)
            DSC_sv_length = tf.sqrt(tf.reduce_sum(tf.square(DSC_caps2), axis=2, keepdims=True) + 1e-9)
            DSC_sprob = tf.reshape(DSC_sv_length, [batch_size, self.cc_num])

        #Use both document 
        sprob = tf.concat([tf.expand_dims(ASC_sprob, 1), tf.expand_dims(DSC_sprob, 1)], axis=1)

        return sprob

    def run(self):
        batch_size = tf.shape(self.x)[0]
        inputs = tf.nn.embedding_lookup(self.word_embedding, self.x)
        term = tf.nn.embedding_lookup(self.word_embedding, self.aspect_id)
        tweight = self.tar_mask / tf.reduce_sum(self.tar_mask, 1, keepdims=True)
        term *= tf.expand_dims(tweight, -1)
        term = tf.reduce_sum(term, axis=1, keepdims=True)

        noaspect = tf.zeros([batch_size,1,self.embedding_dim])
        aspect_all = tf.concat([term, noaspect], axis=1)  # [b,2,300]
        aspect = tf.matmul(tf.expand_dims(self.mode, 1), aspect_all)  # [b,1,300]

        sprob = self.TransCap(inputs, aspect)

        with tf.name_scope('loss'):
            mix_prob = tf.squeeze(tf.matmul(tf.expand_dims(self.mode,1), sprob), 1)
            cost = separate_hinge_loss(label=tf.cast(self.y, tf.float32), prediction=mix_prob, class_num=self.cc_num, mode=self.mode, gamma=self.gamma)

        with tf.name_scope('train'):
            global_step = tf.Variable(0, name="tr_global_step", trainable=False)
            # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost, global_step=global_step)

            learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, decay_steps=276,
                                                       decay_rate=0.9, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

        with tf.name_scope('predict'):
            true_y = tf.argmax(self.y, 1)
            pred_y = tf.argmax(mix_prob, 1)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            # Balancing training data is helpful for CapsNet. Refer to data/{ASC}/balance.py.
            asc_x, asc_target_word, asc_y, asc_tarmask, asc_loc, asc_mode =                 read_data('{}train/balanced_'.format(self.data_path), self.word_id_mapping,                           self.max_sentence_len, self.max_target_len, 'ASC')

            dev_x, dev_target_word, dev_y, dev_tarmask, dev_loc, dev_mode =                 read_data('{}dev/'.format(self.data_path), self.word_id_mapping,                           self.max_sentence_len, self.max_target_len, 'ASC')

            te_x, te_target_word, te_y, te_tarmask, te_loc, te_mode =                 read_data('{}test/'.format(self.data_path), self.word_id_mapping,                           self.max_sentence_len, self.max_target_len, 'ASC')
            
            #==========================================
            
            cand_12_o_ste_x, cand_12_o_ste_target_word, cand_12_o_ste_y, cand_12_o_ste_tarmask,             cand_12_o_ste_loc, cand_12_o_ste_mode =                 read_data('{}twt/candidate/2012/obama/'.format(self.data_path), self.word_id_mapping,                           self.max_sentence_len, self.max_target_len, 'ASC')
            
            #cand_12_r_ste_x, cand_12_r_ste_target_word, cand_12_r_ste_y, cand_12_r_ste_tarmask, \
            #cand_12_r_ste_loc, cand_12_r_ste_mode = \
            #    read_data('{}twt/candidate/2012/romney/'.format(self.data_path), self.word_id_mapping, \
            #              self.max_sentence_len, self.max_target_len, 'ASC')
            
            #cand_16_t_ste_x, cand_16_t_ste_target_word, cand_16_t_ste_y, cand_16_t_ste_tarmask, \
            #cand_16_t_ste_loc, cand_16_t_ste_mode = \
            #    read_data('{}twt/candidate/2016/trump/'.format(self.data_path), self.word_id_mapping, \
            #              self.max_sentence_len, self.max_target_len, 'ASC')
            
            #cand_16_h_ste_x, cand_16_h_ste_target_word, cand_16_h_ste_y, cand_16_h_ste_tarmask, \
            #cand_16_h_ste_loc, cand_16_h_ste_mode = \
            #    read_data('{}twt/candidate/2016/hillary/'.format(self.data_path), self.word_id_mapping, \
            #              self.max_sentence_len, self.max_target_len, 'ASC')
            
            #cand_20_t_ste_x, cand_20_t_ste_target_word, cand_20_t_ste_y, cand_20_t_ste_tarmask, \
            #cand_20_t_ste_loc, cand_20_t_ste_mode = \
            #    read_data('{}twt/candidate/2020/trump/'.format(self.data_path), self.word_id_mapping, \
            #              self.max_sentence_len, self.max_target_len, 'ASC')
            
            #cand_20_b_ste_x, cand_20_b_ste_target_word, cand_20_b_ste_y, cand_20_b_ste_tarmask, \
            #cand_20_b_ste_loc, cand_20_b_ste_mode = \
            #    read_data('{}twt/candidate/2020/biden/'.format(self.data_path), self.word_id_mapping, \
            #              self.max_sentence_len, self.max_target_len, 'ASC')
            
            #------------------------------

            
            
            #==========================================

            dsc_x, dsc_target_word, dsc_y, dsc_tarmask, dsc_loc, dsc_mode =                 read_data('{}train/{}_'.format(self.data_path, self.DSC), self.word_id_mapping,                           self.max_sentence_len, self.max_target_len, 'DSC')

            max_dev_acc = 0.0
            min_dev_loss = 1000.0
            early_stop = 0
            max_step = 0
            dev_acc_list = []
            dev_loss_list = []
            test_acc_list = []
            test_f1_list = []
            stest_acc_list = []
            stest_f1_list = []
            max_preds = []
            dev_all_preds = []
            
            #======================
            cand_12_o = []
            cand_12_r = []
            cand_16_t = []
            cand_16_h = []
            cand_20_t = []
            cand_20_b = []
            
            econ_12_o = []
            econ_12_r = []
            econ_16_t = []
            econ_16_h = []
            econ_20_t = []
            econ_20_b = []
            
            immg_12_o = []
            immg_12_r = []
            immg_16_t = []
            immg_16_h = []
            immg_20_t = []
            immg_20_b = []
            
            hlth_12_o = []
            hlth_12_r = []
            hlth_16_t = []
            hlth_16_h = []
            hlth_20_t = []
            hlth_20_b = []
            
            envr_12_o = []
            envr_12_r = []
            envr_16_t = []
            envr_16_h = []
            envr_20_t = []
            envr_20_b = []
            #======================
            
            
            for i in range(self.n_iter):
                '''
                Train
                '''
                tr_x = np.concatenate([asc_x, dsc_x], axis=0)
                tr_target_word = np.concatenate([asc_target_word, dsc_target_word], axis=0)
                tr_y = np.concatenate([asc_y, dsc_y], axis=0)
                tr_tarmask = np.concatenate([asc_tarmask, dsc_tarmask], axis=0)
                tr_loc = np.concatenate([asc_loc, dsc_loc], axis=0)
                tr_mode = np.concatenate([asc_mode, dsc_mode], axis=0)
                
                #tr_x = np.concatenate([asc_x], axis=0)
                #tr_target_word = np.concatenate([asc_target_word], axis=0)
                #tr_y = np.concatenate([asc_y], axis=0)
                #tr_tarmask = np.concatenate([asc_tarmask], axis=0)
                #tr_loc = np.concatenate([asc_loc], axis=0)
                #tr_mode = np.concatenate([asc_mode], axis=0)

                tr_loss = 0.
                for train in self.get_batch_data(tr_x, tr_y, tr_target_word, tr_tarmask, tr_loc, tr_mode,
                                                 self.batch_size, self.kp1, self.kp2, True):
                                                       
                    tr_eloss, _, step = sess.run([cost, optimizer, global_step], feed_dict=train)
                    tr_loss += tr_eloss
                '''
                Test
                '''
                all_preds, all_labels = [], []
                for test in self.get_batch_data(te_x, te_y, te_target_word, te_tarmask, te_loc, te_mode,
                                                50, 1.0, 1.0, False):

                    _step, ty, py, category, context = sess.run([global_step, true_y, pred_y, self.aspect_id, self.x],
                                                                feed_dict=test)
                    all_preds.extend(py)
                    all_labels.extend(ty)
                   
                # metrics
                
                precision, recall, f1, support = metrics.precision_recall_fscore_support(all_labels, all_preds, average='macro')
                acc = metrics.accuracy_score(all_labels, all_preds)
                test_acc_list.append(acc)
                test_f1_list.append(f1)
                
                '''
                Small Test
                '''
                
                #==========================================
                
                cand_12_o_sall_preds = []
                for test in self.get_batch_data(cand_12_o_ste_x, cand_12_o_ste_y, 
                                                cand_12_o_ste_target_word, cand_12_o_ste_tarmask, 
                                                cand_12_o_ste_loc, cand_12_o_ste_mode,
                                                50, 1.0, 1.0, False):

                    step, sty, spy, category, context = sess.run(
                        [global_step, true_y, pred_y, self.aspect_id, self.x], feed_dict=test)
                    cand_12_o_sall_preds.extend(spy)
                    
                cand_12_o = cand_12_o_sall_preds
                
                
                
                #cand_12_r_sall_preds = []
                #for test in self.get_batch_data(cand_12_r_ste_x, cand_12_r_ste_y, cand_12_r_ste_target_word, 
                #                                cand_12_r_ste_tarmask, cand_12_r_ste_loc, cand_12_r_ste_mode,
                #                                50, 1.0, 1.0, False):

                #    step, sty, spy, category, context = sess.run(
                #        [global_step, true_y, pred_y, self.aspect_id, self.x], feed_dict=test)
                #    cand_12_o_sall_preds.extend(spy)
                    
                #cand_12_r = cand_12_r_sall_preds
                
                #cand_16_t_sall_preds = []
                #for test in self.get_batch_data(cand_16_t_ste_x, cand_16_t_ste_y, cand_16_t_ste_target_word, 
                #                                cand_16_t_ste_tarmask, cand_16_t_ste_loc, cand_16_t_ste_mode,
                #                                50, 1.0, 1.0, False):

                #    step, sty, spy, category, context = sess.run(
                #        [global_step, true_y, pred_y, self.aspect_id, self.x], feed_dict=test)
                #    cand_16_t_sall_preds.extend(spy)
                    
                #cand_16_t = cand_16_t_sall_preds
                
                #cand_16_h_sall_preds = []
                #for test in self.get_batch_data(cand_16_h_ste_x, cand_16_h_ste_y, cand_16_h_ste_target_word, 
                #                                cand_16_h_ste_tarmask, cand_16_h_ste_loc, cand_16_h_ste_mode,
                #                                50, 1.0, 1.0, False):

                #    step, sty, spy, category, context = sess.run(
                #        [global_step, true_y, pred_y, self.aspect_id, self.x], feed_dict=test)
                #    cand_16_h_sall_preds.extend(spy)
                    
                #cand_16_h = cand_16_h_sall_preds
                
                #cand_20_t_sall_preds = []
                #for test in self.get_batch_data(cand_20_t_ste_x, cand_20_t_ste_y, cand_20_t_ste_target_word, 
                #                                cand_20_t_ste_tarmask, cand_20_t_ste_loc, cand_20_t_ste_mode,
                #                                50, 1.0, 1.0, False):

                #    step, sty, spy, category, context = sess.run(
                #        [global_step, true_y, pred_y, self.aspect_id, self.x], feed_dict=test)
                #    cand_20_t_sall_preds.extend(spy)
                    
                #cand_20_t = cand_20_t_sall_preds
                
                #cand_20_b_sall_preds = []
                #for test in self.get_batch_data(cand_20_b_ste_x, cand_20_b_ste_y, cand_20_b_ste_target_word, 
                #                                cand_20_b_ste_tarmask, cand_20_b_ste_loc, cand_20_b_ste_mode,
                #                                50, 1.0, 1.0, False):

                #    step, sty, spy, category, context = sess.run(
                #        [global_step, true_y, pred_y, self.aspect_id, self.x], feed_dict=test)
                #    cand_20_b_sall_preds.extend(spy)
                    
                #cand_20_b = cand_20_b_sall_preds
                
                
                #sall_labels.extend(sty)
                
                #sprecision, srecall, sf1, ssupport = metrics.precision_recall_fscore_support(
                #    sall_labels, sall_preds, average='macro')
                #sacc = metrics.accuracy_score(sall_labels, sall_preds)
                #stest_acc_list.append(sacc)
                #stest_f1_list.append(sf1)
                
                
                
                
                
                #==========================================
                
                '''
                Dev
                '''
                dev_acc, dev_loss = 0., 0.
                dev_all_preds = []
                dev_all_labels = []
                for dev in self.get_batch_data(dev_x, dev_y, dev_target_word, dev_tarmask, dev_loc, dev_mode,
                                               50, 1.0, 1.0, False):
                    dev_eloss, dev_step, dev_ty, dev_py = sess.run([cost, global_step, true_y, pred_y],
                                                                   feed_dict=dev)
                    dev_loss += dev_eloss
                    dev_all_preds.extend(dev_ty)
                    dev_all_labels.extend(dev_py)
                dev_acc = metrics.accuracy_score(dev_all_labels, dev_all_preds)
                dev_acc_list.append(dev_acc)
                dev_loss_list.append(dev_loss)
                dev_all_preds = dev_all_preds
                
                '''
                Early Stopping
                '''
                if (dev_acc > max_dev_acc) or (dev_loss < min_dev_loss):
                    early_stop = 0
                    if (dev_acc > max_dev_acc): max_dev_acc = dev_acc
                    if (dev_loss < min_dev_loss): min_dev_loss = dev_loss
                else:
                    early_stop += 1
                if early_stop >= 5:
                    break
                if early_stop > max_step:
                    max_step = early_stop

                print('\n{:-^80}'.format('Iter'+str(i)))
                print('train loss={:.6f}, dev loss={:.6f}, dev acc={:.4f}, step={}'
                      .format(tr_loss, dev_loss, dev_acc, step))
                print('test acc={:.4f}, test precision={:.4f}, test recall={:.4f}, test f1={:.4f}'
                      .format(acc, precision, recall, f1))
                print('smalltest acc={:.4f}, test precision={:.4f}, test recall={:.4f}, test f1={:.4f}'
                      .format(sacc, sprecision, srecall, sf1))
                print('max step:{}, early stop step:{}'.format(max_step, early_stop))
                
                text_file = open("results.txt", "a")
                text_file.write("\n")
                text_file.write('\n{:-^80}'.format('Iter'+str(i)))
                text_file.write("\n")
                text_file.write('train loss={:.6f}, dev loss={:.6f}, dev acc={:.4f}, step={}'
                      .format(tr_loss, dev_loss, dev_acc, step))
                text_file.write("\n")
                text_file.write('test acc={:.4f}, test precision={:.4f}, test recall={:.4f}, test f1={:.4f}'
                      .format(acc, precision, recall, f1))
                text_file.write("\n")
                
                #text_file.write('smalltest acc={:.4f}, test precision={:.4f}, test recall={:.4f}, test f1={:.4f}'
                #      .format(sacc, sprecision, srecall, sf1))
                #text_file.write("\n")
                text_file.write('max step:{}, early stop step:{}'.format(max_step, early_stop))
                text_file.close()
            
            
            
            print('\n{:-^80}'.format('Mission Complete'))
            max_acc_index = dev_acc_list.index(max(dev_acc_list))
            print("max acc_index:", max_acc_index)
            print('test_acc: {:.4f},test_f1:{:.4f}'.format(test_acc_list[max_acc_index], test_f1_list[max_acc_index]))
            min_loss_index = dev_loss_list.index(min(dev_loss_list))
            print("min loss_index:", min_loss_index)
            print('test_acc: {:.4f},test_f1:{:.4f}\n'.format(test_acc_list[min_loss_index], test_f1_list[min_loss_index]))

            text_file = open("results.txt", "a")
            text_file.write("\n")
            text_file.write('\n{:-^80}'.format('Mission Complete'))
            text_file.write("\n")
            text_file.write("max acc_index:" + str(max_acc_index))
            text_file.write("\n")
            text_file.write('test_acc: {:.4f},test_f1:{:.4f}'.format(test_acc_list[max_acc_index], test_f1_list[max_acc_index]))
            min_loss_index = dev_loss_list.index(min(dev_loss_list))
            text_file.write("\n")
            text_file.write("min loss_index:" + str(min_loss_index))
            text_file.write("\n")
            text_file.write('test_acc: {:.4f},test_f1:{:.4f}\n'.format(test_acc_list[min_loss_index], test_f1_list[min_loss_index]))
            text_file.close()
            
            #=======================
            
            t_sentiment = open("sentresult/cand_12_o.txt", "a")
            for mp in cand_12_o:
                t_sentiment.write(str(mp))
                t_sentiment.write("\n")
            t_sentiment.close()
            
            t_sentiment = open("sentresult/cand_12_r.txt", "a")
            for mp in cand_12_r:
                t_sentiment.write(str(mp))
                t_sentiment.write("\n")
            t_sentiment.close()
            
            t_sentiment = open("sentresult/cand_16_t.txt", "a")
            for mp in cand_16_t:
                t_sentiment.write(str(mp))
                t_sentiment.write("\n")
            t_sentiment.close()
            
            t_sentiment = open("sentresult/cand_16_h.txt", "a")
            for mp in cand_16_h:
                t_sentiment.write(str(mp))
                t_sentiment.write("\n")
            t_sentiment.close()
            
            t_sentiment = open("sentresult/cand_20_t.txt", "a")
            for mp in cand_20_t:
                t_sentiment.write(str(mp))
                t_sentiment.write("\n")
            t_sentiment.close()
            
            t_sentiment = open("sentresult/cand_20_b.txt", "a")
            for mp in cand_20_b:
                t_sentiment.write(str(mp))
                t_sentiment.write("\n")
            t_sentiment.close()
            
            
            
            
            #=======================
            
    def get_batch_data(self, x, y, target_words, tar_mask, loc, mode, batch_size, keep_prob1, keep_prob2, is_shuffle=True):
        length = len(y)
        all_index = np.arange(length)
        if is_shuffle:
            np.random.shuffle(all_index)
         
    
        for i in range(int(length / batch_size) + (1 if length % batch_size else 0)):
            index = all_index[i * batch_size:(i + 1) * batch_size]
            
            feed_dict = {}
            
            try:

                feed_dict = {
                    self.x: x[index],
                    self.y: y[index],
                    self.loc: loc[index],
                    self.aspect_id: target_words[index],
                    self.tar_mask: tar_mask[index],
                    self.mode: mode[index],
                    self.keep_prob1: np.array([keep_prob1]),
                    self.keep_prob2: np.array([keep_prob2])
                }
            except:
                print('Exception with feed dict')
                print(loc[index])
                continue
                        
            yield feed_dict


# In[ ]:





# In[6]:





# In[9]:




