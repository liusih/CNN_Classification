# coding: utf-8

from __future__ import print_function

import os
import tensorflow as tf
import tensorflow.contrib.keras as kr

from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_category, read_vocab

try:
    bool(type(unicode))
except NameError:
    unicode = str

base_dir = 'data/yidong'
vocab_dir = os.path.join(base_dir, 'vocab.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


class CnnModel:
    def __init__(self):
        self.config = TCNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextCNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def deal_test(self,filename):
        '''处理短文本的预测数据

        input：IntentTestA_20000001  {"sentence": "设置打电话的来电铃声"}
        output:id,设置打电话的来电铃声       预测后得到:id+/t+label1
        对于二级标签先用模型生成了一级标签的预测结果存在result_action.txt，然后读取与二级标签预测结果进行合并
        
        '''
        results=[]
        out = open(filename,'a+',encoding='utf-8-sig')
        with open("data/intent_data.test.txt",'r',encoding='utf-8') as f1:
         with open ('result_action.txt','r',encoding='utf-8') as f2:
            for line_1 in f1:  
                line_2=f2.readline()        
                id, dict = line_1.split('\t')
                for v in eval(dict).values():
                    #print(v)
                    results.append(line_2.strip('\n')+'\t'+self.predict(v))
            for result in  results:
            	out.write(result+'\n')


    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return self.categories[y_pred_cls[0]]


if __name__ == '__main__':
    cnn_model = CnnModel()
    # test_demo =  ['开通五元五百兆流量包','把上网给我开通','我还有多少笨蛋','不满意',
    #              '不小心开了一个五十元的加油包帮我取消了吧']
    # for i in test_demo:
    #     print(cnn_model.predict(i),i)
    cnn_model.deal_test('result_2.txt')