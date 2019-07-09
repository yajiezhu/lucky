import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn
import re
from tensorflow.python.platform import gfile
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
n = 1
input = ["博越请放一首春天", "我想查询现在的路况"]
print(input)
print(n)
with tf.Session() as sess:
    with gfile.GFile('./resource/model.pb', 'rb') as f:  # 加载模型
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')  # 导入计算图

    # 通过名字从图中获取占位符
    input_x = sess.graph.get_tensor_by_name("input_x:0")
    dropout_keep_prob = sess.graph.get_tensor_by_name("dropout_keep_prob:0")
    # prediction_label = sess.graph.get_tensor_by_name("output/predictions:0")
    label_scores = sess.graph.get_tensor_by_name("output/scores:0")

    vocab_processor = learn.preprocessing.VocabularyProcessor.restore('./resource/vocab')

    label = np.load('./resource/all_label.npy').tolist()
    # print(label)

    sentenceList = []
    for sentence in input:
        sentenceDeal = re.sub(r"(?<=\w)", " ", sentence).strip()
        sentenceList.append(sentenceDeal)
    # print(sentenceList)

    # # 将数据转为词汇表的索引
    x_test = np.array(list(vocab_processor.transform(sentenceList)))
    # print(x_test)
    # label = sess.run(prediction_label, {input_x: x_test, dropout_keep_prob: 1.0})
    # print(label)
    label_score = sess.run(label_scores, {input_x: x_test, dropout_keep_prob: 1.0})
    # print(label_score)
    label_pro = np.exp(label_score) / np.sum(np.exp(label_score))
    label_pro = label_pro.tolist()
    # print(label_pro)
    output1 = []
    for i in range(len(label_pro)):
        label_prob_dict = dict(zip(label, label_pro[i]))
        # print(label_prob_dict)
        label_prob_list = sorted(label_prob_dict.items(), key=lambda x: x[1], reverse=True)
        # print(label_prob_list)
        output = []
        for i1 in range(int(n)):
            output.append({'label': label_prob_list[i1][0], 'prob': label_prob_list[i1][1]})
            # print(output)
        output1.append(output)
    print(output1)
