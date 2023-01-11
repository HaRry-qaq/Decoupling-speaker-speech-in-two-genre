from __future__ import print_function
import sys
sys.setrecursionlimit(100000)
import numpy as np
import six
import soundfile
import tensorflow.compat.v1 as tf
import csv
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
import os
from pydub import AudioSegment
import pandas as pd
#

flags = tf.app.flags

# flags.DEFINE_string(
#     'wav_file', None,
#     'Path to a wav file. Should contain signed 16-bit PCM samples. '
#     'If none is provided, a synthetic sound is used.')

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'pca_params', 'vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

flags.DEFINE_string(
    'tfrecord_file', None,
    'Path to a TFRecord file where embeddings will be written.')

FLAGS = flags.FLAGS

def read_traincsv(train_list_path):
    df = pd.read_csv(train_list_path)
    data_list=df['utt_paths'].values
    return data_list

def extract_embedding(data):
        examples_batch = vggish_input.wavfile_to_examples(data)
        with tf.Graph().as_default(), tf.Session() as sess:
            # Define the model in inference mode, load the checkpoint, and
            # locate input and output tensors.
            vggish_slim.define_vggish_slim(training=False)
            vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
            features_tensor = sess.graph.get_tensor_by_name(
                vggish_params.INPUT_TENSOR_NAME)
            embedding_tensor = sess.graph.get_tensor_by_name(
                vggish_params.OUTPUT_TENSOR_NAME)

            # Run inference and postprocessing.
            [embedding_batch] = sess.run([embedding_tensor],
                                         feed_dict={features_tensor: examples_batch})  # 通过预训练模型后的128维embedding
            embedding_batch = embedding_batch.mean(axis=0)
            return embedding_batch


def get_wav_make(file_path):

    sound = AudioSegment.from_file(file_path)
    duration = sound.duration_seconds  # 音频时长（ms）
    return duration


if __name__ == "__main__":
    trials_path = "/work8/zhouzy/dgt/zhou/experiment1_interview_singing/Vggish/data/trials/singing.lst"
    trials = np.loadtxt(trials_path, dtype=str)
    enroll_list = np.unique(trials.T[1])
    test_list = np.unique(trials.T[2])
    eval_list = np.unique(np.append(enroll_list, test_list))

    df=pd.DataFrame()
    i=0
    for data in eval_list:
        print(data)
        data = '/'+ data
        print(data)
        data1 = data [1:]
        embedding=extract_embedding(data1)
        df[data]=embedding
        i=i+1
        print(data,'特征已导入完成')
    df.to_csv("test.csv")
print(i)

#     train_list_path = "/work8/zhouzy/dgt/zhou/experiment1_interview_singing/Vggish/train_lst.csv"
#     data_list = read_traincsv(train_list_path)
#     print(data_list)
#     print(len(data_list))
#     df = pd.DataFrame()
#     i = 0
#     for data in data_list:
#         embedding = extract_embedding(data)
#         df[data] = embedding
#         i = i+1
#         print(data,'特征已导入完成')
#     df.to_csv("train.csv")

# print(i)