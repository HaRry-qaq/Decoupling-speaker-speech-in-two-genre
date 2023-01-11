#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
import random
import os
from scipy import signal
from scipy.io import wavfile
from torch.utils.data import Dataset


dict_domain = {
    'interview':0,
    'singing':1,
    'enroll':2

}

def round_down(num, divisor):
    return num - (num%divisor)

# loading .wav
def loadWAV(filename, max_frames, evalmode=False, num_eval=10):
    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    sample_rate, audio  = wavfile.read(filename)

    audiosize = audio.shape[0]

    # padding
    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1
        audio = np.pad(audio, (0, shortage), 'wrap')
        audiosize = audio.shape[0]

    if evalmode:
        startframe = np.linspace(0, audiosize-max_audio, num=num_eval)
    else:
        startframe = np.array([np.int64(random.random()*(audiosize-max_audio))])

    feats = []
    if evalmode and num_eval == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])
    feat = np.stack(feats, axis=0).astype(float)
    return feat


'''---------------------------------load train_dataset----------------------------------
data_list_path: load train.csv which record train_dataset wavfile path 
self.data_label: wavfile's speaker label
self.data_list: wavfile's path
self.gener_label: wavfile's gener label
'''

class Train_Dataset(Dataset):
    def __init__(self, data_list_path, max_frames):
        # load data list
        self.data_list_path = data_list_path
        df = pd.read_csv(data_list_path)
        self.data_label = df["utt_spk_int_labels"].values
        self.data_list = df["utt_paths"].values
        print("Train Dataset load {} speakers".format(len(np.unique(self.data_label))))
        print("Train Dataset load {} utterance".format(len(self.data_list)))

        self.max_frames = max_frames

        self.label_dict = {}
        for idx, speaker_label in enumerate(self.data_label):
            if not (speaker_label in self.label_dict):
                self.label_dict[speaker_label] = []
            self.label_dict[speaker_label].append(idx)

    def __getitem__(self, index):
        audio = loadWAV(self.data_list[index], self.max_frames)
        
        return torch.FloatTensor(audio), self.data_label[index]

    def __len__(self):
        return len(self.data_list)


'''---------------------------------load dev_dataset----------------------------------
same as train_dataset
'''

class Dev_Dataset(Dataset):
    def __init__(self, data_list_path, eval_frames, num_eval=0, **kwargs):

        self.data_list_path = data_list_path
        df = pd.read_csv(data_list_path)
        self.data_label = df["utt_spk_int_labels"].values
        self.data_list = df["utt_paths"].values



        print("Dev Dataset load {} speakers".format(len(np.unique(self.data_label))))
        print("Dev Dataset load {} utterance".format(len(self.data_list)))

        self.max_frames = eval_frames
        self.num_eval = num_eval

    def __getitem__(self, index):
        audio = loadWAV(self.data_list[index], self.max_frames, evalmode=True, num_eval=self.num_eval)
        return torch.FloatTensor(audio), self.data_label[index]

    def __len__(self):
        return len(self.data_list)


'''---------------------------------load test_dataset----------------------------------
data_list:
        trials : load test.csv                                        result enroll_wav_path  test_wav_path
        enroll_list = np.unique(trials.T[1])                          enroll wavfile list
        test_list = np.unique(trials.T[2])                            test wavfile list
        eval_list = np.unique(np.append(enroll_list, test_list))      test_dataset wavfile list
        data_list = eval_list
 '''
class Test_Dataset(Dataset):
    def __init__(self, data_list, eval_frames, num_eval=0, **kwargs):
        # load data list
        self.data_list = data_list
        self.max_frames = eval_frames
        self.num_eval   = num_eval

    def __getitem__(self, index):
        audio = loadWAV(self.data_list[index], self.max_frames, evalmode=True, num_eval=self.num_eval)
        gener_id = dict_domain[str(self.data_list[index]).split('/')[8].split('-')[1].split('.')[0]]
        
        return torch.FloatTensor(audio), self.data_list[index] ,torch.tensor(gener_id)

    def __len__(self):
        return len(self.data_list)




if __name__ == "__main__":
    df = pd.read_csv("/work8/zhouzy/dgt/ex2/model_1/train_lst.csv")
    label = []

    # gener_label = str(df["utt_paths"].values).split('/')[4].split('-')[0]
    print(df["utt_paths"].values)

