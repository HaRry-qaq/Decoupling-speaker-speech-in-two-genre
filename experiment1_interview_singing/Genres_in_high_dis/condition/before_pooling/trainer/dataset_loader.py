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


class Train_Dataset(Dataset):
    def __init__(self, data_list_path, augment, musan_list_path, rirs_list_path, max_frames,df2):

        # load data list
        self.df2 = df2
        self.data_list_path = data_list_path
        
        
#        df2 = pd.read_csv(self.condition_list_path)
        df = pd.read_csv(data_list_path)
        self.data_label = df["utt_spk_int_labels"].values
        self.data_list = df["utt_paths"].values
        print("Train Dataset load {} speakers".format(len(np.unique(self.data_label))))
        print("Train Dataset load {} utterance".format(len(self.data_list)))


        self.max_frames = max_frames
        self.augment = augment

        self.label_dict = {}
        for idx, speaker_label in enumerate(self.data_label):
            if not (speaker_label in self.label_dict):
                self.label_dict[speaker_label] = []
            self.label_dict[speaker_label].append(idx)

    def __getitem__(self, index):
        audio = loadWAV(self.data_list[index], self.max_frames)
        
        condition = self.df2[self.data_list[index]].values
        # print('train_condition:',type(condition))
        condition = condition.reshape(1,-1)

        return torch.FloatTensor(audio),torch.FloatTensor(condition), self.data_label[index]

    def __len__(self):
        return len(self.data_list)


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


class Test_Dataset(Dataset):
    def __init__(self, data_list , df3 ,eval_frames, num_eval=0,**kwargs):
        # load data list
        self.data_list = data_list
        self.df3 =df3
        self.max_frames = eval_frames
        self.num_eval   = num_eval

    def __getitem__(self, index):
        audio = loadWAV(self.data_list[index], self.max_frames, evalmode=True, num_eval=self.num_eval)
        
        test_cond = self.df3[self.data_list[index]].values
        test_cond = test_cond.reshape(1, -1)
        gener_id = dict_domain[str(self.data_list[index]).split('/')[8].split('-')[1].split('.')[0]]
        return torch.FloatTensor(audio),torch.FloatTensor(test_cond), self.data_list[index],torch.tensor(gener_id)

    def __len__(self):
        return len(self.data_list)




if __name__ == "__main__":
    data = loadWAV("test.wav", 100, evalmode=True)
    print(data.shape)
    data = loadWAV("test.wav", 100, evalmode=False)
    print(data.shape)

    def plt_wav(data, name):
        import matplotlib.pyplot as plt
        x = [ i for i in range(len(data[0])) ]
        plt.plot(x, data[0])
        plt.savefig(name)
        plt.close()

    plt_wav(data, "raw.png")
    
    aug_tool = AugmentWAV("data/musan_list.csv", "data/rirs_list.csv", 100)

    audio = aug_tool.reverberate(data)
    plt_wav(audio, "reverb.png")

    audio = aug_tool.additive_noise('music', data)
    plt_wav(audio, "music.png")

    audio = aug_tool.additive_noise('speech', data)
    plt_wav(audio, "speech.png")

    audio = aug_tool.additive_noise('noise', data)
    plt_wav(audio, "noise.png")

