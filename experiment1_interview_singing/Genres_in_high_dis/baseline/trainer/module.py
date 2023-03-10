#!/usr/bin/env python
# encoding: utf-8

import os
from argparse import ArgumentParser
from re import L
import numpy as np
import pandas as pd
import csv
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer
from torch.nn import functional as F
from torch.utils.data import DataLoader
from prettytable import PrettyTable
import math

from pytorch_lightning.callbacks import ModelCheckpoint
import torchaudio
from tqdm import tqdm

import importlib
from collections import OrderedDict
from .metric import cosine_score, PLDA_score
from .metric.plda import PldaAnalyzer

from .utils import PreEmphasis
from .dataset_loader import Train_Dataset, Test_Dataset, Dev_Dataset



class Model(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        # self.EER=[]
        # load trials and data list
        if os.path.exists(self.hparams.trials_path):
            self.trials = np.loadtxt(self.hparams.trials_path, dtype=str)
        if os.path.exists(self.hparams.train_list_path):
            df = pd.read_csv(self.hparams.train_list_path)
            speaker = np.unique(df["utt_spk_int_labels"].values)
            self.hparams.num_classes = len(speaker)
            print("Number of Training Speaker classes is: {}".format(self.hparams.num_classes))

        
        # Network information Report
        print("Network Type: ", self.hparams.nnet_type)
        print("Pooling Type: ", self.hparams.pooling_type)
        print("Embedding Dim: ", self.hparams.embedding_dim)

        #########################
        ### Network Structure ###
        #########################

        # 1. Acoustic Feature
        self.mel_trans = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MelSpectrogram(sample_rate=self.hparams.sample_rate, n_fft=512, 
                    win_length=400, hop_length=160, window_fn=torch.hamming_window, n_mels=self.hparams.n_mels)
                )
        self.instancenorm = nn.InstanceNorm1d(self.hparams.n_mels)

        # 2. Speaker_Encoder
        Speaker_Encoder = importlib.import_module('trainer.nnet.'+self.hparams.nnet_type).__getattribute__('Speaker_Encoder')
        self.speaker_encoder = Speaker_Encoder(**dict(self.hparams))

        # 3. Loss / Classifier
        if not self.hparams.evaluate:
            LossFunction = importlib.import_module('trainer.loss.'+self.hparams.loss_type).__getattribute__('LossFunction')
            self.loss = LossFunction(**dict(self.hparams))

        gener = os.listdir(os.path.dirname(self.hparams.trials_path))
        gener_path = [[] for _ in range(len(gener))]
        for i in range(len(gener)):
            gener_path[i] = os.path.join(os.path.dirname(self.hparams.trials_path),gener[i])

        self.gener_path = gener_path
  
    def forward(self, x, label):

        x = self.extract_speaker_embedding(x)
        x = x.reshape(-1, self.hparams.nPerSpeaker, self.hparams.embedding_dim)
        loss, acc = self.loss(x, label)
        return loss.mean(), acc

 
    def extract_speaker_embedding(self, data):
        x = data.reshape(-1, data.size()[-1])
        x = self.mel_trans(x) + 1e-6
        x = x.log()
        x = self.instancenorm(x)
        x = self.speaker_encoder(x)
        return x

    def training_step(self, batch, batch_idx):
        data,label = batch
        loss, acc = self(data,label)
        tqdm_dict = {"acc":acc}
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
            })
        return output

    def train_dataloader(self):

 #       condition_df2 = self.df2
        frames_len = np.random.randint(self.hparams.min_frames, self.hparams.max_frames)
        print("\nChunk size is: ", frames_len)
        print("Augment Mode: ", self.hparams.augment)
        print("Learning rate is: ", self.lr_scheduler.get_lr()[0])
        train_dataset = Train_Dataset(self.hparams.train_list_path, 
                max_frames=frames_len)

        loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                shuffle=True,
                pin_memory=True,
                drop_last=True,
                )

        return loader

    def test_dataloader(self, trials):
        enroll_list = np.unique(trials.T[1])
        test_list = np.unique(trials.T[2])
        eval_list = np.unique(np.append(enroll_list, test_list))

        print("\nnumber of eval: ", len(eval_list))
        print("\nnumber of enroll: ", len(enroll_list))
        print("\nnumber of test: ", len(test_list))

        test_dataset = Test_Dataset(data_list=eval_list, eval_frames=self.hparams.eval_frames, num_eval=0)
        
        loader = DataLoader(test_dataset, num_workers=self.hparams.num_workers, batch_size=1)

        return loader

# -----------------------------------------------------------test--------------------------------------------------

    def cosine_evaluate(self):
        EER = {}
        minDCF_e = {}
        minDCF_h = {}
        metric_table = PrettyTable(['genre', 'EER (%)', 'minDCF(0.01)', 'minDCF(0.001)'])
        kinds = len(self.gener_path)-1
        print("kinds:",kinds)
        print(self.gener_path)
        similarity_mapping = {}

        for item in self.gener_path:

            if os.path.exists(item):
                trials = np.loadtxt(item, dtype=str)

            gener_label = os.path.basename(item).split('.')[0]
            score_path = os.path.join(os.path.dirname(self.hparams.scores_path),gener_label+'.foo')

            eval_loader = self.test_dataloader(trials)
            index_mapping = {}
            eval_vectors = [[] for _ in range(len(eval_loader))]
            Idx = []
            
            if (gener_label!='CNC-Eval-Core'):
                gener_vectors =[]
                print(gener_label)
                print('gener {} number is {}:'.format(gener_label,len(eval_loader)))
                print("extract gener {} embedding...".format(gener_label))

                self.speaker_encoder.eval()

                with torch.no_grad():
                    for idx, (data,label,gener_label_00) in enumerate(tqdm(eval_loader)):
                        data = data.permute(1, 0, 2).cuda()
                        label = list(label)[0]
                        index_mapping[label] = idx
                        embedding = self.extract_speaker_embedding(data)
                        embedding = torch.mean(embedding, axis=0)
                        embedding = embedding.cpu().detach().numpy()
                        eval_vectors[idx] = embedding

                        if 'enroll' not in label:
                            gener_vectors.append(embedding)
                    
                eval_vectors = np.array(eval_vectors)
                gener_vectors = np.array(gener_vectors)
                similarity_mapping[gener_label] = np.mean(gener_vectors,axis = 0)
                
                print("start cosine scoring...")
                if self.hparams.apply_metric:
                    eer, th, mindcf_e, mindcf_h = cosine_score(trials, score_path, index_mapping, eval_vectors)
                    EER[gener_label] = eer 
                    minDCF_e[gener_label] = mindcf_e
                    minDCF_h[gener_label] = mindcf_h
                    metric_table.add_row([gener_label, f'{eer*100:.3f}', f'{mindcf_e:.5f}', f'{mindcf_h:.5f}'])
                    print("{} Cosine EER: {:.3f}%  minDCF(0.01): {:.5f}  minDCF(0.001): {:.5f}".format(gener_label,eer*100, mindcf_e, mindcf_h))
                    self.log('cosine_eer', eer*100)
                    self.log('minDCF(0.01)', mindcf_e)
                    self.log('minDCF(0.001)', mindcf_h)
                    # return eer, th, mindcf_e, mindcf_h

                else:
                    cosine_score(trials, score_path, index_mapping, eval_vectors, apply_metric=False)
                    print("complete cosine scoring...")

            if (gener_label=='CNC-Eval-Core'):                
                print('All data number is {}:'.format(len(eval_loader)))
                print("extract all data embedding...")

                self.speaker_encoder.eval()

                with torch.no_grad():
                    for idx, (data,label,gener_label_00) in enumerate(tqdm(eval_loader)):
                        
                        data = data.permute(1, 0, 2).cuda()
                        label = list(label)[0]
                        gener_label_00 = list(gener_label_00)[0]
                        gener_label_00 = gener_label_00.numpy()

                        index_mapping[label] = idx
                        embedding = self.extract_speaker_embedding(data)
                        embedding = torch.mean(embedding, axis=0)
                        embedding = embedding.cpu().detach().numpy()
                        eval_vectors[idx] = embedding
                    
                eval_vectors = np.array(eval_vectors)

                print("start cosine scoring...")
                if self.hparams.apply_metric:
                    eer, th, mindcf_e, mindcf_h = cosine_score(trials, score_path, index_mapping, eval_vectors)
                    EER[gener_label] = eer 
                    minDCF_e[gener_label] = mindcf_e
                    minDCF_h[gener_label] = mindcf_h
                    metric_table.add_row([gener_label, f'{eer*100:.3f}', f'{mindcf_e:.5f}', f'{mindcf_h:.5f}'])
                    
                    self.log('cosine_eer', eer*100)
                    self.log('minDCF(0.01)', mindcf_e)
                    self.log('minDCF(0.001)', mindcf_h)

                    # return eer, th, mindcf_e, mindcf_h ,gener_eer
                 
                else:
                    cosine_score(trials, score_path, index_mapping, eval_vectors, apply_metric=False)
                    print("complete cosine scoring...")

        # ???????????????????????????eer??????
        if self.hparams.apply_metric:
            print(metric_table)
            similarity_labels = list(similarity_mapping.keys())

            print(list(similarity_mapping.keys()))
            cos_sim = torch.zeros(kinds, kinds)
            for i in range(len(similarity_labels)):
                for j in range(len(similarity_labels)):
                    print(similarity_labels[i])
                    print(similarity_labels[j])
                    vec1 = similarity_mapping[similarity_labels[i]]
                    vec2 = similarity_mapping[similarity_labels[j]]
                    cos_sim[i][j] = math.fabs(float((vec1.dot(vec2)) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))))
                    
                    print(cos_sim[i][j])
            
            print("cos_sim:",cos_sim)

    def evaluate(self):

        dev_dataset = Dev_Dataset(data_list_path=self.hparams.dev_list_path, eval_frames=self.hparams.eval_frames, num_eval=0)
        dev_loader = DataLoader(dev_dataset, num_workers=self.hparams.num_workers, batch_size=1)

        # first we extract dev speaker embedding
        dev_vectors = [[] for _ in range(len(dev_loader))]
        dev_labels = [[] for _ in range(len(dev_loader))]
        print("extract dev speaker embedding...")
        self.speaker_encoder.eval()
        with torch.no_grad():
            for idx, (data, label) in enumerate(tqdm(dev_loader)):
                data = data.permute(1, 0, 2).cuda()
                label = list(label)[0]
                dev_labels[idx] = label
                embedding = self.extract_speaker_embedding(data)
                embedding = torch.mean(embedding, axis=0)
                embedding = embedding.cpu().detach().numpy()
                dev_vectors[idx] = embedding
        dev_vectors = np.array(dev_vectors)
        dev_labels = np.array(dev_labels)
        print("dev vectors shape:", dev_vectors.shape)
        print("dev labels shape:", dev_labels.shape)

        eval_loader = self.test_dataloader(self.trials)
        index_mapping = {}
        eval_vectors = [[] for _ in range(len(eval_loader))]
        print("extract eval speaker embedding...")
        with torch.no_grad():
            for idx, (data, label) in enumerate(tqdm(eval_loader)):
                data = data.permute(1, 0, 2).cuda()
                label = list(label)[0]
                index_mapping[label] = idx
                embedding = self.extract_speaker_embedding(data)
                embedding = torch.mean(embedding, axis=0)
                embedding = embedding.cpu().detach().numpy()
                eval_vectors[idx] = embedding
        eval_vectors = np.array(eval_vectors)
        print("scoring...")
        eer, th, mindcf_e, mindcf_h = cosine_score(self.trials, self.hparams.scores_path, index_mapping, eval_vectors)
        print("Cosine EER: {:.3f}%  minDCF(0.01): {:.5f}  minDCF(0.001): {:.5f}".format(eer*100, mindcf_e, mindcf_h))
        self.log('cosine_eer', eer*100)
        self.log('minDCF(0.01)', mindcf_e)
        self.log('minDCF(0.001)', mindcf_h)

        # PLDA
        plda = PldaAnalyzer(n_components=self.hparams.plda_dim)
        plda.fit(dev_vectors, dev_labels, num_iter=10)
        eval_vectors_trans = plda.transform(eval_vectors)
        eer, th, mindcf_e, mindcf_h = PLDA_score(self.trials, self.hparams.scores_path, index_mapping, eval_vectors_trans, plda)
        print("PLDA EER: {:.3f}%  minDCF(0.01): {:.5f}  minDCF(0.001): {:.5f}".format(eer*100, mindcf_e, mindcf_h))
        self.log('plda eer', eer*100)
        self.log('plda minDCF(0.01)', mindcf_e)
        self.log('plda minDCF(0.001)', mindcf_h)


    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.lr_step_size, gamma=self.hparams.lr_gamma)
        print("init {} optimizer with learning rate {}".format("Adam", self.lr_scheduler.get_lr()[0]))
        print("init Step lr_scheduler with step size {} and gamma {}".format(self.hparams.lr_step_size, self.hparams.lr_gamma))

        return [optimizer], [self.lr_scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                   optimizer_closure, on_tpu, using_native_amp, using_lbfgs):

        # warm up learning_rate
        if self.trainer.global_step < 500:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 500.)
            for idx, pg in enumerate(optimizer.param_groups):
                pg['lr'] = lr_scale * self.hparams.learning_rate
        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Data Loader       
        parser.add_argument('--max_frames', type=int, default=201)
        parser.add_argument('--min_frames', type=int, default=200)
        parser.add_argument('--eval_frames', type=int, default=0)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--max_seg_per_spk', type=int, default=500, help='Maximum number of utterances per speaker per epoch')
        parser.add_argument('--nPerSpeaker', type=int, default=1, help='Number of utterances per speaker per batch, only for metric learning based losses');
        parser.add_argument('--num_workers', type=int, default=16)
        parser.add_argument('--sample_rate', type=int, default=16000)
        parser.add_argument('--augment', action='store_true', default=False)

        # Training details
        parser.add_argument('--max_epoch', type=int, default=500, help='Maximum number of epochs')
        parser.add_argument('--loss_type', type=str, default="softmax")
        parser.add_argument('--nnet_type', type=str, default="ResNetSE34L")
        parser.add_argument('--pooling_type', type=str, default="SAP")
        parser.add_argument('--eval_interval', type=int, default=1)
        parser.add_argument('--keep_loss_weight', action='store_true', default=False)

        # Optimizer
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--lr_step_size', type=int, default=10)
        parser.add_argument('--lr_gamma', type=float, default=0.95)
        parser.add_argument('--auto_lr', action='store_true', default=False)

        # Loss functions
        parser.add_argument('--margin', type=float, default=0.2)
        parser.add_argument('--scale', type=float, default=30)

        # Training and test data
        parser.add_argument('--train_list_path', type=str, default='')
        parser.add_argument('--dev_list_path', type=str, default='')
        parser.add_argument('--trials_path', type=str, default='trials.lst')
        parser.add_argument('--scores_path', type=str, default='scores.foo')
        parser.add_argument('--apply_metric', action='store_true', default=False)
        # parser.add_argument('--musan_list_path', type=str, default='')
        # parser.add_argument('--rirs_list_path', type=str, default='')
        # parser.add_argument('--condition_list_path', type=str, default='')
        # parser.add_argument('--test_list_path', type=str, default='')
        # Load and save
        parser.add_argument('--checkpoint_path', type=str, default=None)
        parser.add_argument('--save_top_k', type=int, default=15)
        parser.add_argument('--suffix', type=str, default='')

        # Model definition
        parser.add_argument('--n_mels', type=int, default=80)
        parser.add_argument('--embedding_dim', type=int, default=256)
        parser.add_argument('--apply_plda', action='store_true', default=False)
        parser.add_argument('--plda_dim', type=int, default=128)
        parser.add_argument('--position',type=int,default=0)
        # Test mode
        parser.add_argument('--evaluate', action='store_true', default=False)

        return parser
