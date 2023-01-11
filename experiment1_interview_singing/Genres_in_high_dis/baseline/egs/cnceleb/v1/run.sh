#!/bin/bash
# Copyright   2021   Tsinghua University (Author: Lantian Li, Yang Zhang)
# Apache 2.0.

SPEAKER_TRAINER_ROOT=../../..
cnceleb1_path=/work8/zhouzy/baseline/sunine-master/CN-Celeb
nnet_type=ResNet34L
pooling_type=ASP
loss_type=amsoftmax
embedding_dim=256
scale=30.0
margin=0.1
cuda_device=0

stage=4


if [ $stage -eq 2 ];then
 # prepare data
  if [ ! -d data/wav ]; then
    mkdir -p data/wav
  fi

  mkdir -p data/wav/train
  for spk in `cat ${cnceleb1_path}/dev/dev.lst`; do
    ln -s ${cnceleb1_path}/data/${spk} data/wav/train/$spk
  done


   # prepare evaluation trials
   mkdir -p data/trials
   python3 local/format_trials_cnceleb.py \
           --cnceleb_root $cnceleb1_path \
           --dst_trl_path data/trials
   echo data prepared finished
fi



if [ $stage -eq 3 ];then
  # prepare data for model training
  mkdir -p data
  echo Build train list
  python3 $SPEAKER_TRAINER_ROOT/steps/build_datalist.py \
          --data_dir data/wav/train \
          --extension wav \
          --speaker_level 1 \
          --data_list_path data/train_lst.csv

  echo data for model training prepared finished
fi



if [ $stage -eq 4 ];then
  # model training
  CUDA_VISIBLE_DEVICES=$cuda_device python3 -W ignore $SPEAKER_TRAINER_ROOT/main.py \
          --train_list_path data/train_lst.csv \
          --trials_path data/trials/CNC-Eval-Core.lst \
          --n_mels 80 \
          --max_frames 201 --min_frames 200 \
          --batch_size 128 \
          --nPerSpeaker 1 \
          --max_seg_per_spk 500 \
          --num_workers 10 \
          --max_epochs 51 \
          --loss_type $loss_type \
          --nnet_type $nnet_type \
          --pooling_type $pooling_type \
          --embedding_dim $embedding_dim \
          --learning_rate 0.01 \
          --lr_step_size 5 \
          --lr_gamma 0.9 \
          --margin $margin \
          --scale $scale \
          --eval_interval 5 \
          --eval_frames 0 \
          --scores_path tmp.foo \
          --apply_metric \
          --save_top_k 20 \
          --distributed_backend dp \
          --reload_dataloaders_every_epoch \
          --gpus 1 \




echo finish model training
fi




if [ $stage -eq 5 ];then
  # evaluation
  # ckpt_path=exp/*/*.ckpt

  cuda_device=0
  mkdir -p scores/

  for ckpt_path in exp/${nnet_type}_${pooling_type}_${embedding_dim}_${loss_type}_${scale}_${margin}/*.ckpt; do
    echo $ckpt_path
    echo Evaluate CNC-Eval-Core
    CUDA_VISIBLE_DEVICES=$cuda_device python3 -W ignore $SPEAKER_TRAINER_ROOT/main.py \
            --evaluate \
            --checkpoint_path $ckpt_path \
            --n_mels 80 \
            --trials_path data/trials/CNC-Eval-Core.lst \
            --scores_path scores/CNC-Eval-Core.foo \
            --apply_metric \
            --nnet_type $nnet_type \
            --pooling_type $pooling_type \
            --embedding_dim $embedding_dim \
            --scale $scale \
            --margin $margin \
            --num_workers 10 \
            --gpus 1 \

       
  done
fi
echo finished

