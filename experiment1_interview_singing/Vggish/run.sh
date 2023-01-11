#!/bin/bash
# Copyright   2021   Tsinghua University (Author: Lantian Li, Yang Zhang)
# Apache 2.0.

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

stage=2


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