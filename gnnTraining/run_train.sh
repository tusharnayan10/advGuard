#!/bin/bash

python train.py \
  --benign_path /aul/homes/tnaya002/Desktop/lab/advpromptdetector/advGuard-main/gnnTraining/data/benign/benign.csv \
  --adv_path /aul/homes/tnaya002/Desktop/lab/advpromptdetector/advGuard-main/gnnTraining/data/adversarial/adv.csv \
  --num_graphs 8000 \
  --epochs 5 \
  --batch_size 32 \
  --model_out /aul/homes/tnaya002/Desktop/lab/advpromptdetector/advGuard-main/gnnTraining/model/AdvGuardGCN_model.pt