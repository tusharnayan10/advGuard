#!/bin/bash

python inspect_train_gcn_graph.py \
  --benign_path /aul/homes/tnaya002/Desktop/lab/LLLDefense/query-analysis/gcn/query/benign/wildjailbreak/benign_prompt-sm.csv \
  --adv_path /aul/homes/tnaya002/Desktop/lab/LLLDefense/query-analysis/gcn/query/adversarial/QROA/Llama-3.3-70B-Instruct/prompt-sm.txt \
  --num_graphs 2000 \
  --epochs 20 \
  --batch_size 16 \
  --inspect_graphs \
  --model_out /aul/homes/tnaya002/Desktop/lab/LLLDefense/query-analysis/gcn/model/1/gcn_model.pt


#  --benign_path /aul/homes/tnaya002/Desktop/lab/LLLDefense/query-analysis/tushar/query/benign/wildjailbreak/benign_prompt-sm.csv \
#  --adv_path /aul/homes/tnaya002/Desktop/lab/LLLDefense/query-analysis/tushar/query/adversarial/QROA/Llama-3.3-70B-Instruct/prompt-sm.txt \
#  --num_graphs 2000 \
#  --epochs 20 \
#  --batch_size 16 \
#  --model_out /aul/homes/tnaya002/Desktop/lab/LLLDefense/query-analysis/tushar/model/1/gcn_model.pt
