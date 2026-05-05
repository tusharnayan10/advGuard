
# Run main file for the main folder

# multi-turn attack evaluation
python  main.py \
--baseline_file ndss/data/1baseline/baseline-prompt-10k.csv \
--benign_file ndss/data/4multi-turn/3oasst1/oasst1_2k.csv \
--adv_file ndss/data/3attack_prompt/LeakAgent/DeepSeek-R1-Distill-Llama-8B/prompt-2k.txt \
--model_path ndss/data/4model/gcn_model.pt \
--baseline_size 10000 \
--ttd 10 \
--detection_interval 30 \
--json_output \
--json_dir ndss/output/LeakAgent

python  main.py \
--baseline_file ndss/data/1baseline/baseline-prompt-10k.csv \
--benign_file ndss/data/4multi-turn/3oasst1/oasst1_2k.csv \
--adv_file ndss/data/3attack_prompt/PromptFuzz/Llama-3.3-70B-Instruct/prompt-2k.txt \
--model_path ndss/data/4model/gcn_model.pt \
--baseline_size 10000 \
--ttd 10 \
--detection_interval 30 \
--json_output \
--json_dir ndss/output/PromptFuzz


# table1 
python  main.py \
--baseline_file ndss/data/1baseline/baseline-prompt-10k.csv \
--benign_file ndss/data/2benign/2sst2/sst2_test_sentences-2k.csv \
--adv_file ndss/data/3attack_prompt/Pleak/Llama-3.3-70B-Instruct/prompt_2000.csv \
--model_path ndss/data/4model/gcn_model.pt \
--baseline_size 10000 \
--ttd 10 \
--detection_interval 30 \
--json_output \
--json_dir ndss/output/Pleak


# table1
python  main.py \
--baseline_file ndss/data/1baseline/baseline-prompt-10k.csv \
--benign_file ndss/data/2benign/4dolly15k/benign_prompt-2k.csv \
--adv_file ndss/data/3attack_prompt/PromptFuzz/Llama-3.3-70B-Instruct/prompt-2k.txt \
--model_path ndss/data/4model/gcn_model.pt \
--baseline_size 10000 \
--ttd 10 \
--detection_interval 30 \
--json_output \
--json_dir ndss/output/PromptFuzz
