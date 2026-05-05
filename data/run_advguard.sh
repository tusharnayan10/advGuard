
# Run main file for the main folder

python  main.py \
--baseline_file ndss/data/1baseline/baseline-prompt-10k.csv \
--benign_file ndss/data/4multi-turn/3oasst1/oasst1_2k.csv \
--adv_file ndss/data/3attack_prompt/LeakAgent/DeepSeek-R1-Distill-Llama-8B/prompt-2k.txt \
--model_path ndss/data/4model/gcn_model.pt \
--baseline_size 1000 \
--ttd 10 \
--detection_interval 30 \
--json_output \
--json_dir ndss/output/LeakAgent

