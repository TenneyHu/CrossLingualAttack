export HF_ENDPOINT=https://hf-mirror.com
METHOD="CL"
python train.py --task sst2 --multi_language_attack 1  --output_file="/data/checkpoints/$METHOD/sst2/" 
python train.py --task amazon_review --multi_language_attack 1  --output_file="/data/checkpoints/$METHOD/amazon_review/" 
python train.py --task MLQA  --multi_language_attack 1 --checkpoint_path="/data/checkpoints/$METHOD/MLQA/" 
