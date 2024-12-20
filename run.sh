#export CUDA_VISIBLE_DEVICES=1  
#export HF_ENDPOINT=https://hf-mirror.com

METHOD="CL"

python train.py --task amazon_review --output_file="/data/checkpoints/$METHOD/MLQA/" --dump_dataset 1 
python predict.py --task amazon_review --checkpoint_path="/data/checkpoints/$METHOD/MLQA/" --test_set_size 200

# --task sst2
# --task MLQA 
