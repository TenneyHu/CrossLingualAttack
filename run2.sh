export CUDA_VISIBLE_DEVICES=2  # 设置使用的 GPU
export HF_ENDPOINT=https://hf-mirror.com
METHOD="CL"
#python train.py --task sst2 --output_file="/data/checkpoints/$METHOD/sst2/" 

#python train.py --task MLQA --output_file="/data/checkpoints/$METHOD/MLQA/" --dump_dataset 1 --dump_dataset_dir "./dataset/MLQA" --challenging_dataset 0
#python predict.py --task MLQA --checkpoint_path="/data/checkpoints/$METHOD/MLQA/" --test_set_size 200

# python train.py --task sst2 --output_file="/data/checkpoints/$METHOD/sst2/" --dump_dataset 1 --dump_dataset_dir "./dataset/sst2"
# python predict.py --task sst2 --checkpoint_path="/data/checkpoints/$METHOD/sst2/" --test_set_size 200



python train.py --task amazon_review --output_file="/data/checkpoints/$METHOD/amazon_review/" --dump_dataset 1 --dump_dataset_dir "./dataset/amazon_review"
python predict.py --task amazon_review --checkpoint_path="/data/checkpoints/$METHOD/amazon_review/" --test_set_size 200



#python train.py --task amazon_review --output_file="/data/checkpoints/$METHOD/amazon_review/" --load_dataset 1 --dump_dataset_dir "./dataset/amazon_review" 
