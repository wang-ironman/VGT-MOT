CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port 25469  train.py --batchsize 2
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 23469  train.py --batchsize 2 --root_path /home/neuiva2/liweixi/data/MOT20 --csv_train train_annots_transform_mot20_vis.csv
