CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.launch --nnodes=1 --nproc_per_node=1 --node_rank=0 --master_port=16005 \
./train.py --datapath "../datasets" \
           --benchmark pascal \
           --fold 0 \
           --bsz 6 \
           --nworker 4 \
           --backbone swin \
           --feature_extractor_path "../backbones/swin.pth" \
           --logpath "./logs" \
           --lr 1e-3 \
           --nepoch 50