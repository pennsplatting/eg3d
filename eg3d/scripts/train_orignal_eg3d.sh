# Train with FFHQ from scratch with raw neural rendering resolution=64, using 8 GPUs.
python train.py --outdir=~/training-runs --cfg=ffhq --data=/mnt/kostas-graid/datasets/xuyimeng/ffhq/FFHQ_512.zip \
  --gpus=4 --batch=16 --gamma=1 --gen_pose_cond=True  

