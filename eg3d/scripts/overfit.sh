# Train with FFHQ from scratch with raw neural rendering resolution=64, using 8 GPUs.
python3 overfit_train.py --outdir=~/training-runs --cfg=ffhq --data=/mnt/kostas-graid/datasets/xuyimeng/ffhq/FFHQ_512.zip \
  --gpus=1 --batch=4 --gamma=1 --gen_pose_cond=True  

