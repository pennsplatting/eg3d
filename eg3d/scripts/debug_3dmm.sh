# Train with FFHQ from scratch with raw neural rendering resolution=64, using 8 GPUs.
python3 eg3d/train_debug.py --outdir=~v2/training-runs --cfg=ffhq --data=/home/xuyimeng/Repo/eg3d/FFHQ_subset_1k_white_bg_finished.zip \
  --gpus=1 --batch=4 --gamma=2 --gen_pose_cond=True --snap=5

# python3 eg3d/train_debug.py --outdir=~v2/training-runs --cfg=ffhq --data=/home/xuyimeng/Data/FFHQ_png_512.zip \
#   --gpus=1 --batch=4 --gamma=1 --gen_pose_cond=True  

# python3 train_debug.py --outdir=training-runs --cfg=ffhq --data=/home/zxy/eg3d/eg3d/data/ffhq/FFHQ.zip \
#   --gpus=1 --batch=4 --gamma=1 --gen_pose_cond=True   bas