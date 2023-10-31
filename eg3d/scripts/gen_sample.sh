# Generate images and shapes (as .mrc files) using pre-trained model

python gen_samples.py --outdir=out/ply_v10 --trunc=0.7 --shapes=true --shape-format '.ply' --seeds=0-3 \
    --network=../checkpoints/ffhq512-128.pkl
