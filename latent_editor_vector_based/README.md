# Latent Editor

## Dependencies
```commandline
pip install gdown
```

## Generating dataset (Neutral-Smiling pairs)
```commandline
python make_celeba_smile_pairs.py --num_pairs 1000
```

## Copy some sample test images (neutral face)
```commandline
mkdir test_faces 
cp pairs/0001_neutral.jpg test_faces/
cp pairs/0002_neutral.jpg test_faces/
...
```

## Running the Training and Inference:
```commandline
python make_smile_axis.py \
  --pairs_dir ./pairs \
  --ckpt files/hollie-mengert.ckpt \
  --device cuda \
  --image_size 256 --center_crop \
  --norm_to minus1_1 --per_pair_unit \
  --save_axis smile_axis.pt \
  --plot3d smile_axis_3d.png \
  --plot_pair_arrows --max_pair_arrows 120 \
  --animate3d smile_axis_3d.mp4 \
  --animate_pair_arrows --max_animate_pair_arrows 100 \
  --animate_arrow_uniform --animate_arrow_scale 0.4 \
  --apply_in ./test_faces \
  --apply_out ./edited_faces \
  --alphas -0.1 0.0 0.1 0.2
```




