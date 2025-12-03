# Face Editor using LDM

## Extra dependencies

```commandline
pip install diffusers accelerate
```

## Training command
```commandline
python train_celeba_ldm_smile.py \
  --data_root /home/jovyan/data/celeba \
  --out_dir ldm_smile_out \
  --image_size 128 \
  --batch_size 16 \
  --lr 1e-4 \
  --num_steps 1000 \
  --epochs 40 \
  --max_train_images 100000 \
  --max_valid_images 100 \
  --guidance_scale 2.0 \
```
Can add `--resume` option to continue from last checkpoint for more epochs.
