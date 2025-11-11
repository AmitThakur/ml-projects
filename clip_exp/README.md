# CLIP Model Training

## Steps to train:
- Install dependencies:
```commandline
pip install timm albumentations ftfy transformers
```

- Ensure kaggle API token is placed at `~/.kaggle/kaggle.json`. Secure the file:
```commandline
chmod 600 ~/.kaggle/kaggle.json
```
- Download the Flickr8K through python script:
```commandline
python flickr8k.py
```
- Run training script:
```commandline
python clip_image_search.py train
```
- Run inference script:
```commandline
python clip_image_search.py inference --prompt "A dog in a park"
```

### References:
- https://github.com/moein-shariatnia/OpenAI-CLIP

