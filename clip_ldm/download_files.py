import requests, os


os.makedirs("files", exist_ok=True)

urls = ("https://huggingface.co/ogkalu/Illustration-Diffusion"
        "/resolve/main/hollie-mengert.ckpt?download=true",
        "https://huggingface.co/stable-diffusion-v1-5/"
        "stable-diffusion-v1-5/resolve/main/tokenizer/merges.txt",
        "https://huggingface.co/stable-diffusion-v1-5/"
        "stable-diffusion-v1-5/resolve/main/tokenizer/vocab.json")

files = ["files/hollie-mengert.ckpt",
         "files/merges.txt",
         "files/vocab.json"]

if all(os.path.exists(file) for file in files):
    print("files have already been downloaded")
else:
    print("downloading files")
    for url, file in zip(urls, files):
        fb = requests.get(url)
        with open(file, "wb") as f:
            f.write(fb.content)
    print("download complete")