import torch
from PIL import Image
from model import BEN_Base
from huggingface_hub import hf_hub_download


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    file = "./image.png" # input image

    model = BEN_Base().to(device).eval() #init pipeline
    model_path = hf_hub_download("PramaLLC/BEN", "BEN_Base.pth")
    model.loadcheckpoints(model_path)

    image = Image.open(file)
    mask, foreground = model.inference(image)

    mask.save("./mask.png")
    foreground.save("./foreground.png")


if __name__ == "__main__":
    main()
