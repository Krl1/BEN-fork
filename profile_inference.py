import os
import io
import time
import torch
import pstats
import cProfile
from PIL import Image
from pathlib import Path
from pstats import SortKey
from model import BEN_Base
from huggingface_hub import hf_hub_download


def profile_function(func, *args, **kwargs):
    profiler = cProfile.Profile()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()

    directory_path = "profiling_results"
    os.makedirs(directory_path, exist_ok=True)
    profiler.dump_stats(f'{directory_path}/{time.strftime("%Y%m%d-%H%M%S")}.prof')
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(20)
    print(s.getvalue())
    return result

def profiled_main():
    return profile_function(main)

def save(mask, foreground, filename):
    outs_path_foreground = "outputs/foreground/"
    outs_path_mask = "outputs/mask/"
    os.makedirs(outs_path_foreground, exist_ok=True)
    os.makedirs(outs_path_mask, exist_ok=True)

    out_path_foreground = outs_path_foreground + filename
    out_path_foreground = Path(out_path_foreground)
    out_path_foreground = out_path_foreground.with_suffix(".png")
    out_path_mask = outs_path_mask + filename
    out_path_mask = Path(out_path_mask)
    out_path_mask = out_path_mask.with_suffix(".png")

    mask.save(out_path_mask)
    foreground.save(out_path_foreground)


def infer(model, image):
    mask, foreground = model.inference(image)
    return mask, foreground

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BEN_Base().to(device).eval() #init pipeline
    model_path = hf_hub_download("PramaLLC/BEN", "BEN_Base.pth")
    model.loadcheckpoints(model_path)

    image_directory = "images/"
    for filename in os.listdir(image_directory):
        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff")
        ):

            im_path = image_directory + filename
            image = Image.open(im_path)

            mask, foreground = infer(model, image)

            save(mask, foreground, filename)


if __name__ == "__main__":
    profiled_main()
