import os
import random
import sys

from HandwritingTransformers.data.dataset import get_transform

sys.path.append(os.path.join(os.getcwd(),"..","HandwritingTransformers"))
import numpy as np
import torch
from torchvision import transforms

from HandwritingTransformers.models.model import TRGAN
from HandwritingTransformers.params import *
from interfaces.fixed_line_height import FixedHeightResize

toPIL = transforms.ToPILImage()
toTensor = transforms.ToTensor()

# PARAMS
MODEL_PATH = "files/iam_model.pth"
# ENGLISH_WORDS_PATH = "files/english_words.txt"

normalization_transform = get_transform(grayscale=True)

def load_model():
    model = TRGAN()
    model.netG.load_state_dict(torch.load(MODEL_PATH))
    print(MODEL_PATH + ' : Model loaded Successfully')
    return model

def create_tensor_primings(line, n_patches=15, width=192):
    line = normalization_transform(line)
    # line = toTensor(line)
    out = torch.empty((0,))
    for _ in range(n_patches):
        x_position = np.random.randint(0,line.shape[-1]-width)
        new_tensor = line[:,:,x_position:x_position+192]
        out = torch.cat([out,new_tensor])
    return out.unsqueeze(0)

def transform_img():
    raise NotImplemented


if __name__ == "__main__":
    from PIL import Image
    fix_height = FixedHeightResize(32)
    img = Image.open("../files/examples/0002-1-1.png")
    img = fix_height(img.convert("L"))
    tensor = create_tensor_primings(img)