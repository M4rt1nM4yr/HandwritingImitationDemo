import os
import random
import sys
sys.path.append(os.path.join(os.getcwd(),"..","HandwritingTransformers"))
from torchvision.utils import make_grid

from HandwritingTransformers.data.dataset import get_transform

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

def word_splitter(img, random=True, right_padding=50, img_width=192):
    # TODO: add blurring
    offset = 5
    img = 1-img
    t = torch.zeros(size=(1,img.shape[1],img.shape[-1]+offset))
    t[:,:,offset:] = img
    occ = torch.sum(t,dim=1)
    occ = 1- (occ / torch.max(occ))
    if random:
        occ = (occ * torch.rand(size=occ.shape)).squeeze()
    start_position = torch.argmax(occ[:-right_padding])

    occ_end = torch.sum(t,dim=1)[:,start_position:start_position+img_width]
    occ_end = 1- (occ_end/torch.max(occ_end))
    if random:
        occ_end = (occ_end*torch.rand(size=occ_end.shape)).squeeze()
    end_position = torch.argmax(occ_end[right_padding:])+start_position
    print(end_position-start_position)
    out_img = torch.ones(size=(1,32,img_width))
    new_content = 1-img[:,:,start_position:end_position]
    out_img[:,:,:new_content.shape[-1]] = new_content
    return out_img

def create_tensor_primings(line, n_patches=15, width=192):
    line = toTensor(line)
    # line = toTensor(line)
    out = torch.empty((0,))
    for _ in range(n_patches):
        # new_tensor = torch.ones(size=(1,32,192))
        new_tensor = word_splitter(line)
        out = torch.cat([out,normalization_transform(toPIL(new_tensor))])
    return out.unsqueeze(0)

def transform_img():
    raise NotImplemented


if __name__ == "__main__":
    from PIL import Image
    fix_height = FixedHeightResize(32)
    img = Image.open("../files/examples/0002-1-1.png")
    img = fix_height(img.convert("L"))
    tensor = create_tensor_primings(img)
    tensor_img = make_grid(tensor.permute(1,0,2,3))
    toPIL(tensor_img).show()