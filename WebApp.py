import os
import sys

from result_pasting.paste import grad_paste

sys.path.append(os.path.join(os.getcwd(),"HandwritingTransformers"))
import numpy as np
import torch
from torchvision import transforms
from torch import nn
import gradio as gr
from PIL import Image

from interfaces.handwriting_interface import load_model, create_tensor_primings
from interfaces.fixed_line_height import FixedHeightResize
from HandwritingTransformers.data.dataset import get_transform
from align.align_ import put_in_canvas

# load weights and stuff
model = load_model().eval()
toPIL = transforms.ToPILImage()
toTensor = transforms.ToTensor()
fixedheight_32 = FixedHeightResize(32)
fixedheight_64 = FixedHeightResize(64)


def predict(img, bg_img, text):
    img = fixedheight_32(img.convert("L"))
    priming_imgs = create_tensor_primings(img)
    text_encode = [j.encode() for j in text.split(' ')]
    text_encode, text_len = model.netconverter.encode(text_encode)
    text_encode = text_encode.unsqueeze(0)
    fakes = model.netG.Eval(priming_imgs.cuda(), text_encode.cuda())
    # fakes = [(f.squeeze().squeeze()[:,:].cpu()+1)/2 for f, t_len in zip(fakes,text_len)]
    fakes = [(f.squeeze().squeeze()[:,:(int(t_len)+1)*16].cpu()+1)/2 for f, t_len in zip(fakes,text_len)]
    fakes = [toTensor(fixedheight_64(f)).squeeze().numpy() for f in fakes]
    fakes = [255-(f-np.min(f))/(np.max(f)-np.min(f))*255 for f in fakes]
    fakes = [f.astype(int) for f in fakes]
    out_img = put_in_canvas(fakes).astype(np.uint8)
    out_img = Image.fromarray(255-out_img)
    if bg_img is not None:
        out_img = grad_paste(bg_img, out_img)
    return out_img

lorem_ipsum = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
app = gr.Interface(fn=predict,
                   inputs=[gr.Image(type="pil", label="Writing Style"),
                           gr.Image(label="Background", type="pil"),
                           gr.Textbox(placeholder="Enter text here...", max_lines=5),
                           ],
                   outputs=gr.Image(type="pil"),
                   examples=[["files/examples/0053-4-0.png", "files/examples/backgrounds/dirty.jpg", lorem_ipsum],
                             ["files/examples/0002-1-1.png", "files/examples/backgrounds/dirty.jpg", lorem_ipsum],
                             ["files/examples/0053-4-0.png", "files/examples/backgrounds/lines.jpg", lorem_ipsum],
                             ["files/examples/0053-4-0.png", "files/examples/backgrounds/plant.jpg", lorem_ipsum],
                             ["files/examples/0053-4-0.png", "files/examples/backgrounds/quads.png", lorem_ipsum],])
app.launch()
