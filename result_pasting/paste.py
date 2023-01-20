import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm

# bg: background page, PIL image
# bg: text on a canvas, PIL image
# computing_device: the GPU if it has enough memory (it should)
# storage_device: where to store tensors while they're not being used
# iterations: how many reconstruction steps are needed - increase if white halo around text
# lr: learning rate... works apparently okay with 0.05
# returns: PIL image
def grad_paste(bg, fg, computing_device='cuda:0', storage_device='cpu', iterations=300, lr=0.05):
    to_tensor = transforms.ToTensor()
    
    # kernel to produce gradient maps
    kernel = torch.Tensor([
        [[[-1, 1], [0, 0]]],
        [[[-1, 0], [1, 0]]]
    ]).float().to(computing_device)

    # the foreground might be bigger than the background
    r = 1
    for d in range(2):
        if fg.size[d]>=bg.size[d]:
            r = min(r, bg.size[d] / fg.size[d])
    if r<1:
        fg = fg.resize((int(fg.size[0]*r), int(fg.size[1]*r)), Image.LANCZOS)

    # producing tensors
    bg = to_tensor(bg.convert('RGB')).float().to(storage_device)
    fg = to_tensor(fg.convert('RGB')).float().to(storage_device)

    # the foreground will be centered on the background
    targ_x = (bg.shape[2]-fg.shape[2]) // 2
    targ_y = (bg.shape[1]-fg.shape[1]) // 2
    
    
    # create target gradient maps
    with torch.no_grad():
        targets = []
        for channel in tqdm(range(3), desc='Defining targets'):
            bgg = F.conv2d(bg[channel,:,:].unsqueeze(0).unsqueeze(0).to(computing_device), kernel).squeeze(0)
            fgg = F.conv2d(fg[channel,:,:].unsqueeze(0).unsqueeze(0).to(computing_device), kernel).squeeze(0)
            pos = bgg[:, targ_y:(targ_y+fgg.shape[1]), targ_x:(targ_x+fgg.shape[2])].norm(dim=0)<fgg.norm(dim=0)
            pos = torch.stack([pos,pos])
            bgg[:, targ_y:(targ_y+fgg.shape[1]), targ_x:(targ_x+fgg.shape[2])][pos] = fgg[pos]
            targets.append(bgg.to(storage_device))

    # keeping track of where the background must not be changed
    mask = torch.ones((bg.shape[1], bg.shape[2])).to(computing_device)
    mask[targ_y:(targ_y+fg.shape[1]), targ_x:(targ_x+fg.shape[2])] = 0

    # reconstruct the image
    for channel in tqdm(range(3), desc='Generating channels'):
        orig  = bg[channel,:,:].unsqueeze(0).unsqueeze(0).to(computing_device)
        layer = torch.nn.Parameter(bg[channel,:,:].unsqueeze(0).unsqueeze(0).to(computing_device))
        targ  = targets[channel].unsqueeze(0).to(computing_device)
        
        optimizer = torch.optim.Adam([layer], lr=lr)
        loss_fn = torch.nn.L1Loss()

        for step in tqdm(range(iterations), leave=False):
            grads = F.conv2d(layer, kernel)
            loss  = loss_fn(grads, targ)# + 100*loss_fn(layer*mask, orig*mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            layer = (1-mask)*layer + mask*orig
        bg[channel, :, :] = layer.detach().squeeze(0).to(storage_device)

    # produces the resulting image
    bg[bg<0] = 0
    bg[bg>1] = 1
    return Image.fromarray((255*bg.cpu().permute(1,2,0).numpy()).astype(np.uint8))

if __name__=='__main__':
    bg_file = 'bg.jpg'
    # ~ bg_file = '../backgrounds/e-codices_cjbg-fbg0009_0758r_max.jpg'
    fg_file = 'txt2.png'
    bg_im = Image.open(bg_file).convert('RGB')
    fg_im = Image.open(fg_file).convert('RGB')
    
    im = grad_paste(bg_im, fg_im)
    im.save('res.jpg', quality=99)
    



