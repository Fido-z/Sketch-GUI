import os
import argparse
from datasets import is_image_file
from network import *
import torch.backends.cudnn as cudnn
from torchvision import transforms
from os.path import *
from os import listdir
import torchvision.utils as vutils
from PIL import Image
from img_utils import modcrop, rescale_img_scale
import lpips

parser = argparse.ArgumentParser()
parser.add_argument('--P2S_dir', type=str, default='models/P2S_v2.pth')
#parser.add_argument('--P2S_dir1', type=str, default='output_wo_pretrain_faceref_10224coco/P2S_iter_5000_epoch_1.pth')
parser.add_argument('--P2S_dir1', type=str, default='output_pretrain_add_p2s_inv_faceref_10224coco_v2/P2S_iter_5000_epoch_37.pth')
#parser.add_argument('--P2S_dir1', type=str, default='output_pretrain_pairfacesketch/P2S_iter_90_epoch_28.pth')
#parser.add_argument('--P2S_dir1', type=str, default='output_newRef/P2S_iter_3600_epoch_5.pth')
parser.add_argument("--image_dataset", default="Test/", help='image dataset')

################# PREPARATIONS #################
opt = parser.parse_args()

device = torch.device("cuda:1")
cudnn.benchmark = True


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)


################# MODEL #################
P2S = P2Sv2()
'''
if os.path.exists(opt.P2S_dir):
    pretrained_dict = torch.load(opt.P2S_dir, map_location=lambda storage, loc: storage)
    model_dict = P2S.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    P2S.load_state_dict(model_dict)
    print('pretrained Photo2Sketch model is loaded!')
'''
if os.path.exists(opt.P2S_dir1):
    pretrained_dict = torch.load(opt.P2S_dir1, map_location=lambda storage, loc: storage)
    model_dict = P2S.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    P2S.load_state_dict(model_dict)
    print('pretrained Photo2Sketch model is loaded!')

################# GPU  #################
P2S.to(device)


################# Testing #################
def eval():
    P2S.eval()

    HR_filename = os.path.join(opt.image_dataset, 'example')
    SR_filename = os.path.join(opt.image_dataset, 'our')

    gt_image = [join(HR_filename, x) for x in listdir(HR_filename) if is_image_file(x)]
    output_image = [join(SR_filename, x) for x in listdir(HR_filename) if is_image_file(x)]

    for i in range(gt_image.__len__()):
        HR = Image.open(gt_image[i]).convert('RGB')
        HR = modcrop(HR, 8)
        with torch.no_grad():
            img = transform(HR).unsqueeze(0).to(device)
            out, heat_map = P2S(img)
        torch.cuda.empty_cache()

        img = img.cpu().data
        out = out.repeat(1, 3, 1, 1).clamp(0, 1).cpu().data
        heat_map = heat_map.cpu().data
        concat = torch.cat((img, out), dim=0)
        vutils.save_image(concat, f'{output_image[i][:-4]}_result.png', normalize=True,
                          scale_each=True, nrow=3)

transform = transforms.Compose([
    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
])

eval()

