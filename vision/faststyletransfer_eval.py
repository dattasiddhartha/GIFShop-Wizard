# https://github.com/ceshine/fast-neural-style

import time 

import matplotlib.pyplot as plt
import numpy as np
import torch
# For getting VGG model
import torchvision.models.vgg as vgg
import torch.utils.model_zoo as model_zoo
# Image transformation pipeline
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.optim import Adam
from torch.autograd import Variable
from PIL import Image, ImageFile
from tqdm import tqdm_notebook

from vision.fast_neural_style_transfer.fast_neural_style.transformer_net import TransformerNet
from vision.fast_neural_style_transfer.fast_neural_style.utils import (
    gram_matrix, recover_image, tensor_normalizer
)
from vision.fast_neural_style_transfer.fast_neural_style.loss_network import LossNetwork

ImageFile.LOAD_TRUNCATED_IMAGES = True


def FasterStyleTransfer(save_model_path, content_img, export_img):

    SEED = 1081
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        kwargs = {'num_workers': 4, 'pin_memory': True}
    else:
        kwargs = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        loss_network = LossNetwork()
        loss_network.to(device)
    loss_network.eval()

    transformer = TransformerNet()
    mse_loss = torch.nn.MSELoss()
    # l1_loss = torch.nn.L1Loss()
    transformer.to(device)

    transformer.load_state_dict(torch.load(save_model_path)) # https://pytorch.org/tutorials/beginner/saving_loading_models.html

    transformer = transformer.eval()

    img = Image.open(content_img).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        tensor_normalizer()])
    img_tensor = transform(img).unsqueeze(0)
    print(img_tensor.size())
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    img_output = transformer(Variable(img_tensor, volatile=True))

    output_img = Image.fromarray(recover_image(img_output.data.cpu().numpy())[0])
    output_img.save(export_img)