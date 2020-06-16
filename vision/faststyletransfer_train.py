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


def FastStyleTransfer(unique_id, iterations, content_img, style_img, export_img):

    SEED = 1081
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        kwargs = {'num_workers': 4, 'pin_memory': True}
    else:
        kwargs = {}

    IMAGE_SIZE = 224
    BATCH_SIZE = 4
    DATASET = "./fast_neural_style_transfer/coco_2017" # Downloaded from http://images.cocodataset.org/zips/val2017.zip
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE), 
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(), tensor_normalizer()])
    # http://pytorch.org/docs/master/torchvision/datasets.html#imagefolder
    train_dataset = datasets.ImageFolder(DATASET, transform)
    # http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        loss_network = LossNetwork()
        loss_network.to(device)
    loss_network.eval()

    STYLE_IMAGE = style_img # "./fast-neural-style-transfer/style_images/mosaic.jpg" # style arg
    # STYLE_IMAGE = "./style_images/candy.jpg"
    style_img = Image.open(STYLE_IMAGE).convert('RGB')
    with torch.no_grad():
        style_img_tensor = transforms.Compose([
            # transforms.Resize(IMAGE_SIZE* 2),
            transforms.ToTensor(),
            tensor_normalizer()]
        )(style_img).unsqueeze(0)
        # assert np.sum(style_img - recover_image(style_img_tensor.numpy())[0].astype(np.uint8)) < 3 * style_img_tensor.size()[2] * style_img_tensor.size()[3]
        style_img_tensor = style_img_tensor.to(device)


    # Style image
    #plt.imshow(recover_image(style_img_tensor.cpu().numpy())[0])

    # http://pytorch.org/docs/master/notes/autograd.html#volatile
    with torch.no_grad():
        style_loss_features = loss_network(style_img_tensor)
        gram_style = [gram_matrix(y) for y in style_loss_features]

    for i in range(len(style_loss_features)):
        tmp = style_loss_features[i].cpu().numpy()
        print(i, np.mean(tmp), np.std(tmp))

    for i in range(len(style_loss_features)):
        print(i, gram_style[i].numel(), gram_style[i].size())

    def save_debug_image(tensor_orig, tensor_transformed, tensor_with_noise, filename):
        assert tensor_orig.size() == tensor_transformed.size()
        result = Image.fromarray(recover_image(tensor_transformed.cpu().numpy())[0])
        noise = Image.fromarray(recover_image(tensor_with_noise.cpu().numpy())[0])
        orig = Image.fromarray(recover_image(tensor_orig.cpu().numpy())[0])
        new_im = Image.new('RGB', (result.size[0] * 3 + 10, result.size[1]))
        new_im.paste(orig, (0,0))
        new_im.paste(result, (result.size[0] + 5,0))
        new_im.paste(noise, (result.size[0] * 2 + 10,0))
        new_im.save(filename)

    transformer = TransformerNet()
    mse_loss = torch.nn.MSELoss()
    # l1_loss = torch.nn.L1Loss()
    transformer.to(device)

    torch.set_default_tensor_type('torch.FloatTensor')
   
    def train(steps, base_steps=0):
        transformer.train()
        count = 0
        agg_content_loss = 0.
        agg_style_loss = 0.
        agg_reg_loss = 0.   
        agg_stable_loss = 0.
        while True:
            print(0)
            for x, _ in train_loader:
                print(0)
                count += 1
                optimizer.zero_grad()
                x = x.to(device)             
                y = transformer(x)            
                with torch.no_grad():                      
                    mask = torch.bernoulli(torch.ones_like(
                        x, device=device, dtype=torch.float
                    ) * NOISE_P)
                    noise = torch.normal(
                        torch.zeros_like(x), 
                        torch.ones_like(
                            x, device=device, dtype=torch.float
                        ) * NOISE_STD
                    ).clamp(-1, 1)
                    # print((noise * mask).sum())
                y_noise = transformer(x + noise * mask)
                print(0)      
                with torch.no_grad():
                    xc = x.detach()
                    features_xc = loss_network(xc)
            
                features_y = loss_network(y)
                print(0)
                with torch.no_grad():
                    f_xc_c = features_xc[2].detach()

                content_loss = CONTENT_WEIGHT * mse_loss(features_y[2], f_xc_c)
                print(0)
                reg_loss = REGULARIZATION * (
                    torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + 
                    torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))
                print(0)
                style_loss = 0.
                for l, weight in enumerate(STYLE_WEIGHTS):
                    gram_s = gram_style[l]
                    gram_y = gram_matrix(features_y[l])
                    style_loss += float(weight) * mse_loss(gram_y, gram_s.expand_as(gram_y))
                
                stability_loss = NOISE_WEIGHT * mse_loss(y_noise.view(-1), y.view(-1).detach())
                print(0)
                total_loss = content_loss + style_loss + reg_loss + stability_loss
                total_loss.backward()
                optimizer.step()

                agg_content_loss += content_loss
                agg_style_loss += style_loss
                agg_reg_loss += reg_loss
                agg_stable_loss += stability_loss
                print(0)
                if count % LOG_INTERVAL == 0:
                    mesg = "{} [{}/{}] content: {:.2f}  style: {:.2f}  reg: {:.2f} stable: {:.2f} total: {:.6f}".format(
                                time.ctime(), count, steps,
                                agg_content_loss / LOG_INTERVAL,
                                agg_style_loss / LOG_INTERVAL,
                                agg_reg_loss / LOG_INTERVAL,
                                agg_stable_loss / LOG_INTERVAL,
                                (agg_content_loss + agg_style_loss + 
                                 agg_reg_loss + agg_stable_loss) / LOG_INTERVAL
                            )
                    print(mesg)
                    agg_content_loss = 0.
                    agg_style_loss = 0.
                    agg_reg_loss = 0.
                    agg_stable_loss = 0.
                    transformer.eval()
                    y = transformer(x)
                    save_debug_image(x, y.detach(), y_noise.detach(), "./fast_neural_style_transfer/debug/{}.png".format(base_steps + count))
                    transformer.train()
                print(0)
                if count >= steps:
                    return


    CONTENT_WEIGHT = 1
    STYLE_WEIGHTS = np.array([1e-1, 1, 1e1, 5, 1e1]) * 5e3
    REGULARIZATION = 1e-6
    NOISE_P = 0.2
    NOISE_STD = 0.35
    NOISE_WEIGHT = 10 * 2
    LOG_INTERVAL = 50

    LR = 1e-3
    optimizer = Adam(transformer.parameters(), LR)

    print("Started Training...")
    train(iterations, 0) # iteration arg
    print("Finished Training...")

    save_model_path = "./fast_neural_style_transfer/models/"+str(unique_id)+"_weights.pth"
    torch.save(transformer.state_dict(), save_model_path)

    import glob
    fnames = glob.glob(DATASET + r"/*/*")

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

    # Original content image
    #plt.imshow(recover_image(img_tensor.cpu().numpy())[0])

    # Content + Style
    #plt.imshow(recover_image(img_output.data.cpu().numpy())[0])

    output_img = Image.fromarray(recover_image(img_output.data.cpu().numpy())[0])
    output_img.save(export_img)