# GIFShop Wizard <h6>[[demo]](https://m.me/104902671262259) | [[video]](https://youtu.be/A8k03a6ZeP0) | [[devpost]](https://devpost.com/software/gifshop-wizard)</h6>

*Collaborators*: [Siddhartha Datta](https://github.com/dattasiddhartha/),
[Jacky Lee](https://github.com/grenmester)

<img src="vision/first_order_motion/output/FOM.gif?raw=true" height="150px"></img>
<img src="vision/cycle_gan/datasets/zebra.gif?raw=true" height="150px"></img>
<img src="vision/fast_neural_style_transfer/videos/FST.gif?raw=true" height="150px"></img>
<img src="vision/fast_neural_style_transfer/videos/gif.gif?raw=true" height="150px"></img>

## Overview

Computer vision has been left out of the hands of many photoshopping
enthusiasts and chatbot users alike. Our mission is to bring automated
GIF-editing functionality to the masses with GIFShop Wizard.

<img src="vision/foreground_removal/input/before.jpg?raw=true" height="400px"></img>
<img src="vision/foreground_removal/input/after.jpg?raw=true" height="400px"></img>

Current features and functionality:

* Integration with Messenger API and Webhooks
* Quick Replies and Sender Actions
* Dialogue flow and state management
* Error handling and fallback options
* GIF disassembly and reassembly

* First Order of Motion :: DeepFakes, as long as driver video and content image
  are cropped in shape enough
* Foreground Removal :: remove certain objects in images
* Fast Style Transfer :: apply pretrained styles to user input images
* CycleGAN :: stylize specific objects and components of a scene
* Segmented Style Transfer :: apply instance segmentation to images and apply
  FST to those segments only

## Usage

Make sure you first update your credentials in the `credentials.py` file.

``` python
ACCESS_TOKEN = "enter your access token provided by Facebook"
VERIFY_TOKEN = "enter your personalized verify token"
ngrok_link = "your custom ngrok callback url"
```

Run the Messenger bot with the following command.

```bash
python serve.py
ngrok http 8000
```

## Additional Information

### Model Weights

Download weights from
[here](https://drive.google.com/drive/folders/1ANqflh1dxSfgdFwvH1mZqZ8_vPS6WipB?usp=sharing).

* `coco_2017` (with subdirectory val2017) placed in `vision/fast_neural_style_transfer/coco_2017/`
* `maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth` placed in `~/.cache/torch/checkpoints/`
* `vgg16-397923af.pth` placed in `~/.cache/torch/checkpoints/`
* `vgg19-dcbb9e9d.pth` placed in `~/.cache/torch/checkpoints/`
* `vox-cpk.pth.tar` placed in `vision/first_order_motion/`
* `pix2pix/checkpoints` placed in `vision/foreground_removal/`
* `yolo/checkpoints` placed in `vision/foreground_removal/`
* `cycle_gan/checkpoints` placed in `vision/`

### Style Masks

| mask name      | source img     | iterations |
|----------------|----------------|------------|
| mosaic         | mosaic         | 1000       |
| candy          | candy          | 1000       |
| picasso        | picasso        | 1000       |
| rain princess  | rain princess  | 1000       |
| starry night   | starry night   | 1000       |
| tripping       | mosaic         | 200        |
| spaghetti      | spaghetti      | 10000      |
| chocolate cake | chocolate cake | 200        |
| lasagna        | lasagna        | 200        |
| bibimbap       | bibimbap       | 200        |

### Warnings

* Network latencies: we compress the GIF (<=1.0MB) to minimize latencies in
  image sending to users
* GPU memory limits: there is a theoretical limit to the number of consecutive
  permutations users can perform on a single image; even after clearing cache,
  there is residuals left over in memory

## References

* First Order of Motion
  [[paper]](https://papers.nips.cc/paper/8935-first-order-motion-model-for-image-animation)
  [[code]](https://github.com/AliaksandrSiarohin/first-order-model)
* Foreground Removal
  [[paper]](https://arxiv.org/abs/2004.10934)
  [[code]](https://github.com/javirk/Person\_remover)
* Fast Style Transfer
  [[paper]](https://arxiv.org/abs/1603.08155)
  [[code]](https://github.com/ceshine/fast-neural-style)
* CycleGAN
  [[paper]](https://arxiv.org/abs/1703.10593)
  [[code]](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
* Segmented Style Transfer
  [[code]](https://github.com/dattasiddhartha/segmented-style-transfer) [[paper]](https://www.cv-foundation.org/openaccess/content\_cvpr\_2015/papers/Long\_Fully\_Convolutional\_Networks\_2015\_CVPR\_paper.pdf)
  [[code]](https://github.com/spmallick/learnopencv)
