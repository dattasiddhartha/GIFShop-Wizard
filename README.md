# GIFShop Wizard

*Collaborators*: [Siddhartha Datta](https://github.com/dattasiddhartha/), [Jacky Lee](https://github.com/grenmester)

<!--<img src="vision/first_order_motion/output/R3S6U3_FOM.gif?raw=true" height="200px"></img> -->
<!--<img src="vision/fast_neural_style_transfer/videos/DZ1BJU_FST.gif?raw=true" height="200px"></img>-->

<img src="vision/first_order_motion/output/FOM.gif?raw=true" height="200px"></img>
<img src="vision/fast_neural_style_transfer/videos/FST.gif?raw=true" height="200px"></img>
<img src="vision/fast_neural_style_transfer/videos/gif.gif?raw=true" height="200px"></img>

Submission for Facebook Hackathon 2020 (Messenger). [[live demo]](m.me/104902671262259)

Implementing based on [functional
requirements](https://docs.google.com/document/d/1T6mk4aypOCCCxcz2EJtfLNoait8uimbkMFGiEmzRvdg/edit).

Current functionality:

* Receive messages from user, replies with text (non-NLP)
* Quick replies
* Receive user GIF as payload url, and can successfully reply back the same GIF
  (parsing GIFs & modifying them)
* Fast Style Transfer (we store pretrained styles to be applied to user input images)
* Segmented Style Transfer (we apply instance segmentation to images and apply
  FST to those segments only)
* First Order of Motion (i.e. DeepFakes, as long as driver video and content image are cropped in shape enough)
* Foreground Removal
* Supports both static images and GIFs

Next steps:

* Dialogue flow
* Error handling


## Usage

Make sure you first update your credentials in the `credentials.py` file.
```
ACCESS_TOKEN = "enter your access token provided by Facebook"
VERIFY_TOKEN = "enter your personalized verify token"
ngrok_link = "your custom ngrok callback url"
```

Run the chatbot with the following command.

```bash
python serve.py
ngrok http 8000
```

Remember to update the [Messenger
Webhook](https://developers.facebook.com/apps/309273736750794/messenger/settings/)
when running the `ngrok` service.


## Model weights & data

Download weights from [here](https://drive.google.com/drive/folders/1ANqflh1dxSfgdFwvH1mZqZ8_vPS6WipB?usp=sharing).

* <i>coco_2017</i> (with subdirectory val2017) placed in <i>./vision/fast_neural_style_transfer/coco_2017</i>
* <i>maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth</i> placed in <i>C:/Users/YOUR_COMPUTER/.cache/torch/checkpoints/</i>
* <i>vgg16-397923af.pth</i> placed in <i>C:/Users/YOUR_COMPUTER/.cache/torch/checkpoints/</i>
* <i>vgg19-dcbb9e9d.pth</i> placed in <i>C:/Users/YOUR_COMPUTER/.cache/torch/checkpoints/</i>
* <i>vox-cpk.pth.tar</i> placed in <i>./vision/first_order_motion/</i>
* <i>pix2pix/checkpoints</i> placed in <i>./vision/foreground_removal/</i>
* <i>yolo/checkpoints</i> placed in <i>./vision/foreground_removal/</i>

## Style masks

| Mask name | Source img | iterations | Trained against |
| Shroom Trip | mosaic | 200 | vgg |
| mosaic | mosaic | 1000 | vgg |

### Warnings:

* When appending shared docs on public files, be careful about the information
  they or their history contain
* When uploading repo files, be careful with tokens / secrets [might be best to
  create a separate repo later]
