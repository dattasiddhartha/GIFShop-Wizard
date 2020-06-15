# import necessary libraries
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib, random
import torch, torchvision
import torchvision.transforms as T
import numpy as np
import numpy.ma as ma
import cv2
from faststyletransfer_eval import FasterStyleTransfer

# Notes before running the bot: 
# 1. Reconnect calback url
# 2. Reset ngrok link in credentials file


# get the pretrained model from torchvision.models
# Note: pretrained=True will get the pretrained weights for the model.
# model.eval() to use the model for inference
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# default COCO objects
# Separating out the object names will be useful in object-specific filtering, but not instance segmentation
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def random_colour_masks(image):
  """
  random_colour_masks
    parameters:
      - image - predicted masks
    method:
      - the masks of each predicted object is given random colour for visualization
  """
  colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,10)]
  coloured_mask = np.stack([r, g, b], axis=2)
  return coloured_mask

#def clearBackgroundColour(style_img, style_img_index = 0, number_of_checks = 5, rejection_threshold=2):
#def clearBackgroundColour(style_img, style_img_index, number_of_checks, rejection_threshold):

#    pix_value = []
#    for i in range(number_of_checks):
#        pix_value.append(float(style_img[0][int(random.uniform(0,style_img.shape[1]))][int(random.uniform(0,style_img.shape[2]))]))
        
#    count_heuristic = list(set(pix_value))
#    if len(count_heuristic) <= rejection_threshold:
#        for x in set(pix_value):
#            if pix_value.count(x) >rejection_threshold:
#                pix_remove = x
    
#    masked_img = []
#    for i in range(style_img[style_img_index].shape[0]):
#        tmp=[]
#        for j in range(style_img[style_img_index].shape[1]):
#            if float(style_img[style_img_index][i][j]) == float(pix_remove):
#                tmp.append(float(0))
#            else:
#                tmp.append(style_img[style_img_index][i][j])
#        masked_img.append(tmp)
        
#    return masked_img, pix_remove

def get_prediction(img_path, threshold, objects):
  """
  get_prediction
    parameters:
      - img_path - path of the input image
    method:
      - Image is obtained from the image path
      - the image is converted to image tensor using PyTorch's Transforms
      - image is passed through the model to get the predictions
      - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
        ie: eg. segment of cat is made 1 and rest of the image is made 0
    
  """
  img = Image.open(img_path)
  transform = T.Compose([T.ToTensor()])
  img = transform(img)
  pred = model([img])
  pred_score = list(pred[0]['scores'].detach().numpy())
  pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
  masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
  pred_class = [objects[i] for i in list(pred[0]['labels'].numpy())]
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
  masks = masks[:pred_t+1]
  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]
  return masks, pred_boxes, pred_class

def instance_segmentation_api(img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3, objects=COCO_INSTANCE_CATEGORY_NAMES):
  """
  instance_segmentation_api
    parameters:
      - img_path - path to input image
    method:
      - prediction is obtained by get_prediction
      - each mask is given random color
      - each mask is added to the image in the ration 1:0.8 with opencv
      - final output is displayed
  """
  masks, boxes, pred_cls = get_prediction(img_path, threshold, objects)
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  for i in range(len(masks)):
    rgb_mask = random_colour_masks(masks[i])
    img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
    #cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) # no bounding boxes required
    cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
  plt.figure(figsize=(20,30))
  plt.imshow(img)
  plt.xticks([])
  plt.yticks([])
  plt.show()

  return img


def PartialStyleTransfer(mode = 'styled', img_path='./payload/IMG-20200401-WA0002.jpg', style_path="./fast_neural_style_transfer/models/mosaic_style__200_iter__vgg19_weights.pth"):

    print("Started partial style transfer")

    # mode can be 'styled' or 'color'

    img_original = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img_original)
    pred = model([img])

    print("Finished image segmentation")

    # vanilla random color image segmentation
    if mode == 'color':
        img_masked = instance_segmentation_api(img_path=img_path, threshold=0.75, rect_th=0, text_size=0, text_th=0, objects=COCO_INSTANCE_CATEGORY_NAMES)
        matplotlib.image.imsave(str(img_path[:-4]+str("_MASK")+".png"), img_masked)

    if mode == 'styled':

        #print(pred[0]['masks'])

        segment = 0 # assuming a mask exists -- set exception handling for no detected masks
        mask = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()[segment]

        # print mask of image with the original image pixels
        img_array = np.array(img[0])
        # if False, set as 0 (black)
        #print(img_array.shape)
        # img_array[img_array.shape[0]-1][img_array.shape[1]-1]

        masked_img = []
        for i in range(img_array.shape[0]):
            tmp=[]
            for j in range(img_array.shape[1]):
                if mask[i][j] == False:
                    tmp.append(float(0))
                else:
                    tmp.append(img_array[i][j])
            masked_img.append(tmp)
    
        masked_img_array = np.array(masked_img)
        # plt.imshow(masked_img_array) # Export this mask image for style transfer

        matplotlib.image.imsave(str(img_path[:-4]+str("_MASK")+".png"), masked_img_array)

        FasterStyleTransfer(style_path, str(img_path[:-4]+str("_MASK")+".png"), str(img_path[:-4]+str("_FST")+".png"))

        style_img = Image.open(str(img_path[:-4]+str("_FST")+".png"))
        style_img = style_img.convert('RGB')
        transform = T.Compose([T.ToTensor()])
        style_img = transform(style_img)

        def clearBackgroundColour(style_img=style_img, style_img_index = 0, number_of_checks = 5, rejection_threshold1=2, rejection_threshold2=2):

            pix_value = []
            for i in range(number_of_checks):
                #pix_value.append(float(style_img[0][int(random.uniform(0,style_img.shape[1]))][int(random.uniform(0,style_img.shape[2]))]))
                val = float("{:.2f}".format(style_img[0][int(random.uniform(0,style_img.shape[1]))][int(random.uniform(0,style_img.shape[2]))]))
                pix_value.append(val)
        
            count_heuristic = list(set(pix_value))
            if len(count_heuristic) <= rejection_threshold1:
                for x in set(pix_value):
                    if pix_value.count(x) > rejection_threshold2:
                        pix_remove = x
    
            masked_img = []
            for i in range(style_img[style_img_index].shape[0]):
                tmp=[]
                for j in range(style_img[style_img_index].shape[1]):
                    if float(style_img[style_img_index][i][j]) == float(pix_remove):
                        tmp.append(float(0))
                    else:
                        tmp.append(style_img[style_img_index][i][j])
                masked_img.append(tmp)
        
            return np.array(masked_img), pix_remove

        #masked_img, pix_remove = clearBackgroundColour(style_img=style_img, style_img_index = 0, number_of_checks = 10, rejection_threshold=5)
        #masked_img_array = np.array(masked_img)

        masked_img_array, pix_remove = clearBackgroundColour(style_img=style_img, style_img_index = 0, number_of_checks = 100, rejection_threshold1=50, rejection_threshold2=5)
        #plt.imshow(masked_img_array)
        #plt.show() # has some background around the sides, mostly because 0.540 and 0.535 are not exactly the same values

        # reshaping (Assuming style image will be larger than content image)
        req_image = img[0]
        masked_img_array_reshaped = masked_img_array[0:len(req_image)]

        masked_img_array_reshaped_2 = []
        for index in range(len(masked_img_array_reshaped)):
            masked_img_array_reshaped_2.append(masked_img_array_reshaped[index][0:len(req_image[0])])
        masked_img_array_reshaped_2 = np.array(masked_img_array_reshaped_2)

        combined_masks = ma.masked_array(req_image, masked_img_array_reshaped_2>0) # clears out pixels so that we can superimpose
        # split the image as 50% focus on masked_img_array, 50% on combined_masks
        masked_img = cv2.addWeighted(np.array(masked_img_array_reshaped_2, np.float64), 0.5, np.array(combined_masks, np.float64), 0.5, 0)
        matplotlib.image.imsave(str(img_path[:-4]+str("_MASK+FST")+".png"), masked_img_array)

