import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from vision.foreground_removal.pix2pix.utils.dataset import remove_portion, normalize
from vision.foreground_removal.pix2pix.utils.model import Pix2Pix
import matplotlib.pyplot as plt
from vision.foreground_removal.yolo.detect import prepare_detector, detect
from vision.foreground_removal.yolo.yolov3.utils import draw_outputs
from vision.foreground_removal.yolo.yolov3.dataset import transform_images
import cv2, math, os
import numpy as np

def read_image(input):
    img = tf.io.read_file(input)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    
    return img

def prepare_image_yolo(img, size=416):
    img = tf.expand_dims(img, 0)
    img = transform_images(img, size)
    
    return img

def cut_result(output):
    '''
    This function reshapes the tf tensors.
    '''
    boxes, scores, classes, nums = output[0], output[1], output[2], output[3]
    
    amount = nums.numpy()[0]
    boxes = boxes[0,:amount,:]
    scores = scores[0,:amount]
    classes = classes[0, :amount]
    
    return [boxes, scores, classes, nums]

def expand_box(x1y1, x2y2, coeff_area=1.05):
    coeff_length = math.sqrt(coeff_area)
    
    x1y1 = (int(x1y1[0] * (2-coeff_length)), int(x1y1[1] * (2-coeff_length)))
    x2y2 = (int(x2y2[0] * coeff_length), int(x2y2[1] * coeff_length))
    
    return x1y1, x2y2

def get_person_image(img, box, expand=False):
    wh = np.flip(img.shape[0:2])
    # Posiciones de la caja. x1y1 es abajo a la izquierda y x2y2 es arriba a la derecha
    x1y1 = tuple((np.array(box[0:2]) * wh).astype(np.int32))
    x2y2 = tuple((np.array(box[2:4]) * wh).astype(np.int32))
    
    if expand:
        x1y1, x2y2 = expand_box(x1y1, x2y2)

    box_width = x2y2[1] - x1y1[1]
    box_height = x2y2[0] - x1y1[0]

    box_border_1 = (x1y1[0] - box_height // 2, x1y1[1] - box_width // 2)
    box_border_2 = (x2y2[0] + box_height // 2, x2y2[1] + box_width // 2)
    
    im_cut = img[box_border_1[1]:box_border_2[1], box_border_1[0]:box_border_2[0], :]
    im_cut = tf.image.resize(im_cut, [HEIGHT, WIDTH], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    #return im_cut, box_border_1, box_border_2
    return im_cut, x1y1, x2y2

def generate_fake(img, model):
    # img = tf.image.resize(img, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img = normalize(img)
    fake = remove_portion(img, 256, 256)
    fake = tf.expand_dims(fake, 0)
    
    return model.generator(fake)[0]

def insert_into_image(img_real, img_fake, x1y1, x2y2):
    img_fake = img_fake[WIDTH // 4:WIDTH * 3 // 4, HEIGHT // 4:HEIGHT * 3 // 4,:]
    x1y1 = (max(0, x1y1[0]), max(0, x1y1[1]))
    x2y2 = (min(img_real.shape[1], x2y2[0]), min(img_real.shape[0], x2y2[1]))
    
    box_width = x2y2[1] - x1y1[1]
    box_height = x2y2[0] - x1y1[0]
    img_fake_resized = tf.image.resize(img_fake, [box_width, box_height], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img_fake = img_fake.numpy()
    
    img_real[x1y1[1]:x2y2[1], x1y1[0]:x2y2[0], :] = img_fake_resized[:]
    
    return img_real

def insert_blocks(img_real, blocks, coords):
    img_numpy = img_real.numpy()
    for block, coord in zip(blocks, coords):
        x1y1 = coord[0]
        x2y2 = coord[1]
        img_numpy = insert_into_image(img_numpy, block, x1y1, x2y2)
        
    return img_numpy

def create_new_image(img_real, output_yolo, objects):
    img_fake = normalize(img_real)
    boxes, scores, classes, nums = output_yolo[0], output_yolo[1], output_yolo[2], output_yolo[3]
    blocks = []
    coords = []

    for i in range(nums[0]):
        if classes[i].numpy() in objects:
            im_cut, x1y1, x2y2 = get_person_image(img_real, boxes[i])
            fake_block = generate_fake(im_cut, p2p)

            blocks.append(fake_block)
            coords.append((x1y1, x2y2))

    new = insert_blocks(img_fake, blocks, coords)
    return new

yolo = prepare_detector()

classes = './vision/foreground_removal/yolo/data/coco.names'
class_names = [c.strip() for c in open(classes).readlines()]
N_people = 5
HEIGHT = 256
WIDTH = 256

p2p = Pix2Pix(mode='try', checkpoint_dir='./vision/foreground_removal/pix2pix/checkpoint/')

coco_objects_list = ['person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa','pottedplant','bed','diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

def ForeGroundRemoval(input_path = './vision/foreground_removal/input/person_test.jpg', objects = [], export_path = './vision/foreground_removal/output/G.png'):

    image = read_image(input_path) # arg 1: input image
    image_yolo = prepare_image_yolo(image)
    objects_detected = list(set(detect(image_yolo, yolo)[2][0].numpy())) # auto-detected objects, return list

    # hard-coded objects, if user set mannually
    required_removed_objects = objects # style: ['car'] # arg 2
    req_objects_indices=[]
    for i in range(len(coco_objects_list)):
        if coco_objects_list[i] in required_removed_objects:
            req_objects_indices.append(i)

    output_yolo = yolo(image_yolo)
    output_yolo = cut_result(output_yolo)

    if len(objects) == 0:
        print("Objects detected: ", objects_detected)
        final_image = create_new_image(image, output_yolo, objects_detected) # arg 2: objects
    
    if len(objects) > 0:
        final_image = create_new_image(image, output_yolo, req_objects_indices) # arg 2: objects

    plt.imsave(export_path,final_image*0.5+0.5) # arg 3: export


# https://github.com/javirk/Person_remover