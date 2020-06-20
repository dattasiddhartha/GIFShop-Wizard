# supposedly any image/video pairs should work as long as user cropped them already

import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
from vision.first_order_motion.model.demo import load_checkpoints
from vision.first_order_motion.model.demo import make_animation
from skimage import img_as_ubyte


def FirstOrderMotion(export_path = './vision/first_order_motion/output/generated_RelativeKeypointDisplacement.mp4', source_path = './vision/first_order_motion/data/02.png', driving_path = './vision/first_order_motion/data/DZ1BJU.gif', model_path = './vision/first_order_motion/vox-cpk.pth.tar'):

    # model weights: https://drive.google.com/drive/folders/1kZ1gCnpfU0BnpdU47pLM_TQ6RypDDqgw
    generator, kp_detector = load_checkpoints(config_path='./vision/first_order_motion/model/config/vox-256.yaml', checkpoint_path=model_path)

    source_image = imageio.imread(source_path)
    driving_video = imageio.mimread(driving_path)
    # Driving video can accept both .gif and .mp4 input


    #Resize image and video to 256x256
    source_image = resize(source_image, (256, 256))[..., :3]
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

    predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True,
                                 adapt_movement_scale=True)
    #predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True)
    #predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=False, adapt_movement_scale=True)

    #save resulting video
    imageio.mimsave(export_path, [img_as_ubyte(frame) for frame in predictions])

