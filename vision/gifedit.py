import random, os, glob, imageio, pathlib, string, uuid
import urllib.request
from PIL import Image
import numpy as np
from vision import compress
from vision.faststyletransfer_train import FastStyleTransfer
from vision.faststyletransfer_eval import FasterStyleTransfer

from vision.segmentedstyletransfer import PartialStyleTransfer
from vision.firstordermotion import FirstOrderMotion
from vision.foregroundremoval import ForeGroundRemoval, ObjectDetection
from vision.cyclegan import CYCLEGAN


def extractFrames(inGif, outFolder):
    frame = Image.open(inGif)
    nframes = 0
    while frame:
        frame.save(
            "%s/%s-%s.png" % (outFolder, os.path.basename(inGif), nframes), "GIF"
        )
        nframes += 1
        try:
            frame.seek(nframes)
        except EOFError:
            break
    return True


def stitchImages(dir, filename):
    # Load frames
    filenames = []
    for f in glob.iglob(dir + "/*"):
        filenames.append(f)

    ## Save into a GIF file that loops forever
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(str(dir + "_new.gif").replace("\\", "/"), images)


def stitch_FST_Images_ServerStyle(orig_dir, styled_dir, unique_filename, style):

    # Pass in single word to use default weights file
    # Pass in relative path to use custom weights file
    if len(style.split("/")) == 1:
        weights_path = (
            "./vision/fast_neural_style_transfer/models/"
            + str(style).lower().replace(" ", "_")
            + ".pth"
        )
    if len(style.split("/")) > 1:
        weights_path = style

    # Load frames
    filenames = []
    for f in glob.iglob(orig_dir + "/*"):
        filenames.append(f)

    # apply style transfer to each frame
    counter = 0
    for filename in filenames:
        FasterStyleTransfer(
            weights_path,
            filename,
            styled_dir + "/" + str(unique_filename) + "_FST_" + str(counter) + "_.png",
        )
        counter += 1

    ## Save into a GIF file that loops forever
    filenames_ST = []
    for f in glob.iglob(styled_dir + "/*"):
        filenames_ST.append(f)
    images = []
    for filename in filenames_ST:
        images.append(imageio.imread(filename))
    imageio.mimsave(str(orig_dir + ".gif").replace("\\", "/"), images)


def stitch_partialFST_Images_ServerStyle(orig_dir, unique_filename, style):

    # Pass in single word to use default weights file
    # Pass in relative path to use custom weights file
    if len(style.split("/")) == 1:
        weights_path = (
            "./vision/fast_neural_style_transfer/models/"
            + str(style).lower().replace(" ", "_")
            + ".pth"
        )
    if len(style.split("/")) > 1:
        weights_path = style

    # Load frames
    filenames = []
    for f in glob.iglob(orig_dir + "/*"):
        filenames.append(f)

    # apply style transfer to each frame
    counter = 0
    for filename in filenames:
        # PartialStyleTransfer(mode = 'color', img_path=filename, style_path="./fast_neural_style_transfer/models/mosaic_style__200_iter__vgg19_weights.pth")
        PartialStyleTransfer(
            mode="styled", img_path=filename, style_path=weights_path,
        )
        counter += 1

    ## Save into a GIF file that loops forever
    filenames_ST = []
    for f in glob.iglob(orig_dir + "/*"):
        if str(unique_filename) in f:
            if "_MASK+FST.png" in f:
                filenames_ST.append(f)
    images = []
    for filename in filenames_ST:
        images.append(imageio.imread(filename))
    imageio.mimsave(str(orig_dir + ".gif").replace("\\", "/"), images)


def stitch_FST_Images_ClientStyle(orig_dir, styled_dir, unique_filename):
    # Load frames
    filenames = []
    for f in glob.iglob(orig_dir + "/*"):
        filenames.append(f)

    # apply style transfer to each frame
    counter = 0
    for filename in filenames:
        FastStyleTransfer(
            "unique_id",
            50,
            filename,
            "./fast_neural_style_transfer/style_images/mosaic.jpg",
            styled_dir + "/" + str(unique_filename) + "_FST.png",
        )
        counter += 1

    ## Save into a GIF file that loops forever
    filenames_ST = []
    for f in glob.iglob(styled_dir + "/*"):
        filenames_ST.append(f)
    images = []
    for filename in filenames_ST:
        images.append(imageio.imread(filename))
    imageio.mimsave(str(dir + ".gif").replace("\\", "/"), images)


def stitch_FR(orig_dir, styled_dir, unique_filename, objects):
    # Load frames
    filenames = []
    for f in glob.iglob(orig_dir + "/*"):
        filenames.append(f)

    # apply style transfer to each frame
    counter = 0
    for filename in filenames:
        ForeGroundRemoval(
            input_path=filename,
            objects=objects,
            export_path=styled_dir
            + "/"
            + str(unique_filename)
            + "_FR_"
            + str(counter)
            + "_.png",
        )
        counter += 1

    ## Save into a GIF file that loops forever
    filenames_ST = []
    for f in glob.iglob(styled_dir + "/*"):
        filenames_ST.append(f)
    images = []
    for filename in filenames_ST:
        images.append(imageio.imread(filename))
    imageio.mimsave(str(orig_dir + ".gif").replace("\\", "/"), images)


def stitch_CGAN(orig_dir, unique_filename, destination_style):

    CYCLEGAN(
        destination_style=destination_style,
        source_image=orig_dir,
        uniqueID=str(unique_filename),
    )

    filenames = []
    for f in glob.iglob(
        "./vision/cycle_gan/results/"
        + str(destination_style)
        + "/test_latest/images/"
        + "/*"
    ):
        print(f)
        if str(unique_filename) in str(f).split("\\")[1].split("."):
            if "fake.png" in str(f).split("_"):
                filenames.append(f)

    # Save into a GIF file that loops forever
    images = []
    print(filenames)
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(str(orig_dir + ".gif").replace("\\", "/"), images)


def GIFObjectDetection(orig_dir):

    # Load frames
    filenames = []
    for f in glob.iglob(orig_dir + "/*"):
        filenames.append(f)

    # apply style transfer to each frame
    complete_list = [[], []]
    for filename in filenames:
        frame_objects_indices, frame_objects_names = ObjectDetection(
            input_path=filename
        )
        for i in range(len(frame_objects_indices)):
            complete_list[0].append(frame_objects_indices[i])
            complete_list[1].append(frame_objects_names[i])
    objects_gif_arguments = list(set(complete_list[0]))
    objects_gif_readable = list(set(complete_list[1]))
    return objects_gif_arguments, objects_gif_readable

