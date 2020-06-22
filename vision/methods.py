from credentials import ngrok_link
import random, os, glob, imageio, pathlib, string, shutil, time

from vision.gifedit import (
    stitchImages,
    stitch_FST_Images_ServerStyle,
    stitch_partialFST_Images_ServerStyle,
    stitch_FR,
)

# from vision.faststyletransfer_train import FastStyleTransfer
# from vision.faststyletransfer_eval import FasterStyleTransfer
# from vision.segmentedstyletransfer import PartialStyleTransfer
# from vision.firstordermotion import FirstOrderMotion
# from vision.foregroundremoval import ForeGroundRemoval


def fast_style_transfer(state, style):
    """
    Peforms fast style transfer, updates the state, and returns whether the
    operation was successful
    """
    if os.path.exists("./payload/FST_" + str(state["uuid"])):
        if len(
            [
                name
                for name in os.listdir("./payload/FST_" + str(state["uuid"]))
                if os.path.isfile(name)
            ]
        ) == len(
            [
                name
                for name in os.listdir("./payload/" + str(state["uuid"]))
                if os.path.isfile(name)
            ]
        ):
            req_url = str(ngrok_link + "/file/" + str(state["uuid"]) + ".gif")
            shutil.rmtree("./payload/" + str(state["uuid"]), ignore_errors=True)
            shutil.rmtree("./payload/FST_" + str(state["uuid"]), ignore_errors=True)
            return req_url
    else:
        if not os.path.exists("./payload/FST_" + str(state["uuid"])):
            os.makedirs("./payload/FST_" + str(state["uuid"]))
        # Server side style images - pretrained models
        stitch_FST_Images_ServerStyle(
            "./payload/" + str(state["uuid"]),
            "./payload/FST_" + str(state["uuid"]),
            str(state["uuid"]),
            style,
        )

        # GIF compression
        compress.resize_gif(
            "./payload/" + str(state["uuid"]) + ".gif",
            save_as="./payload/" + str(state["uuid"]) + ".gif",
            resize_to=None,
            magnitude=5,
        )

        req_url = str(ngrok_link + "/file/" + str(state["uuid"]) + ".gif")

        print(
            "Img url: ", req_url,
        )

        return req_url


def first_order_of_motion(recipient_id):
    """
    Peforms first order of motion, updates the state, and returns whether the
    operation was successful
    """
    global state
    FirstOrderMotion(
        export_path="./payload/" + str(state["uuid"]) + ".gif",
        source_path="./vision/first_order_motion/data/02.png",
        driving_path="./payload/" + str(state["uuid"]) + ".gif",
        model_path="./vision/first_order_motion/vox-cpk.pth.tar",
    )

    # GIF compression
    compress.resize_gif(
        "./payload/" + str(state["uuid"]) + ".gif",
        save_as="./payload/" + str(state["uuid"]) + ".gif",
        resize_to=None,
        magnitude=5,
    )

    req_url = str(ngrok_link + "/file/" + str(state["uuid"]) + ".gif")

    print(
        "Img url: ", req_url,
    )

    time.sleep(3.0)
    shutil.rmtree("./payload/" + str(state["uuid"]), ignore_errors=True)

    return req_url


def foreground_removal(recipient_id):
    """
    Peforms foreground removal, updates the state, and returns whether the
    operation was successful
    """
    global state
    if os.path.exists("./payload/FR_" + str(state["uuid"])):
        if len(
            [
                name
                for name in os.listdir("./payload/FR_" + str(state["uuid"]))
                if os.path.isfile(name)
            ]
        ) == len(
            [
                name
                for name in os.listdir("./payload/" + str(state["uuid"]))
                if os.path.isfile(name)
            ]
        ):
            req_url = str(ngrok_link + "/file/" + str(state["uuid"]) + ".gif")
            shutil.rmtree("./payload/" + str(state["uuid"]), ignore_errors=True)
            shutil.rmtree("./payload/FR_" + str(state["uuid"]), ignore_errors=True)
            return req_url
    else:
        if not os.path.exists("./payload/FR_" + str(state["uuid"])):
            os.makedirs("./payload/FR_" + str(state["uuid"]))
        stitch_FR(
            "./payload/" + str(state["uuid"]),
            "./payload/FR_" + str(state["uuid"]),
            str(state["uuid"]),
            objects,
        )

        # GIF compression
        compress.resize_gif(
            "./payload/" + str(state["uuid"]) + ".gif",
            save_as="./payload/" + str(state["uuid"]) + ".gif",
            resize_to=None,
            magnitude=5,
        )

        req_url = str(ngrok_link + "/file/" + str(state["uuid"]) + ".gif")

        print(
            "Img url: ", req_url,
        )

        return req_url


def segmented_style_transfer(recipient_id, style):
    """
    Peforms segmented style transfer, updates the state, and returns whether
    the operation was successful
    """
    global state
    if os.path.exists("./payload/FST_" + str(state["uuid"])):
        if len(
            [
                name
                for name in os.listdir("./payload/FST_" + str(state["uuid"]))
                if os.path.isfile(name)
            ]
        ) == len(
            [
                name
                for name in os.listdir("./payload/" + str(state["uuid"]))
                if os.path.isfile(name)
            ]
        ):
            req_url = str(ngrok_link + "/file/" + str(state["uuid"]) + ".gif")
            shutil.rmtree("./payload/" + str(state["uuid"]), ignore_errors=True)
            shutil.rmtree("./payload/FST_" + str(state["uuid"]), ignore_errors=True)
            return req_url
    else:
        if not os.path.exists("./payload/FST_" + str(state["uuid"])):
            os.makedirs("./payload/FST_" + str(state["uuid"]))
        # Server side style images - pretrained models
        stitch_partialFST_Images_ServerStyle(
            "./payload/" + str(state["uuid"]), str(state["uuid"]), style
        )

        # GIF compression
        compress.resize_gif(
            "./payload/" + str(state["uuid"]) + ".gif",
            save_as="./payload/" + str(state["uuid"]) + ".gif",
            resize_to=None,
            magnitude=5,
        )

        req_url = str(ngrok_link + "/file/" + str(state["uuid"]) + ".gif")

        print(
            "Img url: ", req_url,
        )

        return req_url
