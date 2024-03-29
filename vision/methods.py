from credentials import ngrok_link
import random, os, glob, imageio, pathlib, string, shutil, time

from vision import compress
from vision.gifedit import (
    stitchImages,
    stitch_FST_Images_ServerStyle,
    stitch_partialFST_Images_ServerStyle,
    stitch_FR,
    stitch_CGAN,
)
from vision.firstordermotion import FirstOrderMotion


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
            req_url = str(ngrok_link + "/file/" + str(state["uuid"]) + "_compressed.gif")
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

        req_url = str(ngrok_link + "/file/" + str(state["uuid"]) + "_compressed.gif")

        print(
            "Img url: ", req_url,
        )

        return req_url


def first_order_of_motion(state, source):
    """
    Peforms first order of motion, updates the state, and returns whether the
    operation was successful
    """
    source_path = (
        "./vision/first_order_motion/data/"
        + str(source.lower().replace(" ", "_"))
        + ".png"
    )

    FirstOrderMotion(
        export_path="./payload/" + str(state["uuid"]) + ".gif",
        source_path=source_path,
        driving_path="./payload/" + str(state["uuid"]) + ".gif",
        model_path="./vision/first_order_motion/vox-cpk.pth.tar",
    )

    req_url = str(ngrok_link + "/file/" + str(state["uuid"]) + "_compressed.gif")

    print(
        "Img url: ", req_url,
    )

    shutil.rmtree("./payload/" + str(state["uuid"]), ignore_errors=True)

    return req_url


def foreground_removal(state, objects):
    """
    Peforms foreground removal, updates the state, and returns whether the
    operation was successful
    """
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
            req_url = str(ngrok_link + "/file/" + str(state["uuid"]) + "_compressed.gif")
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

        req_url = str(ngrok_link + "/file/" + str(state["uuid"]) + "_compressed.gif")

        print(
            "Img url: ", req_url,
        )

        return req_url


def segmented_style_transfer(state, style):
    """
    Peforms segmented style transfer, updates the state, and returns whether
    the operation was successful
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
            req_url = str(ngrok_link + "/file/" + str(state["uuid"]) + "_compressed.gif")
            return req_url
    else:
        if not os.path.exists("./payload/FST_" + str(state["uuid"])):
            os.makedirs("./payload/FST_" + str(state["uuid"]))
        # Server side style images - pretrained models
        stitch_partialFST_Images_ServerStyle(
            "./payload/" + str(state["uuid"]), str(state["uuid"]), style
        )

        req_url = str(ngrok_link + "/file/" + str(state["uuid"]) + "_compressed.gif")

        print(
            "Img url: ", req_url,
        )

        return req_url


def cycle_gan(state, style):
    """
    Runs cycle generative adversarial network, updates the state, and returns whether
    the operation was successful
    """
    stitch_CGAN(
        orig_dir="./payload/" + str(state["uuid"]),
        unique_filename=str(state["uuid"]),
        destination_style=style,
    )

    req_url = str(ngrok_link + "/file/" + str(state["uuid"]) + "_compressed.gif")

    print(
        "Img url: ", req_url,
    )

    return req_url
