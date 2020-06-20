from flask import Flask, jsonify, request, send_file

from credentials import ngrok_link, ACCESS_TOKEN, VERIFY_TOKEN
from chat.bot import Bot

import random, os, glob, imageio, pathlib, string, uuid, shutil, time

import urllib.request
from PIL import Image
import numpy as np
from vision import compress
from vision.gifedit import extractFrames, stitchImages, stitch_FST_Images_ServerStyle, stitch_partialFST_Images_ServerStyle, stitch_FR
from vision.faststyletransfer_train import FastStyleTransfer
from vision.faststyletransfer_eval import FasterStyleTransfer
from vision.segmentedstyletransfer import PartialStyleTransfer
from vision.firstordermotion import FirstOrderMotion
from vision.foregroundremoval import ForeGroundRemoval



IMAGE_PROCESSING_OPTIONS = [
    "Style Transfer",
    "Fake Motion",
    "Object Removal",
    "Segmented ST",
    "Finish",
]

IMAGE_PROCESSING_HANDLERS = [
    lambda x: fast_style_transfer(x),
    lambda x: first_order_of_motion(x),
    lambda x: foreground_removal(x),
    lambda x: segmented_style_transfer(x),
    lambda x: finish(x),
]

if not os.path.exists("./payload/"):
    os.makedirs("./payload/")


app = Flask(__name__)
bot = Bot(ACCESS_TOKEN)
state = {
    "has_image": False,
    "url": None,
    "uuid" : None
}


@app.route("/", methods=["GET", "POST"])
def process_request():
    """
    Main webhook endpoint for bot
    """
    global state
    # This route is only for the initial bot verification
    if request.method == "GET":
        # Facebook requires a verify token to be verified before allowing
        # requests sent in Messenger to be directed to this bot
        if request.args.get("hub.verify_token") == VERIFY_TOKEN:
            return request.args.get("hub.challenge")
        else:
            return "Invalid verification token"

    # This route is for processing messages sent to the bot
    if request.method == "POST":
        data = request.get_json()
        responses = []

        for entry in data["entry"]:
            # The array will always only contain one messaging object
            messaging = entry["messaging"][0]
            # Respond to a message from a user
            if messaging.get("message"):
                message = messaging["message"]
                sender_id = messaging["sender"]["id"]

                if not state["has_image"]:
                    response = begin_processing(sender_id, message)
                else:
                    response = continue_processing(sender_id, message)
            # Ignore actions that are not messages
            else:
                response = "Action ignored"

            responses.append(response)
        return jsonify(responses)


@app.route("/file/<string:path>", methods=["GET"])
def get_file(path=""):
    return send_file("./payload/" + path)


def begin_processing(recipient_id, message):
    """
    Takes in a received message, processes it, and returns a response
    """
    global state
    # If there is more than one attachment, use the first one
    if message.get("attachments") and message["attachments"][0]["type"] == "image":
        url = message["attachments"][0]["payload"]["url"]
        #filetype = get_filetype(url)

        # Process GIF and images
        #if filetype in ["gif", "png", "jpeg", "jpg"]:
        temp_filename = str(uuid.uuid4()).replace("-","_")
        # parse the GIF as individual images > Reverse the order of the images > Restitch into GIF > Load it as a URL to be sent (?)
        urllib.request.urlretrieve(
            url, "./payload/" + str(temp_filename) + ".gif"
        )
        if bot.send_quick_reply(
            recipient_id,
            "How would you like to process this GIF?",
            IMAGE_PROCESSING_OPTIONS,
        ):
            state["has_image"] = True
            state["url"] = url
            state["uuid"] = temp_filename

    # Not an image attachment
    else:
        bot.send_text(recipient_id, "Please send a GIF or an image to be processed")

    return "Began processing"


def continue_processing(recipient_id, message):
    """
    Takes in a received message, processes it, and returns a response
    """
    global state
    if not os.path.exists("./payload/FR_" + str(state['uuid'])):
        os.makedirs("./payload/" + str(state["uuid"]))
    extractFrames(
        "./payload/" + str(state["uuid"]) + ".gif", "./payload/" + str(state["uuid"])
    )
    if message.get("text"):
        text = message["text"].lower()
        # Check if valid option has been provided
        for option, handler in zip(IMAGE_PROCESSING_OPTIONS, IMAGE_PROCESSING_HANDLERS):
            if text == option.lower():
                if handler(recipient_id):
                    if text == IMAGE_PROCESSING_OPTIONS[4].lower():
                        finish(recipient_id)
                        return "Finished processing"
                    else:                   

                        if text == IMAGE_PROCESSING_OPTIONS[0].lower():
                            bot.send_image_url(recipient_id, fast_style_transfer(recipient_id))

                        if text == IMAGE_PROCESSING_OPTIONS[3].lower():
                            bot.send_image_url(recipient_id, segmented_style_transfer(recipient_id))

                        if text == IMAGE_PROCESSING_OPTIONS[1].lower():
                            bot.send_image_url(recipient_id, first_order_of_motion(recipient_id))

                        if text == IMAGE_PROCESSING_OPTIONS[2].lower():
                            bot.send_image_url(recipient_id, foreground_removal(recipient_id))

                        
                        bot.send_quick_reply(
                            recipient_id,
                            "(wait while image loads)\nWhat would you like to do next?",
                            IMAGE_PROCESSING_OPTIONS,
                        )
                        return "Continued processing"
                else:
                    return "Failed processing"
        # No valid option was selected
        bot.send_quick_reply(
            recipient_id, "Please send a valid command", IMAGE_PROCESSING_OPTIONS,
        )
    else:
        bot.send_quick_reply(
            recipient_id, "Please send a command", IMAGE_PROCESSING_OPTIONS,
        )
    return "Continued processing"


def fast_style_transfer(recipient_id):
    """
    Peforms fast style transfer, updates the state, and returns whether the
    operation was successful
    """
    global state
    if os.path.exists("./payload/FST_" + str(state['uuid'])):
        if len([name for name in os.listdir("./payload/FST_" + str(state['uuid'])) if os.path.isfile(name)]) == len([name for name in os.listdir("./payload/" + str(state['uuid'])) if os.path.isfile(name)]):
            req_url = str(ngrok_link + "/file/" + str(state['uuid']) + ".gif")
            shutil.rmtree("./payload/" + str(state['uuid']), ignore_errors=True); shutil.rmtree("./payload/FST_" + str(state['uuid']), ignore_errors=True)
            return req_url
    else:
        if not os.path.exists("./payload/FST_" + str(state['uuid'])):
            os.makedirs("./payload/FST_" + str(state['uuid']))
        # Server side style images - pretrained models
        stitch_FST_Images_ServerStyle(
            "./payload/" + str(state['uuid']),
            "./payload/FST_" + str(state['uuid']),
            str(state['uuid']),
        )

        # GIF compression
        compress.resize_gif(
            "./payload/" + str(state['uuid']) + ".gif",
            save_as="./payload/" + str(state['uuid']) + ".gif",
            resize_to=None,
            magnitude=5,
        )

        req_url = str(ngrok_link + "/file/" + str(state['uuid']) + ".gif")

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
        export_path="./payload/" + str(state['uuid']) + ".gif",
        source_path="./vision/first_order_motion/data/02.png",
        driving_path="./payload/" + str(state['uuid']) + ".gif",
        model_path="./vision/first_order_motion/vox-cpk.pth.tar",
    )

    # GIF compression
    compress.resize_gif(
        "./payload/" + str(state['uuid']) + ".gif",
        save_as="./payload/" + str(state['uuid']) + ".gif",
        resize_to=None,
        magnitude=5,
    )

    req_url = str(ngrok_link + "/file/" + str(state['uuid']) + ".gif")

    print(
        "Img url: ", req_url,
    )

    time.sleep(3.0); shutil.rmtree("./payload/" + str(state['uuid']), ignore_errors=True)

    return req_url


def foreground_removal(recipient_id):
    """
    Peforms foreground removal, updates the state, and returns whether the
    operation was successful
    """
    global state
    if os.path.exists("./payload/FR_" + str(state['uuid'])):
        if len([name for name in os.listdir("./payload/FR_" + str(state['uuid'])) if os.path.isfile(name)]) == len([name for name in os.listdir("./payload/" + str(state['uuid'])) if os.path.isfile(name)]):
            req_url = str(ngrok_link + "/file/" + str(state['uuid']) + ".gif")
            shutil.rmtree("./payload/" + str(state['uuid']), ignore_errors=True); shutil.rmtree("./payload/FR_" + str(state['uuid']), ignore_errors=True)
            return req_url
    else:
        if not os.path.exists("./payload/FR_" + str(state['uuid'])):
            os.makedirs("./payload/FR_" + str(state['uuid']))
        stitch_FR(
            "./payload/" + str(state['uuid']),
            "./payload/FR_" + str(state['uuid']),
            str(state['uuid']),
            objects
        )

        # GIF compression
        compress.resize_gif(
            "./payload/" + str(state['uuid']) + ".gif",
            save_as="./payload/" + str(state['uuid']) + ".gif",
            resize_to=None,
            magnitude=5,
        )

        req_url = str(ngrok_link + "/file/" + str(state['uuid']) + ".gif")

        print(
            "Img url: ", req_url,
        )

        return req_url


def segmented_style_transfer(recipient_id):
    """
    Peforms segmented style transfer, updates the state, and returns whether
    the operation was successful
    """
    global state
    if os.path.exists("./payload/FST_" + str(state['uuid'])):
        if len([name for name in os.listdir("./payload/FST_" + str(state['uuid'])) if os.path.isfile(name)]) == len([name for name in os.listdir("./payload/" + str(state['uuid'])) if os.path.isfile(name)]):
            req_url = str(ngrok_link + "/file/" + str(state['uuid']) + ".gif")
            shutil.rmtree("./payload/" + str(state['uuid']), ignore_errors=True); shutil.rmtree("./payload/FST_" + str(state['uuid']), ignore_errors=True)
            return req_url
    else:
        if not os.path.exists("./payload/FST_" + str(state['uuid'])):
            os.makedirs("./payload/FST_" + str(state['uuid']))
        # Server side style images - pretrained models
        stitch_partialFST_Images_ServerStyle(
            "./payload/" + str(state['uuid']),
            str(state['uuid']),
        )

        # GIF compression
        compress.resize_gif(
            "./payload/" + str(state['uuid']) + ".gif",
            save_as="./payload/" + str(state['uuid']) + ".gif",
            resize_to=None,
            magnitude=5,
        )

        req_url = str(ngrok_link + "/file/" + str(state['uuid']) + ".gif")

        print(
            "Img url: ", req_url,
        )

        return req_url


def finish(recipient_id):
    """
    Completes image processing and returns whether the operation was successful
    """
    global state
    if bot.send_text(recipient_id, "Processing complete and ready for a new image"):
        # Reset the application state
        state = {
            "has_image": False,
            "url": None,
            "uuid" : None
        }
        return True
    return False
