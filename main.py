from flask import Flask, jsonify, request, send_file

from credentials import ngrok_link, ACCESS_TOKEN, VERIFY_TOKEN
from chat.bot import Bot

import random, os, glob, imageio, pathlib, string, shutil, time

from uuid import uuid4
from urllib.request import urlretrieve
from PIL import Image
import numpy as np
from vision import compress
from vision.methods import (
    fast_style_transfer,
    #    first_order_of_motion,
    #    foreground_removal,
    #    segmented_style_transfer,
)
from vision.gifedit import extractFrames

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

app = Flask(__name__)
bot = Bot(ACCESS_TOKEN)
state = {"has_image": False, "uuid": None}


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
            os.makedirs(os.path.join(os.getcwd(), "payload"), exist_ok=True)
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


@app.route("/file/<string:filename>")
def get_file(filename):
    return send_file(os.path.join(os.getcwd(), "payload", filename))


def begin_processing(recipient_id, message):
    """
    Takes in a received message, processes it, and returns a response
    """
    global state

    # If there is more than one attachment, use the first one
    if message.get("attachments") and message["attachments"][0]["type"] == "image":
        # Save GIF in `payload' directory
        url = message["attachments"][0]["payload"]["url"]
        uuid = str(uuid4()).replace("-", "_")
        urlretrieve(url, os.path.join(os.getcwd(), "payload", f"{uuid}.gif"))

        if bot.send_quick_reply(
            recipient_id,
            "How would you like to process this GIF?",
            IMAGE_PROCESSING_OPTIONS,
        ):
            state["has_image"] = True
            state["uuid"] = uuid

    # Not an image attachment
    else:
        bot.send_text(recipient_id, "Please send a GIF or an image to be processed")

    return "Began processing"


def continue_processing(recipient_id, message):
    """
    Takes in a received message, processes it, and returns a response
    """
    global state

    os.makedirs(os.path.join(os.getcwd(), "payload", state["uuid"]), exist_ok=True)
    extractFrames(
        os.path.join(os.getcwd(), "payload", f"{state['uuid']}.gif"),
        os.path.join(os.getcwd(), "payload", state["uuid"]),
    )
    if message.get("text"):
        text = message["text"].lower()
        # Check if valid option has been provided
        for option, handler in zip(IMAGE_PROCESSING_OPTIONS, IMAGE_PROCESSING_HANDLERS):
            # TEMP: refactor later
            FST_OPTIONS = [
                "Mosaic",
                "Candy",
                "Picasso",
                "Starry Night",
                "Rain Princess",
            ]
            if text in map(lambda x: x.lower(), FST_OPTIONS):
                bot.send_image_url(
                    recipient_id, fast_style_transfer(state, text.replace(" ", "_"))
                )
                bot.send_quick_reply(
                    recipient_id,
                    "(wait while image loads)\nWhat would you like to do next?",
                    FST_OPTIONS,
                )
                return "Continued processing"

            if text == option.lower():
                if True or handler(recipient_id):
                    if text == IMAGE_PROCESSING_OPTIONS[4].lower():
                        finish(recipient_id)
                        return "Finished processing"
                    else:

                        if text == IMAGE_PROCESSING_OPTIONS[0].lower():
                            print("here")
                            FST_OPTIONS = [
                                "Mosaic",
                                "Candy",
                                "Picasso",
                                "Starry Night",
                                "Rain Princess",
                            ]
                            bot.send_quick_reply(
                                recipient_id, "Which style do you want", FST_OPTIONS,
                            )
                            return "Continued processing"

                        if text == IMAGE_PROCESSING_OPTIONS[3].lower():
                            bot.send_image_url(
                                recipient_id, segmented_style_transfer(recipient_id)
                            )

                        if text == IMAGE_PROCESSING_OPTIONS[1].lower():
                            bot.send_image_url(
                                recipient_id, first_order_of_motion(recipient_id)
                            )

                        if text == IMAGE_PROCESSING_OPTIONS[2].lower():
                            bot.send_image_url(
                                recipient_id, foreground_removal(recipient_id)
                            )

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


def finish(recipient_id):
    """
    Completes image processing and returns whether the operation was successful
    """
    global state
    if bot.send_text(recipient_id, "Processing complete and ready for a new image"):
        # Reset the application state
        state = {"has_image": False, "uuid": None}
        return True
    return False
