from flask import Flask, jsonify, request, send_file
from urllib.request import urlretrieve
from uuid import uuid4
import os, shutil

from chat.bot import Bot
from credentials import ngrok_link, ACCESS_TOKEN, VERIFY_TOKEN
from vision.gifedit import extractFrames, GIFObjectDetection
from vision import compress
from vision.methods import (
    fast_style_transfer,
    first_order_of_motion,
    foreground_removal,
    segmented_style_transfer,
)

FAKE_MOTION_OPTIONS = [
    "Putin",
    "Obama",
    "Trump",
]
#OBJECT_REMOVAL_OPTIONS = [
#    "Option 1",
#    "Option 2",
#    "Option 3",
#    "Option 4",
#    "Option 5",
#]
SEGMENTED_ST_OPTIONS = [
    "Candy",
    "Mosaic",
    "Picasso",
    "Rain Princess",
    "Starry Night",
    "Tripping Balls",
]
STYLE_TRANSFER_OPTIONS = [
    "Candy",
    "Mosaic",
    "Picasso",
    "Rain Princess",
    "Starry Night",
    "Tripping Balls",
]
IMAGE_PROCESSING_OPTIONS = [
    "Fake Motion",
    "Object Removal",
    "Style Transfer",
    "Segmented ST",
    "Finish",
]

app = Flask(__name__)
bot = Bot(ACCESS_TOKEN)
state = {"has_image": False, "selected_option": None, "uuid": None}


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
            recipient_id, "Select a process to run", IMAGE_PROCESSING_OPTIONS
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

    # frame extraction
    os.makedirs(os.path.join(os.getcwd(), "payload", state["uuid"]), exist_ok=True)
    extractFrames(
        os.path.join(os.getcwd(), "payload", f"{state['uuid']}.gif"),
        os.path.join(os.getcwd(), "payload", state["uuid"]),
    )
    # object detection
    objects_gif_arguments, OBJECT_REMOVAL_OPTIONS = GIFObjectDetection("./payload/" + str(state["uuid"]))
    print("List of objects in GIF: ", OBJECT_REMOVAL_OPTIONS)

    if message.get("text"):
        text = message["text"].lower()
        if state["selected_option"] == "fake motion":
            if text in map(lambda x: x.lower(), FAKE_MOTION_OPTIONS):
                bot.send_text(recipient_id, "Processing image, please wait")
                res_url = first_order_of_motion(state, text.replace(" ", "_"))
                state["selected_option"] = None
                bot.send_image_url(recipient_id, res_url)
                bot.send_quick_reply(
                    recipient_id,
                    "Select the next process to run",
                    IMAGE_PROCESSING_OPTIONS,
                )
                return "Continued processing"
        elif state["selected_option"] == "object removal":
            if text in map(lambda x: x.lower(), OBJECT_REMOVAL_OPTIONS):
                bot.send_text(recipient_id, "Processing image, please wait")
                objects = [objects_gif_arguments[OBJECT_REMOVAL_OPTIONS.index(text.replace(" ", "_"))]]
                res_url = foreground_removal(state, objects)
                state["selected_option"] = None
                if os.path.exists("./payload/FR_" + str(state["uuid"])):
                    integ_check_fst = len(os.listdir("./payload/FR_" + str(state["uuid"]))); integ_check_orig = len(os.listdir("./payload/" + str(state["uuid"])))
                    print("Send link if match: ", integ_check_fst, integ_check_orig)
                    if integ_check_fst > 0:
                        if integ_check_fst == integ_check_orig:
                            bot.send_image_url(recipient_id, res_url)
                            print("Link: ", res_url)
                            shutil.rmtree("./payload/" + str(state["uuid"]), ignore_errors=True)
                            shutil.rmtree("./payload/FR_" + str(state["uuid"]), ignore_errors=True)
                            bot.send_quick_reply(
                                recipient_id,
                                "Select the next process to run",
                                IMAGE_PROCESSING_OPTIONS,
                            )
                return "Continued processing"
        elif state["selected_option"] == "segmented st":
            if text in map(lambda x: x.lower(), SEGMENTED_ST_OPTIONS):
                bot.send_text(recipient_id, "Processing image, please wait")
                res_url = segmented_style_transfer(state, text.replace(" ", "_"))
                state["selected_option"] = None
                if os.path.exists("./payload/FST_" + str(state["uuid"])):
                    integ_check_fst = len(os.listdir("./payload/FST_" + str(state["uuid"]))); integ_check_orig = len(os.listdir("./payload/" + str(state["uuid"])))
                    print("Send link if match: ", integ_check_fst, integ_check_orig)
                    if integ_check_fst > 0:
                        if integ_check_fst == integ_check_orig:
                            bot.send_image_url(recipient_id, res_url)
                            print("Link: ", res_url)
                            shutil.rmtree("./payload/" + str(state["uuid"]), ignore_errors=True)
                            shutil.rmtree("./payload/FST_" + str(state["uuid"]), ignore_errors=True)
                            bot.send_quick_reply(
                                recipient_id,
                                "Select the next process to run",
                                IMAGE_PROCESSING_OPTIONS,
                            )
                return "Continued processing"
        elif state["selected_option"] == "style transfer":
            if text in map(lambda x: x.lower(), STYLE_TRANSFER_OPTIONS):
                bot.send_text(recipient_id, "Processing image, please wait")
                res_url = fast_style_transfer(state, text.replace(" ", "_"))
                state["selected_option"] = None
                if os.path.exists("./payload/FST_" + str(state["uuid"])):
                    integ_check_fst = len(os.listdir("./payload/FST_" + str(state["uuid"]))); integ_check_orig = len(os.listdir("./payload/" + str(state["uuid"])))
                    print("Send link if match: ", integ_check_fst, integ_check_orig)
                    if integ_check_fst > 0:
                        if integ_check_fst == integ_check_orig:
                            bot.send_image_url(recipient_id, res_url)
                            print("Link: ", res_url)
                            shutil.rmtree("./payload/" + str(state["uuid"]), ignore_errors=True)
                            shutil.rmtree("./payload/FST_" + str(state["uuid"]), ignore_errors=True)
                            bot.send_quick_reply(
                                recipient_id,
                                "Select the next process to run",
                                IMAGE_PROCESSING_OPTIONS,
                            )
                return "Continued processing"

        # No option is currently selected so prompt for an option
        else:
            # Handle completion of processing
            if text == "finish":
                if bot.send_text(
                    recipient_id, "Processing complete and ready for a new image"
                ):
                    # Reset the application state
                    state = {"has_image": False, "selected_option": None, "uuid": None}
                    return "Continued processing"
                else:
                    return "Failed processing"
            # Handle all the other options
            elif text in map(lambda x: x.lower(), IMAGE_PROCESSING_OPTIONS):
                state["selected_option"] = text
                # Programmatically determine which constant to use
                options = eval(f"{text.upper()} OPTIONS".replace(" ", "_"))
                bot.send_quick_reply(recipient_id, "Select an option to apply", options)
                return "Continued processing"

    # If we are here then it means an invalid command was sent
    state["selected_option"] = None
    bot.send_quick_reply(
        recipient_id, "Please send a valid command", IMAGE_PROCESSING_OPTIONS
    )
    return "Continued processing"
