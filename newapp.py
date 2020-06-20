from flask import Flask, jsonify, request

from credentials import ngrok_link, ACCESS_TOKEN, VERIFY_TOKEN
from utils.bot import Bot


IMAGE_PROCESSING_OPTIONS = [
    "Fast Style Transfer",
    "First Order of Motion",
    "Foreground Removal",
    "Segmented Style Transfer",
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
state = {
    "has_image": False,
    "url": None,
    "filetype": None,
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


def get_filetype(url):
    """
    Takes in a URL and returns the filetype of the resource
    """
    # TODO: Write function
    return "png"


def begin_processing(recipient_id, message):
    """
    Takes in a received message, processes it, and returns a response
    """
    global state
    # If there is more than one attachment, use the first one
    if message.get("attachments") and message["attachments"][0]["type"] == "image":
        url = message["attachments"][0]["payload"]["url"]
        filetype = get_filetype(url)

        # Process image
        if filetype in ["jpeg", "jpg", "png"]:
            if bot.send_quick_reply(
                recipient_id,
                "How would you like to process this image?",
                IMAGE_PROCESSING_OPTIONS,
            ):
                state["has_image"] = True
                state["url"] = url
                state["filetype"] = filetype

        # Process GIF
        elif filetype == "gif":
            if bot.send_quick_reply(
                recipient_id,
                "How would you like to process this GIF?",
                IMAGE_PROCESSING_OPTIONS,
            ):
                state["has_image"] = True
                state["url"] = url
                state["filetype"] = filetype

        # Unrecognized filetype
        else:
            bot.send_text(
                recipient_id,
                "Media not recognized, please send a GIF or an image to be processed",
            )

    # Not an image attachment
    else:
        bot.send_text(recipient_id, "Please send a GIF or an image to be processed")

    return "Began processing"


def continue_processing(recipient_id, message):
    """
    Takes in a received message, processes it, and returns a response
    """
    global state
    if message.get("text"):
        text = message["text"].lower()
        # Check if valid option has been provided
        for option, handler in zip(IMAGE_PROCESSING_OPTIONS, IMAGE_PROCESSING_HANDLERS):
            if text == option:
                if handler(recipient_id):
                    bot.send_image_url(recipient_id, state["url"])
                    bot.send_quick_reply(
                        recipient_id,
                        "What would you like to do next?",
                        IMAGE_PROCESSING_OPTIONS,
                    )
                break
        # No valid option was selected
        bot.send_text(recipient_id, "Please send a valid command")
    else:
        bot.send_text(recipient_id, "Please send a command")
    return "Continued processing"


def fast_style_transfer(recipient_id):
    """
    Peforms fast style transfer, updates the state, and returns whether the
    operation was successful
    """
    return True


def first_order_of_motion(recipient_id):
    """
    Peforms first order of motion, updates the state, and returns whether the
    operation was successful
    """
    return True


def foreground_removal(recipient_id):
    """
    Peforms foreground removal, updates the state, and returns whether the
    operation was successful
    """
    return True


def segmented_style_transfer(recipient_id):
    """
    Peforms segmented style transfer, updates the state, and returns whether
    the operation was successful
    """
    return True


def finish(recipient_id):
    """
    Completes image processing and returns whether the operation was successful
    """
    global state
    if bot.send_text(recipient_id, "Processing complete"):
        # Reset the application state
        state = {
            "has_image": False,
            "url": None,
            "filetype": None,
        }
        return True
    return False
