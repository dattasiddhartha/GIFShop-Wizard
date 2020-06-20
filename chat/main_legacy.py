import random, os, glob, imageio, pathlib, string, uuid
from flask import Flask, request, send_from_directory, send_file
from chat.bot import Bot
from credentials import ACCESS_TOKEN, VERIFY_TOKEN, ngrok_link
import urllib.request
from PIL import Image
import numpy as np
from vision.faststyletransfer_train import FastStyleTransfer
from vision.faststyletransfer_eval import FasterStyleTransfer
from vision.segmentedstyletransfer import PartialStyleTransfer
from vision.firstordermotion import FirstOrderMotion
from vision.foregroundremoval import ForeGroundRemoval
from vision import compress
from vision.gifedit import extractFrames, stitchImages, stitch_FST_Images_ServerStyle, stitch_partialFST_Images_ServerStyle, stitch_FR

app = Flask(__name__)  # Initializing our Flask application
bot = Bot(ACCESS_TOKEN)

print("server link: ", ngrok_link)
payload_directory = "./payload/"
if not os.path.exists(payload_directory):
    os.makedirs(payload_directory)


# Importing standard route and two requst types: GET and POST.
# We will receive messages that Facebook sends our bot at this endpoint
@app.route("/", methods=["GET", "POST"])
def receive_message():
    if request.method == "GET":
        # Before allowing people to message your bot Facebook has implemented a verify token
        # that confirms all requests that your bot receives came from Facebook.
        token_sent = request.args.get("hub.verify_token")
        return verify_fb_token(token_sent)
    # If the request was not GET, it  must be POSTand we can just proceed with sending a message
    # back to user
    else:
        # get whatever message a user sent the bot
        output = request.get_json()
        for event in output["entry"]:
            messaging = event["messaging"]
            for message in messaging:
                if message.get("message"):
                    # Facebook Messenger ID for user so we know where to send response back to
                    recipient_id = message["sender"]["id"]
                    if message["message"].get("text"):
                        response_sent_text = message["message"].get("text")
                        send_message(recipient_id, response_sent_text)
                        bot.send_quick_reply_test(recipient_id)
                    # if user send us a GIF, photo, video or any other non-text item
                    if message["message"].get("attachments"):
                        response_sent_text = message["message"].get("attachments")
                        url_of_attachment = response_sent_text[0]["payload"]["url"]
                        # send_message(recipient_id, response_sent_text)
                        # send_QuickReplies(recipient_id, response_sent_text)

                        temp_filename = str(uuid.uuid4()).replace("-","_")

                        try:

                            # Note: GIF sending functionality works under stable internet connection
                            # There is a 10 seconds timeout limit for sending images
                            # Source: https://developers.facebook.com/docs/messenger-platform/reference/attachment-upload-api/

                            ## GIF callback test
                            # send_ImageBackToUser(recipient_id, url_of_attachment)

                            # vanilla FST
                            # send_FastStyleTransfer(
                            #    temp_filename,
                            #    recipient_id,
                            #    url_of_attachment,
                            #    'FST'
                            # )

                            ## segmented FST
                            # send_FastStyleTransfer(
                            #    temp_filename,
                            #    recipient_id,
                            #    url_of_attachment,
                            #    'pFST'
                            # )

                            ## deepfakes
                            #send_FirstOrderMotion(
                            #    temp_filename, 
                            #    recipient_id, 
                            #    url_of_attachment
                            #)

                            ## crop object out of background
                            send_ForegroundRemoval(
                                temp_filename, 
                                recipient_id, 
                                url_of_attachment, 
                                objects=[]
                            )

                        except:
                           send_message(recipient_id, "please wait a sec")

                        # clear up backlog of responses during development
                        send_message(recipient_id, "cleared")

    return "Message Processed"


@app.route("/file/<string:path>", methods=["GET"])
def get_file(path=""):
    return send_file("./payload/" + path)


def verify_fb_token(token_sent):
    # take token sent by Facebook and verify it matches the verify token you sent
    # if they match, allow the request, else return an error
    if token_sent == VERIFY_TOKEN:
        return request.args.get("hub.challenge")
    return "Invalid verification token"


# Send response to the user
def send_message(recipient_id, response):
    # sends user the text message provided via input response parameter
    bot.send_text_message(recipient_id, response)
    return "success"


def send_ImageBackToUser(recipient_id, url_of_attachment):
    # parse the GIF as individual images > Reverse the order of the images > Restitch into GIF > Load it as a URL to be sent (?)
    temp_filename = id_generator()
    urllib.request.urlretrieve(
        url_of_attachment, "./payload/" + str(temp_filename) + ".gif"
    )
    os.mkdir("./payload/" + str(temp_filename))
    extractFrames(
        "./payload/" + str(temp_filename) + ".gif", "./payload/" + str(temp_filename)
    )
    stitchImages("./payload/" + str(temp_filename), str(temp_filename))

    # GIF compression
    compress.resize_gif(
        "./payload/" + str(temp_filename) + "_new.gif",
        save_as="./payload/" + str(temp_filename) + "_new.gif",
        resize_to=None,
        magnitude=5,
    )

    print("Img url: ", str(ngrok_link + "/file/" + str(temp_filename) + "_new.gif"))
    bot.send_attachment_url(
        recipient_id,
        "image",
        str(ngrok_link + "/file/" + str(temp_filename) + "_new.gif"),
    )

    return "success"

# (.pth) weight files used stored in C:\Users\datta\.cache\torch\checkpoints

# with custom Training has some issues
def send_FastStyleTransfer(temp_filename, recipient_id, url_of_attachment, mode):
    # parse the GIF as individual images > Reverse the order of the images > Restitch into GIF > Load it as a URL to be sent (?)
    # temp_filename = id_generator()
    urllib.request.urlretrieve(
        url_of_attachment, "./payload/" + str(temp_filename) + ".gif"
    )
    os.mkdir("./payload/" + str(temp_filename))
    extractFrames(
        "./payload/" + str(temp_filename) + ".gif", "./payload/" + str(temp_filename)
    )

    # Client side - Issue is TORCH.DATALOADER, keeps refreshing flask
    # stitch_FST_Images_ClientStyle('./payload/'+str(temp_filename), './payload/FST_'+str(temp_filename), str(temp_filename))

    if mode == "FST":
        os.mkdir("./payload/FST_" + str(temp_filename))
        # Server side style images - pretrained models
        stitch_FST_Images_ServerStyle(
            "./payload/" + str(temp_filename),
            "./payload/FST_" + str(temp_filename),
            str(temp_filename),
        )

        # GIF compression
        compress.resize_gif(
            "./payload/" + str(temp_filename) + "_FST.gif",
            save_as="./payload/" + str(temp_filename) + "_FST.gif",
            resize_to=None,
            magnitude=5,
        )

        print(
            "Img url: ", str(ngrok_link + "/file/" + str(temp_filename) + "_FST.gif"),
        )
        bot.send_attachment_url(
            recipient_id,
            "image",
            str(ngrok_link + "/file/" + str(temp_filename) + "_FST.gif"),
        )

    if mode == "pFST":
        # partial style transfer
        stitch_partialFST_Images_ServerStyle(
            "./payload/" + str(temp_filename), str(temp_filename)
        )

        # GIF compression
        compress.resize_gif(
            "./payload/" + str(temp_filename) + "_MASK+FST.gif",
            save_as="./payload/" + str(temp_filename) + "_MASK+FST.gif",
            resize_to=None,
            magnitude=5,
        )

        print(
            "Img url: ",
            str(ngrok_link + "/file/" + str(temp_filename) + "_MASK+FST.gif"),
        )
        bot.send_attachment_url(
            recipient_id,
            "image",
            str(ngrok_link + "/file/" + str(temp_filename) + "_MASK+FST.gif"),
        )

    return "success"

def send_FirstOrderMotion(temp_filename, recipient_id, url_of_attachment):
    # i.e. DeepFakes
    urllib.request.urlretrieve(
        url_of_attachment, "./payload/" + str(temp_filename) + ".gif"
    )

    FirstOrderMotion(
        export_path="./payload/" + str(temp_filename) + "_FOM.gif",
        source_path="./vision/first_order_motion/data/02.png",
        driving_path="./payload/" + str(temp_filename) + ".gif",
        model_path="./vision/first_order_motion/vox-cpk.pth.tar",
    )

    # GIF compression
    compress.resize_gif(
        "./payload/" + str(temp_filename) + "_FOM.gif",
        save_as="./payload/" + str(temp_filename) + "_FOM.gif",
        resize_to=None,
        magnitude=5,
    )

    print(
        "Img url: ", str(ngrok_link + "/file/" + str(temp_filename) + "_FOM.gif"),
    )
    bot.send_attachment_url(
        recipient_id,
        "image",
        str(ngrok_link + "/file/" + str(temp_filename) + "_FOM.gif"),
    )


def send_ForegroundRemoval(temp_filename, recipient_id, url_of_attachment, objects=[]):

    urllib.request.urlretrieve(
        url_of_attachment, "./payload/" + str(temp_filename) + ".gif"
    )
    os.mkdir("./payload/" + str(temp_filename))
    extractFrames(
        "./payload/" + str(temp_filename) + ".gif", "./payload/" + str(temp_filename)
    )

    os.mkdir("./payload/FR_" + str(temp_filename))

    stitch_FR(
        "./payload/" + str(temp_filename),
        "./payload/FR_" + str(temp_filename),
        str(temp_filename),
        objects
    )

    # GIF compression
    compress.resize_gif(
        "./payload/" + str(temp_filename) + "_FR.gif",
        save_as="./payload/" + str(temp_filename) + "_FR.gif",
        resize_to=None,
        magnitude=5,
    )

    print(
        "Img url: ", str(ngrok_link + "/file/" + str(temp_filename) + "_FR.gif"),
    )

    try:
        bot.send_attachment_url(
            recipient_id,
            "image",
            str(ngrok_link + "/file/" + str(temp_filename) + "_FR.gif"),
        )
    except:
        bot.send_attachment_url(
            recipient_id,
            "image",
            url_of_attachment,
        )

# Add description here about this if statement.
if __name__ == "__main__":
    app.run()
