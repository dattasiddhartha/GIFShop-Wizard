import random, os, glob, imageio, pathlib, string
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
from vision import compress

app = Flask(__name__)  # Initializing our Flask application
bot = Bot(ACCESS_TOKEN)

print("server link: ", ngrok_link)

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

                        temp_filename = id_generator()

                        # try:

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
                        send_FirstOrderMotion(
                            temp_filename, recipient_id, url_of_attachment
                        )

                        # except:
                        #    send_message(recipient_id, "please wait a sec")

                        # clear up backlog of responses during development
                        # send_message(recipient_id, "cleared")

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


def stitch_FST_Images_ServerStyle(orig_dir, styled_dir, unique_filename):
    # Load frames
    filenames = []
    for f in glob.iglob(orig_dir + "/*"):
        filenames.append(f)

    # apply style transfer to each frame
    counter = 0
    for filename in filenames:
        FasterStyleTransfer(
            "./vision/fast_neural_style_transfer/models/mosaic_style__200_iter__vgg19_weights.pth",
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
    imageio.mimsave(str(orig_dir + "_FST.gif").replace("\\", "/"), images)


def stitch_partialFST_Images_ServerStyle(orig_dir, unique_filename):
    # Load frames
    filenames = []
    for f in glob.iglob(orig_dir + "/*"):
        filenames.append(f)

    # apply style transfer to each frame
    counter = 0
    for filename in filenames:
        # PartialStyleTransfer(mode = 'color', img_path=filename, style_path="./fast_neural_style_transfer/models/mosaic_style__200_iter__vgg19_weights.pth")
        PartialStyleTransfer(
            mode="styled",
            img_path=filename,
            style_path="./vision/fast_neural_style_transfer/models/mosaic_style__200_iter__vgg19_weights.pth",
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
    imageio.mimsave(str(orig_dir + "_MASK+FST.gif").replace("\\", "/"), images)


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
    imageio.mimsave(str(dir + "_FST.gif").replace("\\", "/"), images)


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


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


# Add description here about this if statement.
if __name__ == "__main__":
    app.run()
