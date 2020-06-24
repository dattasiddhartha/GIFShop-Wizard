from flask import Flask, jsonify, request, send_file
from urllib.request import urlretrieve
from uuid import uuid4
import os, shutil, torch, glob

from chat.bot import Bot
from credentials import ngrok_link, ACCESS_TOKEN, VERIFY_TOKEN

from vision.gifedit import extractFrames, GIFObjectDetection
from vision import compress
from vision.methods import (
    fast_style_transfer,
    first_order_of_motion,
    foreground_removal,
    segmented_style_transfer,
    cycle_gan,
)

FAKE_MOTION_IMAGES = [
    "https://upload.wikimedia.org/wikipedia/commons/8/8d/Vladimir_Putin_%282020-02-20%29.jpg",
    "https://will.illinois.edu/images/uploads/50405/president_barack_obama.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/5/56/Donald_Trump_official_portrait.jpg",
]
STYLE_TRANSFER_IMAGES = [
    "https://inews.gtimg.com/newsapp_match/0/10560893188/0",
    "https://images.deepai.org/converted-papers/1906.01123/images/qualitive1/style_2.jpg",
    "https://www.stunewsnewport.com/images/editorial/jan2/CubismLRG.jpg",
    "https://d2halst20r4hcy.cloudfront.net/wallpapers2/079/340/083/456/original/file.jpg",
    "https://bsaber.com/wp-content/uploads/2020/01/49352bc094b7e66f706b5671d5da669a6266964d.jpg",
    "https://10mosttoday.com/wp-content/uploads/2014/12/Vincent_van_Gogh_self_portrait.jpg",
    "https://pic1.zhimg.com/v2-d82300aae5b1fd5f7303b057abca8a41_200x112.jpg",
    "https://1.bp.blogspot.com/-UMnqX3NZWxo/XX8I0MhluCI/AAAAAAAAADA/MoU7Mi8jp1Ejrsvf7bBLAuUNWOovdJP4QCLcBGAsYHQ/s1600/202685_0-803-5447-5447.jpg",
    "https://www.receiteria.com.br/wp-content/uploads/receitas-com-queijo-mussarela-0-1200x738.jpg",
    "https://asian-recipe.com/wp-content/uploads/2017/11/bibimbap-mixed-rice-1200x900.jpg",
]
SEGMENTED_ST_IMAGES = [
    "https://inews.gtimg.com/newsapp_match/0/10560893188/0",
    "https://images.deepai.org/converted-papers/1906.01123/images/qualitive1/style_2.jpg",
    "https://www.stunewsnewport.com/images/editorial/jan2/CubismLRG.jpg",
    "https://d2halst20r4hcy.cloudfront.net/wallpapers2/079/340/083/456/original/file.jpg",
    "https://bsaber.com/wp-content/uploads/2020/01/49352bc094b7e66f706b5671d5da669a6266964d.jpg",
    "https://10mosttoday.com/wp-content/uploads/2014/12/Vincent_van_Gogh_self_portrait.jpg",
    "https://pic1.zhimg.com/v2-d82300aae5b1fd5f7303b057abca8a41_200x112.jpg",
    "https://1.bp.blogspot.com/-UMnqX3NZWxo/XX8I0MhluCI/AAAAAAAAADA/MoU7Mi8jp1Ejrsvf7bBLAuUNWOovdJP4QCLcBGAsYHQ/s1600/202685_0-803-5447-5447.jpg",
    "https://www.receiteria.com.br/wp-content/uploads/receitas-com-queijo-mussarela-0-1200x738.jpg",
    "https://asian-recipe.com/wp-content/uploads/2017/11/bibimbap-mixed-rice-1200x900.jpg",
]
GAN_IMAGES = [
    "https://ychef.files.bbci.co.uk/976x549/p07v2wjn.jpg",
    "https://i0.wp.com/www.horsetalk.co.nz/wp-content/uploads/2016/08/shiny-coat-stock.jpg?resize=800%2C445",
    "https://www.atlasofplaces.com/atlas-of-places-images/ATLAS-OF-PLACES-CLAUDE-MONET-LA-LUMIE%CC%80RE-GPH-2.jpg",
    "https://www.holland.com/upload_mm/3/d/9/68950_fullimage_vangogh-portert-1360.jpg",
    "https://www.nationalgeographic.com/content/dam/travel/2019-digital/yosemite-guide/yosemite-national-park-california.jpg",
]
FAKE_MOTION_OPTIONS = [
    "Putin",
    "Obama",
    "Trump",
]
FAKE_MOTION_IMAGES = [
    "https://upload.wikimedia.org/wikipedia/commons/8/8d/Vladimir_Putin_%282020-02-20%29.jpg",
    "https://will.illinois.edu/images/uploads/50405/president_barack_obama.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/5/56/Donald_Trump_official_portrait.jpg",
]
STYLE_TRANSFER_OPTIONS = [
    "Candy",
    "Mosaic",
    "Picasso",
    "Rain Princess",
    "Starry Night",
    "Tripping",
    "Spaghetti",
    "Chocolate Cake",
    "Lasagna",
    "Bibimbap",
]
STYLE_TRANSFER_IMAGES = [
    "https://inews.gtimg.com/newsapp_match/0/10560893188/0",
    "https://images.deepai.org/converted-papers/1906.01123/images/qualitive1/style_2.jpg",
    "https://www.stunewsnewport.com/images/editorial/jan2/CubismLRG.jpg",
    "https://d2halst20r4hcy.cloudfront.net/wallpapers2/079/340/083/456/original/file.jpg",
    "https://bsaber.com/wp-content/uploads/2020/01/49352bc094b7e66f706b5671d5da669a6266964d.jpg",
    "https://10mosttoday.com/wp-content/uploads/2014/12/Vincent_van_Gogh_self_portrait.jpg",
    "https://pic1.zhimg.com/v2-d82300aae5b1fd5f7303b057abca8a41_200x112.jpg",
    "https://1.bp.blogspot.com/-UMnqX3NZWxo/XX8I0MhluCI/AAAAAAAAADA/MoU7Mi8jp1Ejrsvf7bBLAuUNWOovdJP4QCLcBGAsYHQ/s1600/202685_0-803-5447-5447.jpg",
    "https://www.receiteria.com.br/wp-content/uploads/receitas-com-queijo-mussarela-0-1200x738.jpg",
    "https://asian-recipe.com/wp-content/uploads/2017/11/bibimbap-mixed-rice-1200x900.jpg",
]
SEGMENTED_ST_OPTIONS = [
    "Candy",
    "Mosaic",
    "Picasso",
    "Rain Princess",
    "Starry Night",
    "Tripping",
    "Spaghetti",
    "Chocolate Cake",
    "Lasagna",
    "Bibimbap",
]
SEGMENTED_ST_IMAGES = [
    "https://inews.gtimg.com/newsapp_match/0/10560893188/0",
    "https://images.deepai.org/converted-papers/1906.01123/images/qualitive1/style_2.jpg",
    "https://www.stunewsnewport.com/images/editorial/jan2/CubismLRG.jpg",
    "https://d2halst20r4hcy.cloudfront.net/wallpapers2/079/340/083/456/original/file.jpg",
    "https://bsaber.com/wp-content/uploads/2020/01/49352bc094b7e66f706b5671d5da669a6266964d.jpg",
    "https://10mosttoday.com/wp-content/uploads/2014/12/Vincent_van_Gogh_self_portrait.jpg",
    "https://pic1.zhimg.com/v2-d82300aae5b1fd5f7303b057abca8a41_200x112.jpg",
    "https://1.bp.blogspot.com/-UMnqX3NZWxo/XX8I0MhluCI/AAAAAAAAADA/MoU7Mi8jp1Ejrsvf7bBLAuUNWOovdJP4QCLcBGAsYHQ/s1600/202685_0-803-5447-5447.jpg",
    "https://www.receiteria.com.br/wp-content/uploads/receitas-com-queijo-mussarela-0-1200x738.jpg",
    "https://asian-recipe.com/wp-content/uploads/2017/11/bibimbap-mixed-rice-1200x900.jpg",
]
IMAGE_PROCESSING_OPTIONS = [
    "Fake Motion",
    "Object Removal",
    "Style Transfer",
    "GAN",
    "Segmented ST",
    "Finish",
]
GAN_OPTIONS = [
    "apple2orange",
    "horse2zebra",
    "style_monet",
    "style_vangogh",
    "summer2winter",
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

    # Frame extraction
    os.makedirs(os.path.join(os.getcwd(), "payload", state["uuid"]), exist_ok=True)
    extractFrames(
        os.path.join(os.getcwd(), "payload", f"{state['uuid']}.gif"),
        os.path.join(os.getcwd(), "payload", state["uuid"]),
    )
    # Object detection
    OBJECT_REMOVAL_IMAGES=[]
    objects_gif_arguments, OBJECT_REMOVAL_OPTIONS = GIFObjectDetection(
        "./payload/" + str(state["uuid"])
    )
    for i in OBJECT_REMOVAL_OPTIONS:
        OBJECT_REMOVAL_IMAGES.append("https://assets.weforum.org/article/image/25NbfYbkiuMvTW3P_YO8QeE-SxtvNKGRX9Dgr6W-gNE.jpg")
    print("List of objects in GIF: ", OBJECT_REMOVAL_OPTIONS)
    torch.cuda.empty_cache()
    # OBJECT_REMOVAL_OPTIONS = ["test"]

    if message.get("text"):
        text = message["text"].lower()
        if state["selected_option"] == "fake motion":
            if text in map(lambda x: x.lower(), FAKE_MOTION_OPTIONS):
                bot.send_text(recipient_id, "Processing image, please wait")
                bot.send_typing_on(recipient_id)
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
                objects = [
                    objects_gif_arguments[
                        OBJECT_REMOVAL_OPTIONS.index(text)
                    ]
                ]
                bot.send_typing_on(recipient_id)
                res_url = foreground_removal(state, objects)
                state["selected_option"] = None
                if os.path.exists("./payload/FR_" + str(state["uuid"])):
                    integ_check_fst = len(
                        os.listdir("./payload/FR_" + str(state["uuid"]))
                    )
                    integ_check_orig = len(
                        os.listdir("./payload/" + str(state["uuid"]))
                    )
                    print("Send link if match: ", integ_check_fst, integ_check_orig)
                    if integ_check_fst > 0:
                        if integ_check_fst == integ_check_orig:
                            bot.send_image_url(recipient_id, res_url)
                            print("Link: ", res_url)
                            shutil.rmtree(
                                "./payload/" + str(state["uuid"]), ignore_errors=True
                            )
                            shutil.rmtree(
                                "./payload/FR_" + str(state["uuid"]), ignore_errors=True
                            )
                            bot.send_quick_reply(
                                recipient_id,
                                "Select the next process to run",
                                IMAGE_PROCESSING_OPTIONS,
                            )
                return "Continued processing"
        elif state["selected_option"] == "segmented st":
            if text in map(lambda x: x.lower(), SEGMENTED_ST_OPTIONS):
                bot.send_text(recipient_id, "Processing image, please wait")
                bot.send_typing_on(recipient_id)
                res_url = segmented_style_transfer(state, text.replace(" ", "_"))
                state["selected_option"] = None
                if os.path.exists("./payload/FST_" + str(state["uuid"])):
                    filenames_ST = []
                    for f in glob.iglob("./payload/" + str(state["uuid"]) + "/*"):
                        if str(state["uuid"]) in f:
                            if "_MASK+FST.png" in f:
                                filenames_ST.append(f)
                    filenames = []
                    for f in glob.iglob("./payload/" + str(state["uuid"]) + "/*"):
                        if str(state["uuid"]) in f:
                            if "_MASK+FST.png" not in f:
                                if "_MASK.png" not in f:
                                    if "_FST.png" not in f:
                                        filenames.append(f)
                    print(len(filenames_ST), len(filenames))
                    if len(filenames_ST) > 0:
                        if len(filenames_ST) == len(filenames):
                            bot.send_image_url(recipient_id, res_url)
                            print("Link: ", res_url)
                            shutil.rmtree(
                                "./payload/" + str(state["uuid"]), ignore_errors=True
                            )
                            shutil.rmtree(
                                "./payload/FST_" + str(state["uuid"]),
                                ignore_errors=True,
                            )
                            bot.send_quick_reply(
                                recipient_id,
                                "Select the next process to run",
                                IMAGE_PROCESSING_OPTIONS,
                            )
                return "Continued processing"
        elif state["selected_option"] == "style transfer":
            if text in map(lambda x: x.lower(), STYLE_TRANSFER_OPTIONS):
                bot.send_text(recipient_id, "Processing image, please wait")
                bot.send_typing_on(recipient_id)
                res_url = fast_style_transfer(state, text.replace(" ", "_"))
                state["selected_option"] = None
                if os.path.exists("./payload/FST_" + str(state["uuid"])):
                    integ_check_fst = len(
                        os.listdir("./payload/FST_" + str(state["uuid"]))
                    )
                    integ_check_orig = len(
                        os.listdir("./payload/" + str(state["uuid"]))
                    )
                    print("Send link if match: ", integ_check_fst, integ_check_orig)
                    if integ_check_fst > 0:
                        if integ_check_fst == integ_check_orig:
                            bot.send_image_url(recipient_id, res_url)
                            print("Link: ", res_url)
                            shutil.rmtree(
                                "./payload/" + str(state["uuid"]), ignore_errors=True
                            )
                            shutil.rmtree(
                                "./payload/FST_" + str(state["uuid"]),
                                ignore_errors=True,
                            )
                            bot.send_quick_reply(
                                recipient_id,
                                "Select the next process to run",
                                IMAGE_PROCESSING_OPTIONS,
                            )
                return "Continued processing"
        elif state["selected_option"] == "gan":
            if text in map(lambda x: x.lower(), GAN_OPTIONS):
                bot.send_text(recipient_id, "Processing image, please wait")
                res_url = cycle_gan(state, text.replace(" ", "_"))
                state["selected_option"] = None
                bot.send_image_url(recipient_id, res_url)
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
                # Programmatically determine which constants to use
                options = eval(f"{text.upper()} OPTIONS".replace(" ", "_"))
                src_images = eval(f"{text.upper()} IMAGES".replace(" ", "_"))
                bot.send_quick_reply(
                    recipient_id, "Select an option to apply", options, src_images
                )
                return "Continued processing"

    # If we are here then it means an invalid command was sent
    state["selected_option"] = None
    #bot.send_quick_reply(
    #    recipient_id, "Please send a valid command", IMAGE_PROCESSING_OPTIONS
    #)
    return "Continued processing"
