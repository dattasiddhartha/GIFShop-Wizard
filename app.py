import random, os, glob, imageio, pathlib, string
from flask import Flask, request, send_from_directory, send_file
from chat.bot import Bot
import credentials as cred
import urllib.request
from PIL import Image
import numpy as np
from faststyletransfer_train import FastStyleTransfer
from faststyletransfer_eval import FasterStyleTransfer

app = Flask(__name__)       # Initializing our Flask application
ACCESS_TOKEN = cred.ACCESS_TOKEN
VERIFY_TOKEN = cred.VERIFY_TOKEN
bot = Bot(ACCESS_TOKEN)

# Importing standard route and two requst types: GET and POST.
# We will receive messages that Facebook sends our bot at this endpoint
@app.route('/', methods=['GET', 'POST'])
def receive_message():
    if request.method == 'GET':
        # Before allowing people to message your bot Facebook has implemented a verify token
        # that confirms all requests that your bot receives came from Facebook.
        token_sent = request.args.get("hub.verify_token")
        return verify_fb_token(token_sent)
    # If the request was not GET, it  must be POSTand we can just proceed with sending a message
    # back to user
    else:
            # get whatever message a user sent the bot
        output = request.get_json()
        for event in output['entry']:
            messaging = event['messaging']
            for message in messaging:
                if message.get('message'):
                    # Facebook Messenger ID for user so we know where to send response back to
                    recipient_id = message['sender']['id']
                    if message['message'].get('text'):
                        #response_sent_text = get_message()
                        response_sent_text = message['message'].get('text')
                        send_message(recipient_id, response_sent_text)
                    # if user send us a GIF, photo, video or any other non-text item
                    if message['message'].get('attachments'):
                        response_sent_text = get_message()
                        response_sent_text = message['message'].get('attachments')
                        url_of_attachment = response_sent_text[0]['payload']['url']
                        #send_message(recipient_id, response_sent_text)
                        #send_QuickReplies(recipient_id, response_sent_text)

                        temp_filename = id_generator()
                        #send_ImageBackToUser(recipient_id, url_of_attachment)
                        try:
                            send_FastStyleTransfer(temp_filename, recipient_id, url_of_attachment)
                        except:
                            send_message(recipient_id, "cleared")

                        # clear up backlog of responses during development
                        #send_message(recipient_id, "cleared")

    return "Message Processed"

@app.route('/file/<string:path>', methods=['GET'])
def get_file(path=''):
    return send_file("./payload/"+path)


def verify_fb_token(token_sent):
    # take token sent by Facebook and verify it matches the verify token you sent
    # if they match, allow the request, else return an error
    if token_sent == VERIFY_TOKEN:
        return request.args.get("hub.challenge")
    return 'Invalid verification token'


def get_message():
    sample_responses = ["You are stunning!", "We're proud of you",
                        "Keep on being you!", "We're greatful to know you :)"]
    # return selected item to the user
    return random.choice(sample_responses)


# Send response to the user
def send_message(recipient_id, response):
    # sends user the text message provided via input response parameter
    bot.send_text_message(recipient_id, response)
    return "success"

def send_QuickReplies(recipient_id, response):
    #bot.send_attachment_url(recipient_id, "image", 'https://images.unsplash.com/photo-1587251702548-f39b762ec623?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=500&q=60')
    quick_reply_message = "What's your favorite House in Game of Thrones?"
    quick_rep_option = (["Stark","stark_payload"],["Lannister","lan_payload"],["Targaryan","tar_payload"],["none","None"])
    bot.send_quickreply(recipient_id,quick_reply_message,quick_rep_option)
    print("sent")
    return "success"

def send_ImageBackToUser(recipient_id, url_of_attachment):
    # parse the GIF as individual images > Reverse the order of the images > Restitch into GIF > Load it as a URL to be sent (?)
    temp_filename = id_generator()
    urllib.request.urlretrieve(url_of_attachment, './payload/'+str(temp_filename)+'.gif')
    os.mkdir('./payload/'+str(temp_filename))
    extractFrames('./payload/'+str(temp_filename)+'.gif', './payload/'+str(temp_filename))
    stitchImages('./payload/'+str(temp_filename), str(temp_filename))

    print("Img url: ", str(cred.ngrok_link+"file/"+str(temp_filename)+"_new.gif"))
    bot.send_attachment_url(recipient_id, "image", str(cred.ngrok_link+"file/"+str(temp_filename)+"_new.gif"))

    return "success"

def extractFrames(inGif, outFolder):
    frame = Image.open(inGif)
    nframes = 0
    while frame:
        frame.save( '%s/%s-%s.png' % (outFolder, os.path.basename(inGif), nframes ) , 'GIF')
        nframes += 1
        try:
            frame.seek( nframes )
        except EOFError:
            break;
    return True

def stitchImages(dir, filename):
    # Load frames
    filenames = []
    for f in glob.iglob(dir+"/*"):
        filenames.append(f)

    ## Save into a GIF file that loops forever
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(str(dir+'_new.gif').replace("\\","/"), images)


# with custom Training has some issues
def send_FastStyleTransfer(temp_filename, recipient_id, url_of_attachment):
    # parse the GIF as individual images > Reverse the order of the images > Restitch into GIF > Load it as a URL to be sent (?)
    #temp_filename = id_generator()
    urllib.request.urlretrieve(url_of_attachment, './payload/'+str(temp_filename)+'.gif')
    os.mkdir('./payload/'+str(temp_filename))
    extractFrames('./payload/'+str(temp_filename)+'.gif', './payload/'+str(temp_filename))

    os.mkdir('./payload/FST_'+str(temp_filename))

    # Client side - Issue is TORCH.DATALOADER, keeps refreshing flask
    #stitch_FST_Images_ClientStyle('./payload/'+str(temp_filename), './payload/FST_'+str(temp_filename), str(temp_filename))

    # Server side style images - pretrained models
    stitch_FST_Images_ServerStyle('./payload/'+str(temp_filename), './payload/FST_'+str(temp_filename), str(temp_filename))

    print("Img url: ", str(cred.ngrok_link+"file/"+str(temp_filename)+"_FST.gif"))
    bot.send_attachment_url(recipient_id, "image", str(cred.ngrok_link+"/file/"+str(temp_filename)+"_FST.gif"))

    return "success"

def stitch_FST_Images_ServerStyle(orig_dir, styled_dir, unique_filename):
    # Load frames
    filenames = []
    for f in glob.iglob(orig_dir+"/*"):
        filenames.append(f)

    # apply style transfer to each frame
    counter=0
    for filename in filenames:
        FasterStyleTransfer("./fast_neural_style_transfer/models/mosaic_style__200_iter__vgg19_weights.pth", filename, styled_dir+"/"+str(unique_filename)+"_FST_"+str(counter)+"_.png")
        counter+=1

    ## Save into a GIF file that loops forever
    filenames_ST = []
    for f in glob.iglob(styled_dir+"/*"):
        filenames_ST.append(f)
    images = []
    for filename in filenames_ST:
        images.append(imageio.imread(filename))
    imageio.mimsave(str(orig_dir+'_FST.gif').replace("\\","/"), images)

def stitch_FST_Images_ClientStyle(orig_dir, styled_dir, unique_filename):
    # Load frames
    filenames = []
    for f in glob.iglob(orig_dir+"/*"):
        filenames.append(f)

    # apply style transfer to each frame
    counter=0
    for filename in filenames:
        FastStyleTransfer("unique_id", 50, filename, "./fast_neural_style_transfer/style_images/mosaic.jpg", styled_dir+"/"+str(unique_filename)+"_FST.png")
        counter+=1

    ## Save into a GIF file that loops forever
    filenames_ST = []
    for f in glob.iglob(styled_dir+"/*"):
        filenames_ST.append(f)
    images = []
    for filename in filenames_ST:
        images.append(imageio.imread(filename))
    imageio.mimsave(str(dir+'_FST.gif').replace("\\","/"), images)



def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
   return ''.join(random.choice(chars) for _ in range(size))


# Add description here about this if statement.
if __name__ == "__main__":
    app.run()
