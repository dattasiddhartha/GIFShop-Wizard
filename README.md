# GIFShop Wizard

<i>Collaborators: Siddhartha Datta, Jacky Lee</i>

<img src="https://github.com/dattasiddhartha/GIFShop-Wizard/blob/master/payload/DZ1BJU_FST.gif?raw=true" width=400px></img>

Submission for Facebook Hackathon 2020 (Messenger).

Implementing based on [functional requirements](https://docs.google.com/document/d/1T6mk4aypOCCCxcz2EJtfLNoait8uimbkMFGiEmzRvdg/edit). 

Current functionality:
* Receive messages from user, replies with text (non-NLP)
* Receive user GIF as payload url, and can successfully reply back the same GIF (parsing GIFs & modifying them)
* Fast Style Transfer (we store pretrained styles to be applied to user input images)
* Segmented Style Transfer (we apply instance segmentation to images and apply FST to those segments only)

Next steps:
* Image editing functions
* Quick replies

## Usage

Make sure you first update your credentials in the `credentials.py` file.

Run the chatbot with the following command.
```
python serve.py
ngrok http 8000
```

Remember to update the [Messenger Webhook](https://developers.facebook.com/apps/309273736750794/messenger/settings/) when running the `ngrok` service.

Warnings:
* When appending shared docs on public files, be careful about the information they or their history contain
* When uploading repo files, be careful with tokens / secrets [might be best to create a separate repo later]
