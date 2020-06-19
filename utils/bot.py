import json
import requests


class Bot:
    def __init__(self, access_token, api_version=7.0):
        self.access_token = access_token
        self.api_version = api_version
        self.graph_url = f"https://graph.facebook.com/v{api_version}"
        self.messenger_url = f"{self.graph_url}/me/messages?access_token={access_token}"

    def send_request(self, data, headers={"content-type": "application/json"}):
        """
        Sends request to Messenger API and returns whether the operation was
        successful
        """
        response = requests.post(
            self.messenger_url,
            params={"access_token": self.access_token},
            data=json.dumps(data),
            headers={"content-type": "application/json"},
        )
        if response.status_code is not 200:
            print(f"[ERROR] {response.status_code} Error")
        return response.status_code is 200

    def send_text(self, recipient_id, text):
        """
        Sends text to Messenger API and returns whether the operation was
        successful
        """
        data = {"recipient": {"id": recipient_id}, "message": {"text": text}}
        return send_request(data)

    def send_quick_reply(self, recipient_id, text, quick_replies):
        """
        Sends quick reply to Messenger API and returns whether the operation
        was successful
        """
        quick_replies_arr = map(
            lambda quick_reply: {
                "content_type": "text",
                "title": quick_reply,
                "payload": quick_reply.lower(),
            },
            quick_replies,
        )
        data = {
            "recipient": {"id": recipient_id},
            "message": {"text": text, "quick_replies": quick_replies_arr},
        }
        return send_request(data)

    def send_image_url(self, recipient_id, image_url):
        """
        Sends image URL to Messenger API and returns whether the operation was
        successful
        """
        data = {
            "recipient": {"id": recipient_id},
            "message": {"attachment": {"type": "image", "payload": {"url": image_url}}},
        }
        return send_request(data)

    # TODO: make this work
    def TODO_send_attachment(self, recipient_id, filename):
        data = json.dumps(
            {"recipient": {"id": recipient_id}, "message": {"text": text},}
        )
        multipart_data = MultipartEncoder(payload)
        multipart_header = {"Content-Type": multipart_data.content_type}
        return requests.post(
            self.messenger_url,
            params={"access_token": self.access_token},
            data=data,
            headers={"Content-Type": "application/json"},
        )