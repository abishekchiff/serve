import io

from PIL import Image
from torchvision import transforms

from ts.torch_handler.image_classifier import ImageClassifier
import base64
import torch
import logging

class MNISTDigitClassifier(ImageClassifier):
    """
    MNISTDigitClassifier handler class. This handler extends class ImageClassifier from image_classifier.py, a
    default handler. This handler takes an image and returns the number in that image.

    Here method postprocess() has been overridden while others are reused from parent class.
    """

    image_processing = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,))
    ])

    def preprocess(self, data):
        images = []

        # for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
        if isinstance(data, dict):
            image = data.get("data") or data.get("body")
        else:
            image = data
        print("Mnist image code", image)
        image = torch.FloatTensor(image)
        print("Mnist image code tensor", image)
        #images.append(image)

        return torch.Tensor(image)

    def postprocess(self, data):
        logging.info(f"mnist kf postprocess: {data}")
        return data.argmax(0).tolist()
    
    def get_insights(self, data, raw_data, target):
        print("input shape",data.shape)
        print(f"get insights unsqueezed data {torch.unsqueeze(data,0).shape}")
        return self.ig.attribute(torch.unsqueeze(data,0), target=target, n_steps=15)
