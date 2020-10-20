# Copyright 2019 kubeflow.org.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import kfserving
from typing import List, Dict
from PIL import Image
import torchvision.transforms as transforms
import logging
logging.basicConfig(level=kfserving.constants.KFSERVING_LOGLEVEL)
import io
import numpy as np
import base64
EXPLAINER_URL_FORMAT = "http://{0}/v1/models/{1}:explain"
import json
import tornado

image_processing = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def image_transform(instance):
    byte_array = base64.b64decode(instance["data"])
    image = Image.open(io.BytesIO(byte_array))
    a = np.asarray(image)
    im = Image.fromarray(a)
    
    instance["data"] = image_processing(im).tolist()
    logging.info(instance)
    return instance


class ImageTransformer(kfserving.KFModel):
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host
        self.explainer_host = predictor_host
        logging.info("MODEL NAME %s", name)
        logging.info("PREDICTOR URL %s", self.predictor_host)
        logging.info("EXPLAINER URL %s", self.explainer_host)
        self.timeout = 4000

    def preprocess(self, inputs: Dict) -> Dict:
        return {'instances': [image_transform(instance) for instance in inputs['instances']]}

    def postprocess(self, inputs: List) -> List:
        return inputs

    async def explain(self, request: Dict) -> Dict:
        if self.explainer_host is None:
            raise NotImplementedError
        logging.info(f"Inside Image Transformer explain {EXPLAINER_URL_FORMAT.format(self.explainer_host, self.name)}")
        response = await self._http_client.fetch(
            EXPLAINER_URL_FORMAT.format(self.explainer_host, self.name),
            method='POST',
            request_timeout=self.timeout,
            body=json.dumps(request)
        )
        if response.code != 200:
            raise tornado.web.HTTPError(
                status_code=response.code,
                reason=response.body)
        return json.loads(response.body)
