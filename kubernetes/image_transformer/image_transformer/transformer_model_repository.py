import os

import tornado.ioloop
import tornado.web
import tornado.httpserver
import tornado.log

from kfserving.kfmodel_repository import KFModelRepository
import requests
import kfserving
import logging
logging.basicConfig(level=kfserving.constants.KFSERVING_LOGLEVEL)
from image_transformer import ImageTransformer

LOAD_URL_FORMAT = "http://{0}/v1/models/{1}/load"
UNLOAD_URL_FORMAT = "http://{0}/v1/models/{1}/unload"

class TransformerModelRepository(KFModelRepository):

    def __init__(self, predictor_host:str):
        super().__init__()
        logging.info("ImageTSModelRepo is initialized")
        self.predictor_host = predictor_host

    def load(self, name: str,) -> bool:
        logging.info(f"ImageTSModelRepository : loading model {name}")
        img_transformer = ImageTransformer(name, self.predictor_host)
        if not self.predictor_host:
            raise NotImplementedError
        self.name = name
        logging.info(f"Transformer loading model {self.name}")

        response = requests.post(	        
            LOAD_URL_FORMAT.format(self.predictor_host, self.name))	  
        if response.status_code != 200:	        
            raise tornado.web.HTTPError(status_code=response.status_code,reason=response.content)	 
        else :
            self.update(img_transformer)
            img_transformer.ready = True

        return self.is_model_ready(img_transformer)    
    
    def unload(self, name: str):
        logging.info(f"ImageTSModelRepository : unloading model {name}")
        if not self.predictor_host:
            raise NotImplementedError
        self.name = name
        logging.info(f"Transformer unloading model {self.name}")

        response = requests.post(	        
            UNLOAD_URL_FORMAT.format(self.predictor_host, self.name))	  
        if response.status_code != 200:	        
            raise tornado.web.HTTPError(status_code=response.status_code,reason=response.content)	 
        else :
            del self.models[name]
          
        