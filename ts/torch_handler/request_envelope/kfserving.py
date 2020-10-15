import json
from itertools import chain
from base64 import b64decode
import logging
import torch

from .base import BaseEnvelope
logger = logging.getLogger(__name__)

class KFservingEnvelope(BaseEnvelope):
    """
    Implementation. Captures batches in JSON format, returns
    also in JSON format.
    """
    def __init__(self, handle_fn):
        super(KFservingEnvelope, self).__init__(handle_fn)
        self.dtype_map = {"torch.int8" : "INT8",  "torch.int16" : "INT16", "torch.int32" : "INT32", 
                    "torch.int64" : "INT64", "torch.uint8" : "UINT8", "torch.float16" : "FP16", "torch.float32" : "FP32",
                    "torch.float64" : "FP64", "torch.bool" : "BOOL"}
        self._inputs = []
        self._outputs = []
        self.input_id = None
        self.parameters = None

    def parse_input(self, data):
        print("Parsing input in KFServing.py")
        self._data_list = [row.get("data") or row.get("body") for row in data]
        
        print("Parse input data_list ",self._data_list)
        """selecting the first input from the list torchserve creates as kfserving sends batches
        inside inputs label """
        data = self._data_list[0]
        inp_dict = {}
        #IF the KF Transformer and Explainer sends in data as bytesarray 
        if isinstance(data, (bytes, bytearray)):
            
            data = data.decode()
            data = json.loads(data)
            print("Bytes array is ",data)
        print("KFServing parsed inputs", data)
        
        #kf's v1_beta1 spec request data

        if isinstance(data, dict):
            if "id" in data:
                self.input_id = data["id"]
            if "parameters" in data:
                self.parameters = data["parameters"]
            if "outputs" in data:
                self._outputs = data["outputs"]

            self._inputs = data["inputs"]
        
            for req_inp in self._inputs:
                inp_dict[str(req_inp["name"])] = req_inp["data"] 

            if not "data" in inp_dict:
                raise AttributeError("The request input does not contain data")
            

        return inp_dict

    def format_output(self, results):

        # if "explanations" in output:
        #     response["explanations"] =  output["explanations"]
  
        # self._outputs = [data_2.get("outputs") for data_2 in self._data_list]
        #Processing only the first output, as we are not handling batch inference
        # self._outputs = self._outputs[0]

        outputs_list = []
        response = {}
        print("The self._outputs  in format output", self._outputs)
        if self._outputs: 
            #Processing only the first output, as we are not handling batch inference
            results = results[0]
            print("The results received in format output", results)
            if isinstance(results, dict):
                results = results.get("predictions")
                for output in self._outputs:
                    output_dict = {}
                    result = results[0]
                    if isinstance(output, dict):
                        #processing only the first response in the batch
                        
                        print("The output dict in format output", output)
                        print("The results received in format output", result)
                        
                        if output["name"] in result.keys():
                            output_dict["name"] = output["name"]
                           
                            data = result[output["name"]]
                            data_tensor = data
                            if not isinstance(data, torch.Tensor):
                                data_tensor = torch.tensor([data])
                            output_dict["data"] = data
                            output_dict["shape"] = data_tensor.shape #static shape should be replaced with result shape

                            output_dict["datatype"] = self.dtype_map[str(data_tensor.dtype)] #Static types should be replaced with types based on result
                            outputs_list.append(output_dict)
                        else :
                            logger.info("The request key %s is not present in the prediction", output['name'])
                response["outputs"] = outputs_list

            else :
                logger.info("The prediction response does not contain labels")
                response["outputs"] = results
        else :
            response["outputs"] = results
        
        logger.info("The Response of KFServing %s", response)
        return [response]