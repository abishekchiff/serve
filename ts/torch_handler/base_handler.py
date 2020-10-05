"""
Base default handler to load torchscript or eager mode [state_dict] models
Also, provides handle method per torch serve custom model specification
"""
import abc
import logging
import os
import importlib.util
import torch
from ..utils.util import list_classes_from_module, load_label_mapping

logger = logging.getLogger(__name__)


class BaseHandler(abc.ABC):
    """
    Base default handler to load torchscript or eager mode [state_dict] models
    Also, provides handle method per torch serve custom model specification
    """
    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False
        self.context = None
        self.manifest = None
        self.map_location = None
        self.explain = False
        self.batch_size = 1

    def initialize(self, context):
        """First try to load torchscript else load eager mode state_dict based model"""

        properties = context.system_properties
        self.map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.map_location + ":" + str(properties.get("gpu_id"))
                                   if torch.cuda.is_available() else self.map_location)
        self.manifest = context.manifest

        model_dir = properties.get("model_dir")
        self.batch_size = properties.get("batch_size")
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)

        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        # model def file
        model_file = self.manifest['model'].get('modelFile', '')

        if model_file:
            logger.debug('Loading eager model')
            self.model = self._load_pickled_model(model_dir, model_file, model_pt_path)
        else:
            logger.debug('Loading torchscript model')
            self.model = self._load_torchscript_model(model_pt_path)

        self.model.to(self.device)
        self.model.eval()

        logger.debug('Model file %s loaded successfully', model_pt_path)

        # Load class mapping for classifiers
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        self.mapping = load_label_mapping(mapping_file_path)

        self.initialized = True

    def _load_torchscript_model(self, model_pt_path):
        return torch.jit.load(model_pt_path, map_location=self.map_location)

    def _load_pickled_model(self, model_dir, model_file, model_pt_path):
        model_def_path = os.path.join(model_dir, model_file)
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model.py file")

        module = importlib.import_module(model_file.split(".")[0])
        model_class_definitions = list_classes_from_module(module)
        if len(model_class_definitions) != 1:
            raise ValueError("Expected only one class as model definition. {}".format(
                model_class_definitions))

        model_class = model_class_definitions[0]
        state_dict = torch.load(model_pt_path, map_location=self.map_location)
        model = model_class()
        model.load_state_dict(state_dict)
        return model



    def preprocess(self, data):
        """
        Override to customize the pre-processing
        :param data: Python list of data items
        :return: input tensor on a device
        """
        return torch.as_tensor(data, device=self.device)

    def inference(self, data, *args, **kwargs):
        """
        Override to customize the inference
        :param data: Torch tensor, matching the model input shape
        :return: Prediction output as Torch tensor
        """
        marshalled_data = data.to(self.device)
        with torch.no_grad():
            results = self.model(marshalled_data, *args, **kwargs)
        return results

    def postprocess(self, data):
        """
        Override to customize the post-processing
        :param data: Torch tensor, containing prediction output from the model
        :return: Python list
        """

        return data.tolist()
    # def handle(self, data, context, explain = False):
    #     """
    #     Entry point for default handler
    #     """

    #     # It can be used for pre or post processing if needed as additional request
    #     # information is available in context
    #     self.context = context
    #     output_explain  = None
    #     #explain from header 

    #     data = self.preprocess(data)
    #     output = self.inference(data)
    #     output_explain = self.explain_handle(context, data)
    #     output = self.postprocess(output)
    #     return output, output_explain

    # def handle(self, data, context):
    #     """
    #     Entry point for default handler
    #     """
    #     self.context = context

    #     data_list = process_batches(data,self.preprocess)

    #     data_list = self.inference(data_list)
        
    #     data_list = process_batches(data,self.postprocess)
        
    #     return data_list

    def handle(self, data, context):
        """
        Entry point for default handler
        """

        # It can be used for pre or post processing if needed as additional request
        # information is available in context
        self.context = context
        self.initialize(self.context)
  
        #preproces
        print("Base handler data length :", len(data))
        data_list = self.process_batches(data,self.preprocess)
        #inference
        data_list_tensor = torch.stack(data_list,0)
        inf_list = self.inference(data_list_tensor)
        print("base handler inference output",inf_list) 
        #postprocess
        output_list = self.process_batches(inf_list,self.postprocess)
       
        #explain if explain is set true in context
        output_explain_list  = self.explain_handle(data_list, data)
       
        #Out postprocess
        output = {}
        output["predictions"] = output_list
        output["explanations"] = output_explain_list
        #output_explain = process_batches(data_list, self.explain_handle(context, data)
        return [output]
    
    def explain_handle(self, data_list, raw_data):
        
        output_explain_list  = None
        if  self.context and self.context.get_request_header(0,"explain"):
            if self.context.get_request_header(0,"explain") == "True":
                self.explain = True
                print("IsExplain",self.explain)
                print("The explainations are being calculated", data_list)

                targets = []
                inputs = []
                for r_d in raw_data:
                    if isinstance(r_d, dict):
                        targets.append( r_d.get("target"))
                        inputs.append(r_d.get("data"))

                print("The targets in explain_handler", targets)
                output_explain_list = self.process_batches((data_list, inputs, targets), self.get_insights, explain = True)
            
        return output_explain_list

    def process_batches(self, data, func, explain = False):
        
        data_list = []
        print("process batches :", data)
        if self.batch_size == 1:
            if not explain:
                func_output = func(data[0])
            elif explain:
                data_list, data, targets = data
                func_output = func(data_list[0], data[0], targets[0])     
            data_list.append(func_output)


        elif isinstance(data, (list,torch.Tensor)):
            print(f"Process_batches get executed for {len(data)} times")
            for d in data:
                func_output = func(d)
                data_list.append(func_output)
        
        elif explain:
            #dat, targets = data
            for tensor_data, datas, targets  in zip(*data):
                func_output = func(tensor_data, datas, targets)
                data_list.append(func_output)

        print("process batches output :", data_list)
        return data_list

# def is_explain(context):
#     if  context and context.get_request_header(0,"explain"):
#         if context.get_request_header(0,"explain") == "True":
#                 return True
#     return False

# def handle(self, data, context):
#     """
#     Entry point for default handler
#     """
#     self.context = context

#     data_list = process_batches(data,self.preprocess)

#     data_list = self.inference(data_list)
    
#     data_list = process_batches(data,self.postprocess)
    
#     return data_list
  
                
  

