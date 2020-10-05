# pylint: disable=E1102
# TODO remove pylint disable comment after https://github.com/pytorch/pytorch/issues/24807 gets merged.
"""
Module for text classification default handler
DOES NOT SUPPORT BATCH!
"""
import torch
import torch.nn.functional as F
from torchtext.data.utils import ngrams_iterator
from .text_handler import TextHandler
from ..utils.util  import map_class_to_label

from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization

class TextClassifier(TextHandler):
    """
    TextClassifier handler class. This handler takes a text (string) and
    as input and returns the classification text based on the model vocabulary.
    """

    ngrams = 2
    
    def preprocess(self, data):
        """
        Normalizes the input text for PyTorch model using following basic cleanup operations :
            - remove html tags
            - lowercase all text
            - expand contractions [like I'd -> I would, don't -> do not]
            - remove accented characters
            - remove punctuations
        Converts the normalized text to tensor using the source_vocab.
        Returns a Tensor
        """

        
        # Compat layer: normally the envelope should just return the data
        # directly, but older versions of Torchserve didn't have envelope.
        print("Using KFServing text classifier")
        #Processing only the first input, not handling batch inference
        text = None
        inp = data
        if isinstance(inp, dict):
            name = inp.get("name")
            if name == "context":
                text = inp.get("data")
                print("Inside KFServing preprocess, ",text)
            target = inp.get("target")
                
        else:
            text = inp
        #text = text.decode('utf-8')
        print("text classifier : text type",type(text))
        text = self._remove_html_tags(text)
        text = text.lower()
        text = self._expand_contractions(text)
        text = self._remove_accented_characters(text)
        text = self._remove_punctuation(text)
        text = self._tokenize(text)
        self.input_text = text
        text = torch.as_tensor(
            [
                self.source_vocab[token]
                for token in ngrams_iterator(text, self.ngrams)
            ],
            device=self.device
        )
        return text

    def inference(self, data, *args, **kwargs):
        print("inference data",data)
        offsets = None
        #offsets = torch.as_tensor([0], device=self.device)
        return super().inference(data, offsets)

    def postprocess(self, data):
        print("inference shape",data.shape)
        data = F.softmax(data)
        data = data.tolist()
        print("postprocess data :", data)
        return  map_class_to_label([data], self.mapping)[0]


    def get_insights(self, text_tensor, text, target):
        token_reference = TokenReferenceBase()
        print("input_text shape", len(text_tensor.shape))
        print("get_insights target",target)
        offsets = torch.tensor([0])

        # text_tensor = torch.as_tensor(
        #     [self.source_vocab[tok]
        #     for tok in self.input_text_tensor
        #     ],
        #     device=self.device
        # )
        print("text_tensor tokenized shape", text_tensor.shape)
        reference_indices = token_reference.generate_reference(text_tensor.shape[0], device=self.device).squeeze(0)
        print("reference indices ", reference_indices.shape, reference_indices)
        

        all_tokens = self.get_word_token(text)
        attributions = self.lig.attribute(text_tensor, reference_indices, \
                                             additional_forward_args=(offsets), return_convergence_delta=False,target=target)
        
        print("attributions shape",attributions.shape)
        attributions_sum = self.summarize_attributions(attributions)
        response = {}
        
        response["importances"] = attributions_sum.tolist()
        response["words"] = all_tokens
        return response

    
