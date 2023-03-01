import os
from typing import Dict, List

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, PreTrainedTokenizer, TensorType

class TritonPythonModel:
    tokenizer: PreTrainedTokenizer

    def initialize(self, args: Dict[str, str]) -> None:
        """
        Initialize the tokenization process
        :param args: arguments from Triton config file
        """
        self.tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")

    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
        """
        Parse and tokenize each request
        :param requests: 1 or more requests received by Triton server.
        :return: text as input tensors
        """
        responses = []
        for request in requests:
            query = [
                t.decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "TEXT")
                .as_numpy()
                .tolist()
            ]
            tokens: Dict[str, np.ndarray] = self.tokenizer(
                text=query, max_length=50, padding='max_length',
                truncation=True, add_special_tokens=True, return_tensors=TensorType.NUMPY
            )
            input_ids = pb_utils.Tensor('input_ids', tokens['input_ids'])
            attention_mask = pb_utils.Tensor('attention_mask', tokens['attention_mask'])
            inference_response = pb_utils.InferenceResponse(output_tensors=[input_ids, attention_mask])
            responses.append(inference_response)

        return responses