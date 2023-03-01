import struct
import json
from locust import HttpUser, task, between
import random
import tritonclient.http as http
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import logging



logger = logging.getLogger('logs')
logger.setLevel(logging.INFO)
fh = logging.FileHandler('logs.log')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

maxlen = 50
tokenizer = AutoTokenizer.from_pretrained('finiteautomata/bertweet-base-sentiment-analysis')
class_names = {
    "0": "Negative",
    "1": "Neutral",
    "2": "Positive"
}

batch_size = 1024
logger.info(batch_size)

def _get_inference_request(inputs, request_id, outputs, sequence_id,
                           sequence_start, sequence_end, priority, timeout):
    infer_request = {}
    parameters = {}
    if request_id != "":
        infer_request['id'] = request_id
    if sequence_id != 0:
        parameters['sequence_id'] = sequence_id
        parameters['sequence_start'] = sequence_start
        parameters['sequence_end'] = sequence_end
    if priority != 0:
        parameters['priority'] = priority
    if timeout is not None:
        parameters['timeout'] = timeout

    infer_request['inputs'] = [
        this_input._get_tensor() for this_input in inputs
    ]
    if outputs:
        infer_request['outputs'] = [
            this_output._get_tensor() for this_output in outputs
        ]
    else:
        # no outputs specified so set 'binary_data_output' True in the
        # request so that all outputs are returned in binary format
        parameters['binary_data_output'] = True

    if parameters:
        infer_request['parameters'] = parameters

    request_body = json.dumps(infer_request)
    json_size = len(request_body)
    binary_data = None
    for input_tensor in inputs:
        raw_data = input_tensor._get_binary_data()
        if raw_data is not None:
            if binary_data is not None:
                binary_data += raw_data
            else:
                binary_data = raw_data

    if binary_data is not None:
        request_body = struct.pack(
            '{}s{}s'.format(len(request_body), len(binary_data)),
            request_body.encode(), binary_data)
        return request_body, json_size

    return request_body, None


class PerformanceTest(HttpUser):
    wait_time = between(1, 3)

    @task(1)
    def testApi(self):
        inputs = ["i love you to the moon and back",
                  "i hate you",
                  "you are bad guy but i still love you"]
        input_text = random.choice(inputs)
        input = http.InferInput('TEXT', (1,), 'BYTES')
        input.set_data_from_numpy(np.asarray([input_text], dtype=object))
        output = http.InferRequestedOutput('output', binary_data=False)

        request_body, json_size = _get_inference_request(inputs=[input], request_id="", outputs=[output],
                       sequence_id=0, sequence_start=False, sequence_end=False,
                       priority=0, timeout=None)
        
        headers = None
        if json_size is not None:
            if headers is None:
                headers = {}
            headers["Inference-Header-Content-Length"] = str(json_size)
        
        response = self.client.post("/v2/models/ensemble/infer", data=request_body, headers=headers)
        outputs = response.json()['outputs'][0]['data']
        logits = np.asarray(outputs, dtype=np.float32)
        probabilities = softmax(logits)
        predictions = np.argmax(probabilities)
        predictions = [class_names[str(predictions)]]
        logging.info(predictions)


