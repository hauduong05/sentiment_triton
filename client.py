import tritonclient.http as http
import numpy as np
import random
import requests
from scipy.special import softmax
from transformers import AutoTokenizer



model_name = 'ensemble'
url = '0.0.0.0:8000'
maxlen = 64
input_texts = ["i love you to the moon and back",
                  "i hate you",
                  "you are bad guy but i still love you"]
class_names = {
    "0": "Negative",
    "1": "Neutral",
    "2": "Positive"
}
input_text = random.choice(input_texts)

triton_client = http.InferenceServerClient(url=url, verbose=False)
input0 = http.InferInput('TEXT', (1,), 'BYTES')
input0.set_data_from_numpy(np.asarray([input_text], dtype=object))


output = http.InferRequestedOutput('output', binary_data=False)
response = triton_client.infer(model_name=model_name, model_version='1', inputs=[input0], outputs=[output])
logits = response.as_numpy('output')
logits = np.asarray(logits, dtype=np.float32)
probabilities = softmax(logits)
predictions = np.argmax(probabilities, axis=1)
predictions = [class_names[str(label)] for label in predictions]
print(predictions)
