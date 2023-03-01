import transformers
from transformers.onnx import FeaturesManager
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import torch

model_id = "finiteautomata/bertweet-base-sentiment-analysis"

feature = "sequence-classification"
model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
# onnx_config = model_onnx_config(model.config)

# # export
# onnx_inputs, onnx_outputs = transformers.onnx.export(
#         preprocessor=tokenizer,
#         model=model,
#         config=onnx_config,
#         opset=13,
#         output=Path("model.onnx"))

text = "this film is so good"
inputs = tokenizer.encode_plus(text,
                               max_length=64,
                               padding='max_length',
                               truncation=True,
                               add_special_tokens=True,
                               return_tensors='pt')

input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

torch.onnx.export(model,
                  (input_ids, attention_mask),
                  "model/1/model.onnx",
                  verbose=False,
                  do_constant_folding=True,
                  input_names=['input_ids', 'attention_mask'],
                  output_names=['output'],
                  dynamic_axes={
                        "input_ids": {0: "batch_size", 1: "sequence"},
                        "attention_mask": {0: "batch_size", 1: "sequence"},
                        "output": {0: "batch_size"},
                  })
