# python自带库
import time
import traceback
from typing import Union

# 第三方库
import torch
from fastapi import FastAPI

# 自己写的模块
from data_schema import TextClassifyResponse
from data_schema import TextClassifyRequest
from logger import logger

from transformers import BertTokenizer, BertForSequenceClassification

app = FastAPI(title="外卖评价BERT服务")

# 设备和模型加载
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('./assets/models/bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('./assets/weights/bert_waimai.pt', num_labels=2)
model.to(device)
model.eval()

def model_for_bert(text: str) -> str:
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).item()
    return str(pred)

@app.post("/v1/text-cls/bert")
def bert_classify(req: TextClassifyRequest) -> TextClassifyResponse:
    start_time = time.time()
    response = TextClassifyResponse(
        request_id=req.request_id,
        request_text=req.request_text,
        classify_result="",
        classify_time=0,
        error_msg=""
    )

    logger.info(f"{req.request_id} {req.request_text}")
    try:
        response.classify_result = model_for_bert(req.request_text)
        response.error_msg = "ok"
    except Exception as err:
        response.classify_result = ""
        response.error_msg = traceback.format_exc()

    response.classify_time = round(time.time() - start_time, 3)
    return response
