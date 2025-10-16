from pydantic import BaseModel, Field
from typing import List
from typing_extensions import Literal

import openai
import json

client = openai.OpenAI(
    api_key="sk-67a5c9ec491c4832b45b0cbb567bc67f", # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


class ExtractionAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'],
                    "description": response_model.model_json_schema()['description'],
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'],
                        "required": response_model.model_json_schema()['required'],
                    },
                }
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None

# class Ticket(BaseModel):
#     """根据用户提供的信息查询火车时刻"""
#     date: str = Field(description="要查询的火车日期")#如果需要强制字段为必需，应使用 Field(..., required=True) 语法
#     departure: str = Field(description="出发城市或车站")
#     destination: str = Field(description="要查询的火车日期")
# result = ExtractionAgent(model_name = "qwen-plus").call("你能帮我查一下2024年1月1日从北京南站到上海的火车票吗？", Ticket)
# print(result)



# class Text(BaseModel):
#     """抽取句子中的的单词，进行文本分词"""
#     keyword: List[str] = Field(description="单词")
# result = ExtractionAgent(model_name = "qwen-plus").call('小强是小王的好朋友。谢大脚是长贵的老公。', Text)
# print(result)


# class Text(BaseModel):
#     """分析文本的情感"""
#     sentiment: Literal["正向", "反向"] = Field(description="情感类型")
#     #限制变量或字段只能取指定的字面量值，在类型检查时提供更精确的类型约束，eg. Literal["postivate", "negative"] Literal[1, 2, 3] Literal[True, False]
# result = ExtractionAgent(model_name = "qwen-plus").call('我今天很开心。', Text)
# print(result)


# class Text(BaseModel):
#     """分析文本的情感"""
#     sentiment: Literal["postivate", "negative"] = Field(description="情感类型")
# result = ExtractionAgent(model_name = "qwen-plus").call('我今天很开心。', Text)
# print(result)


# class Text(BaseModel):
#     """抽取实体"""
#     person: List[str] = Field(description="人名")
#     location: List[str] = Field(description="地名")
# result = ExtractionAgent(model_name = "qwen-plus").call('今天我和徐也也去海淀吃饭，强哥也去了。', Text)
# print(result)


# class Text(BaseModel):
#     """抽取句子中所有实体之间的关系"""
#     source_person: List[str] = Field(description="原始实体")
#     target_person: List[str] = Field(description="目标实体")
#     relationship: List[Literal["朋友", "亲人", "同事"]] = Field(description="待选关系")
# result = ExtractionAgent(model_name = "qwen-plus").call('小强是小王的好朋友。谢大脚是长贵的老公。', Text)
# print(result)


# class Text(BaseModel):
#     """抽取句子的摘要"""
#     abstract: str = Field(description="摘要结果")
# result = ExtractionAgent(model_name = "qwen-plus").call("20年来，中国探月工程从无到有、从小到大、从弱到强。党的十八大后，一个个探月工程任务连续成功，不断刷新世界月球探测史的中国纪录嫦娥三号实现我国探测器首次地外天体软着陆和巡视探测，总书记肯定“在人类攀登科技高峰征程中刷新了中国高度”；", Text)
# print(result)


class Text(BaseModel):
    """进行意图识别"""
    intent_recognition: List[Literal["OPEN", "SEARCH", "REPLAY_ALL", "NUMBER_QUERY", "DIAL", "CLOSEPRICE_QUERY", "SEND", "LAUNCH", "PLAY", "REPLY", "RISERATE_QUERY", "DOWNLOAD", "QUERY", "LOOK_BACK", "CREATE", "FORWARD", "DATE_QUERY", "SENDCONTACTS", "DEFAULT", "TRANSLATION", "VIEW", "NaN", "ROUTE", "POSITION"]] = Field(description="根据输入的语句，进行意图识别")
    domain_recognition: List[Literal["music", "app", "radio", "lottery", "stock", "novel", "weather", "match", "map", "website", "news", "message", "contacts", "translation", "tvchannel", "cinemas", "cookbook", "joke", "riddle", "telephone", "video", "train", "poetry", "flight", "epg", "health", "email", "bus", "story"]] = Field(description="根据输入的语句，进行领域识别")
    entity_recognition: List[Literal["code", "Src", "startDate_dateOrig", "film", "endLoc_city", "artistRole", "location_country", "location_area", "author", "startLoc_city", "season", "dishNamet", "media", "datetime_date", "episode", "teleOperator", "questionWord", "receiver", "ingredient", "name", "startDate_time", "startDate_date", "location_province", "endLoc_poi", "artist", "dynasty", "area", "location_poi"]]= Field(description="根据输入的语句，进行实体识别")
result = ExtractionAgent(model_name = "qwen-plus").call('把高建清的号码发给潘青华', Text)
print(result)

