import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import os
import matplotlib.pyplot as plt
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import warnings

warnings.filterwarnings('ignore')



PROMTPT = """
你是一个电影推荐专家。
请根据用户的历史电影观看记录：{movie_seq},
从电影库{movie_list}中推荐5部电影，并给出推荐打分（1~5分，数值越大，推荐度越高）和推荐理由，输出为json格式，如下：
'''
{{
    "title": "电影名称",
    "score": "推荐打分", 
    "reason": "推荐理由"
}},
...}}
"""
#读取电影数据
movies = pd.read_csv(os.path.join('./M_ML-100K', 'movies.dat'),sep="::", header=None, engine='python',encoding='latin1')
movies.columns = ["movie_id", "movie_name", "movie_genres"]
#读取评分数据
ratings = pd.read_csv(os.path.join('./M_ML-100K', 'ratings.dat'),sep="::", header=None, engine='python',encoding='latin1')
ratings.columns = ["user_id", "movie_id", "rating_score", "timestamp"]
#读取movie_review数据
movie_reviews = pd.read_excel('./M_ML-100K/text.xls', engine='xlrd')
movie_reviews = movie_reviews.rename(columns={'movie-id': 'movie_id'})
print("movie_reviews.head()\n",movie_reviews.head())

ratings = ratings.sort_values(by=["user_id", "timestamp"], ascending=[True, True])
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s', utc=True)
print('ratings.head()_sortedby_user_id\n',ratings.head(100))
# 合并表格
movie_rating_merged = pd.merge(ratings, movies, on='movie_id', how='left')
movie_rating_review_merged = pd.merge(movie_rating_merged, movie_reviews, on='movie_id', how='left')
# new_cols = ['user_id', 'movie_id', 'movie_name', 'movie_genres', 'rating_score', 'timestamp']
new_cols = ['user_id', 'movie_name', 'movie_genres', 'review', 'rating_score','timestamp']
user_movies= movie_rating_review_merged[new_cols]


# 2. 定义转换函数：将 DataFrame 的每一行转换为字典
def row_to_dict(row):
    return {
        "电影名字": row['movie_name'],
        "电影风格": row['movie_genres'],
        "电影点评": row['review'],
        "用户评分": str(row['rating_score']), # 将评分转为字符串
        "观看时间": str(row['timestamp']) # 将时间戳转为字符串
    }
user_movie_dict_list = user_movies.groupby('user_id').apply(
    lambda x: [row_to_dict(row) for _, row in x.iterrows()]
)

# 获取所有电影列表
# 获取所有电影列表（去重）
all_user_movie_tuples = [
    (movie['电影名字'], movie['电影风格'], movie['电影点评'])
    for user_history in user_movie_dict_list
    for movie in user_history
]

# 去重
unique_movie_tuples = list(set(all_user_movie_tuples))

# 转回字典格式
all_movie_list = [
    {
        "电影名字": movie[0],
        "电影风格": movie[1],
        "电影点评": movie[2]
    }
    for movie in unique_movie_tuples
]

# 打印结果
print(f"所有电影记录总数: {len(all_user_movie_tuples)}")
print(f"不重复的电影总数: {len(all_movie_list)}")
print(f"电影表示样例：")
print(json.dumps(all_movie_list[:5], ensure_ascii=False, indent=2))

# 测试：获取第一个用户的数据并打印
test_user_id = user_movie_dict_list.index[0]
test_user_history = user_movie_dict_list[test_user_id]
# 计算该用户看过的电影数量
movie_count = len(test_user_history)
print(f"用户 {test_user_id} 看过的电影个数为: {movie_count}")
# 1. 提取该用户所有记录的时间戳
timestamps = [item['观看时间'] for item in test_user_history]

# 2. 计算最大值和最小值
# 注意：如果 timestamps 是字符串格式（如 "1997-12-04..."），需要先转换回 datetime 对象或时间戳数字
# 假设它们是 Pandas 的 Timestamp 对象或 datetime 对象：
if isinstance(timestamps[0], str):
    import datetime
    # 如果是字符串，尝试解析（这里假设格式是标准的，如果是 Pandas Timestamp 也可以直接转）
    # 简单处理：如果之前没转成 datetime，这里用 pd.to_datetime 最方便
    import pandas as pd
    timestamps = pd.to_datetime(timestamps)

min_time = min(timestamps)
max_time = max(timestamps)

# 3. 计算时间差
time_diff = max_time - min_time

# 4. 打印结果
print(f"最早观看时间: {min_time}")
print(f"最近观看时间: {max_time}")
print(f"时间跨度: {time_diff.days} 天")
print(f"用户 {test_user_id} 的观影记录字典列表：")
# 使用 json.dumps 可以让打印出来的字典格式更美观（需要 import json）
print(json.dumps(test_user_history[:5], ensure_ascii=False, indent=2))


final_prompt = PROMTPT.format(
    movie_seq=test_user_history[:-100],
    movie_list=all_movie_list
)

import asyncio
import os
os.environ["OPENAI_API_KEY"] = "sk-8dfd0034a9d7404b827dad9b02e1e9d4"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
from agents import Agent, function_tool, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings, Runner, ItemHelpers
from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

import openai

client = openai.OpenAI(
    api_key="sk-f5faa996493b49f38215a7382be119a3", # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    model="qwen-flash",
    messages=[
        {
            "role": "system",
            "content": "你是一个电影推荐专家。"
        },
        {
            "role": "user",
            "content": final_prompt
        }
    ]
)
print(f"\n根据用户{test_user_id} 的观影记录进行电影推荐为：")
print(completion.choices[0].message.content)



# 其他方法
# https://www.promptingguide.ai/techniques