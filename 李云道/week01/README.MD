## 第一周作业

1、配置好python环境，包括常见的jieba、sklearn、pytorch等
（注意：这个作业截图提交）

2、使用 dataset.csv数据集完成文本分类操作，需要尝试2种不同的模型。（注意：这个作业代码实操提交）

### KNN
```text
E:\Go\conda_envs\py312\python.exe D:\2025.0717BDLearnAI\FastNLP\homework-new\week01\run_on_sklearn_knn.py 
Building prefix dict from the default dictionary ...
Loading model from cache C:\Users\Hide\AppData\Local\Temp\jieba.cache
Loading model cost 0.959 seconds.
Prefix dict has been built successfully.
还有 双鸭山 到 淮阴 的 汽车票 吗 13 号 的
模型: KNN(n_neighbors=1), 待预测的文本: 我要 去 看 一下 郭德纲 的 小品, 预测结果: ['FilmTele-Play']
模型: KNN(n_neighbors=2), 待预测的文本: 我要 去 看 一下 郭德纲 的 小品, 预测结果: ['FilmTele-Play']
模型: KNN(n_neighbors=3), 待预测的文本: 我要 去 看 一下 郭德纲 的 小品, 预测结果: ['FilmTele-Play']
模型: KNN(n_neighbors=4), 待预测的文本: 我要 去 看 一下 郭德纲 的 小品, 预测结果: ['Travel-Query']

```

### SVM
```text
```