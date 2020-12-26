# tianchi_ft

[NLP中文预训练模型泛化能力挑战赛](https://tianchi.aliyun.com/competition/entrance/531841/information)

## 任务概述

## 难点
1. tnews：15分类，且存在标签分布不均衡的情况
2. ocemotion: 个别标签存在类别不均衡的情况
3. ocnli

## TODO

## 如何使用

### 环境配置

```bash
pip install -r requirements.txt
```

### 数据准备

```bash
# 生成训练/验证/测试集
python gen_data.py
```

### 模型训练

```bash
# DATASET_NAME 数据集名称，可选`ocnli`, `ecemotion`, `tnews`
# PRE_TRAINED_MODEL_NAME 预训练模型，可选 `roberta`
# MAX_SEQ_LEN　最大序列长度，根据数据集具体情况设置，后续应该改为动态长度
bash run_classifier.sh $DATASET_NAME $PRE_TRAINED_MODEL_NAME $MAX_SEQ_LEN
# 跑全部任务
bash run_all.sh
```

### 提交成绩

```bash
bash submit.sh
```

## 模型结果

