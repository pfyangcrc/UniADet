<p align="center"> <a href="README.md">English</a> | <b>ZH</b></p>

# UniADet
UniADet 的非官方 pytorch 复现.

## 项目使用说明

### 1. 环境准备

安装项目依赖

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. 训练流程

1. 修改配置文件 `config_train.yaml`，设置数据集路径、模型参数、迭代轮数、批次大小等训练相关参数
2. 启动训练脚本

```
python train.py
```

训练完成后，模型权重将自动保存至指定目录。

### 3. 评估 / 推理流程

1. 修改配置文件 `config_infer.yaml`，配置待加载的训练权重路径、测试集路径、评估指标等推理参数
2. 运行推理评估脚本

```
python infer.py
```

脚本加载训练好的模型，在测试集上完成评估，并输出指标结果。

## 文件简要说明

- `config_train.yaml`：训练阶段配置文件
- `config_infer.yaml`：推理 & 评估阶段配置文件
- `train.py`：模型训练入口脚本
- `infer.py`：模型评估推理入口脚本

## 使用注意事项

1. 训练与推理使用各自独立的 yaml 配置，参数分开管理，不要混用
2. 推理配置中权重路径需要和训练保存路径保持一致
3. 如需修改网络、数据增强等参数，在`config_train.yaml`中调整


