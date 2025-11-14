<div align="center">

# SemanticNN: Compressive and Error-Resilient Semantic Offloading for Extremely Weak Devices

[![arXiv][arxiv-image]][arxiv-url] [![EmNets][EmNets-image]][EmNets-url]

[英文版](./README.md) | [简介](#intro) | [快速开始](#quickstart) | [基线方法](#baselines) | [引用](#citation)
</div>

<a id="intro"></a>
## 📝 简介

SemanticNN（语义神经网络卸载）是一种能够容忍位级错误的语义编解码器，追求语义级正确性。在严格的计算和通信约束下，实现压缩和弹性协作卸载。



### Features

- 支持多种语义通信的深度神经网络模型架构
- 集成可解释AI（XAI）方法，提高模型透明度
- 支持多种模型，包括ResNet50、MobileNetv2等
- 支持多种数据集，包括CIFAR-10/100、ImageNet等



<a id="quickstart"></a>
## 🔥 快速开始

### 环境要求
> [!TIP]
> Python 3.6+, PyTorch 1.7+, torchvision, CUDA


```bash
git clone https://github.com/zju-emnets/SemanticNN.git
cd SemanticNN 
pip install -r requirements.txt
```

下载数据集并将它们放入data文件夹。CIFAR-10和CIFAR-100将自动下载。下表还提供了预训练模型。您可以下载它们并将它们放入"test_models/base_model"文件夹以快速开始。

| 数据集 | 模型 | 准确率 |
|-----|-----|-----|
| CIFAR-100 | [ResNet50](https://drive.google.com/file/d/1EcxbK0MlLY_UNQPzdvRr_5pCCyHzx6b1/view?usp=sharing) | 67.91% |
| CIFAR-100 | [MobileNetv2](https://drive.google.com/file/d/1EcxbK0MlLY_UNQPzdvRr_5pCCyHzx6b1/view?usp=sharing) | 58.26% |
| [ImageNet-200](https://drive.google.com/file/d/19Ug2FnZh1FDblP_QOjth2TZlqRKX8dmb/view?usp=sharing)  | [ResNet50](https://drive.google.com/file/d/1UDPZzYV_UdFZseDzUuh_50ijYJMxq2sz/view?usp=sharing) | 79.67% |
| [ImageNet-200](https://drive.google.com/file/d/19Ug2FnZh1FDblP_QOjth2TZlqRKX8dmb/view?usp=sharing)  | [MobileNetv2](https://drive.google.com/file/d/1q4sm8JehISu7KCp9uPusvdD0lV_fYWe4/view?usp=sharing) | 63.67% |


### 原始任务模型的训练和测试

训练好的模型将保存在save文件夹中。您可以更改超参数设置（例如，学习率`--lr`, 批处理大小`--trainBatchSize`,训练轮数`--epoch`）以获得更好的模型训练结果。
```bash
python train_models_ours.py \
  --models ResNet50 --datasets cifar100 
```
如果你想测试你训练好的模型，你需要更改`train_models_ours.py`中的`load_model`函数中的模型路径。为了快速开始，你可以直接将预训练模型放置在相应的文件夹中。
```bash
python train_models_ours.py \
  --models ResNet50 --datasets cifar100 \
  --testonly true
```
### SemanticNN的训练和测试
我们通过两阶段特征增强学习方法训练SemanticNN。通过使用`--stage`来控制训练阶段。我们提供了四种基础模型架构，包括`ResNetonImageNet`、`ResNetoncifar100`、`MobilenetonImageNet`、`Mobilenetoncifar100`。您可以通过设置`--basemodel`来选择基础模型架构。请注意，基础模型架构必须与数据集`--datasets`一致。
#### 1. 去噪自编码器训练
首先，执行*去噪自编码器训练*作为第一阶段：
```bash
python train_models_ours.py \
  --models semanticnn --datasets imagenet \
  --basemodel ResNetonImageNet \
  --trainBatchSize 128 --epoch 50\
  --stage 1
```
#### 2. 任务导向语义级训练
然后进行第二阶段的*任务导向语义级训练*。您可以通过设置 `--useXAI true` 来启用XAI方法。请注意，XAI方法将增加训练时间。默认情况下，`--UseXAI` 被设置为 `false`。
```bash
python train_models_ours.py \
  --models semanticnn --datasets imagenet \
  --basemodel ResNetonImageNet \
  --trainBatchSize 128 --epoch 50 
  --stage 2 --useXAI true 
```
#### 3. SemanticNN Testing
#### 3. 语义NN测试
在`load_model`函数中更改您训练的semanticNN路径，以进行模型测试。请注意，基本模型架构 `--basemodel` 和数据集 `--datasets` 必须与测试模型一致。
```bash
python train_models_ours.py \
  --models semanticnn --datasets imagenet \
  --basemodel ResNetonImageNet \
  --testonly true 
```




<a id="baselines"></a>
## 📊 基线方法

本项目实现了各种基线方法，主要包括：

1. [**ADJSCC**](https://arxiv.org/abs/2012.00533): 可以在 `DeepSC.py` 中添加任何ADJSCC结构，并通过在 *semanticnn* 中的`self.model`来定义并训练它。我们还在DeepSC.py中提供了各种ADJSCC版本（仅支持`ResNetonImageNet`和`Resetoncifar100`）。

2. [**DeepCOD**](https://dl.acm.org/doi/10.1145/3384419.3430898): 
设置 `--deepcodkd true` 将启用DeepCOD方法。
    ```bash
    python train_models_ours.py \
      --models semanticnn --datasets imagenet \
      --basemodel ResNetonImageNet \
      --trainBatchSize 128 --epoch 50\
      --deepcodkd true
    ```

<a id="citation"></a>
## 🙌 Contribution and Citation

Bug reports and feature requests are welcome. If you would like to contribute to this project, please feel free to submit pull requests.

If you use the code or ideas from this project, please consider citing the relevant papers (specific citation format to be added).
```bibtex
@misc{
  title = {SemanticNN: Compressive and Error-Resilient Semantic Offloading for Extremely Weak Devices},
  author = {Jiaming Huang, Yi Gao*, Fuchang Pan, Renjie Li, and Wei Dong*.},
  year = {2025}
}
```



[arxiv-image]: https://img.shields.io/badge/Paper-arXiv-B31B1B
[arxiv-url]: https://github.com/zju-emnets/SemanticNN/blob/main/SemanticNN.pdf
[EmNets-image]: https://img.shields.io/badge/ZJU-EmNets-blue
[EmNets-url]: https://emnets.cn
