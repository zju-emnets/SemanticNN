<div align="center">

# SemanticNN: Compressive and Error-Resilient Semantic Offloading for Extremely Weak Devices

[![arXiv][arxiv-image]][arxiv-url] [![EmNets][EmNets-image]][EmNets-url]

[中文版](./README_CN.md) | [Introduction](#intro) | [Quickstart](#quickstart) | [Baselines](#baselines) | [Citation](#citation)
</div>

<a id="intro"></a>
## 📝 Introduction

SemanticNN (Semantic Neural Network Offloading) is a semantic codec that tolerates bit-level errors in pursuit of semantic-level correctness, enabling compressive and resilient collaborative inference offloading under strict computational and communication constraints.



### Features

- Support for multiple deep semantic communication model architectures
- Integration of XAI (eXplainable AI) methods to improve model transparency
- Support for multiple models, including ResNet50, MobileNetv2, etc.
- Support for multiple datasets, including CIFAR-10/100, ImageNet, etc.



<a id="quickstart"></a>
## 🔥 Quickstart

### Prerequisites
> [!TIP]
> Python 3.6+, PyTorch 1.7+, torchvision, CUDA


```bash
git clone https://github.com/zju-emnets/SemanticNN.git
cd SemanticNN 
pip install -r requirements.txt
```

Download datasets and put them in `data` folder. CIFAR-10 and CIFAR-100 will be downloaded automatically. We also provide pretrained models in the table below. You can download them and put them in the `test_models/base_model` folder for a quick start.

| Dataset | Model | Accuracy |
|-----|-----|-----|
| CIFAR-100 | [ResNet50](https://drive.google.com/file/d/1EcxbK0MlLY_UNQPzdvRr_5pCCyHzx6b1/view?usp=sharing) | 67.91% |
| CIFAR-100 | [MobileNetv2](https://drive.google.com/file/d/1EcxbK0MlLY_UNQPzdvRr_5pCCyHzx6b1/view?usp=sharing) | 58.26% |
| [ImageNet-200](https://drive.google.com/file/d/19Ug2FnZh1FDblP_QOjth2TZlqRKX8dmb/view?usp=sharing)  | [ResNet50](https://drive.google.com/file/d/1UDPZzYV_UdFZseDzUuh_50ijYJMxq2sz/view?usp=sharing) | 79.67% |
| [ImageNet-200](https://drive.google.com/file/d/19Ug2FnZh1FDblP_QOjth2TZlqRKX8dmb/view?usp=sharing)  | [MobileNetv2](https://drive.google.com/file/d/1q4sm8JehISu7KCp9uPusvdD0lV_fYWe4/view?usp=sharing) | 63.67% |


### Original Model training and testing

The trained model will be saved in the `save` folder. You can change the hyperparameter settings (e.g., learning rate `--lr`, batch size `--trainBatchSize`, number of epochs `--epoch`) to get better model training results. 
```bash
python train_models_ours.py \
  --models ResNet50 --datasets cifar100 
```
If you want to test the model you have trained, you need to change the path of the model in the `load_model` function. To get started quickly, you can directly place the pre-trained model in the corresponding folder.
```bash
python train_models_ours.py \
  --models ResNet50 --datasets cifar100 \
  --testonly true
```
### SemanticNN training and testing
We train SemanticNN through two-stage Feature-augmentation Learning methods. By using `--stage` to control the training phase. We provide four basemodel architectures, including `ResNetonImageNet`, `ResNetoncifar100`, `MobilenetonImageNet`, `Mobilenetoncifar100`. You can choose the basemodel architecture by setting `--basemodel`. Note that the basemodel architecture must be consistent with the dataset `--datasets`.
#### 1. Denoising Autoencoder Training
First, carry out the *Denoising Autoencoder Training* as the first stage:
```bash
python train_models_ours.py \
  --models semanticnn --datasets imagenet \
  --basemodel ResNetonImageNet \
  --trainBatchSize 128 --epoch 50\
  --stage 1
```
#### 2. Task-Oriented Semantic-level Training
Then carry out the *Task-Oriented Semantic-level Training* as the second stage. You can enable XAI methods by setting `--useXAI true`. Note that the XAI methods will increase the training time. By default, `--useXAI` is set to `false`.
```bash
python train_models_ours.py \
  --models semanticnn --datasets imagenet \
  --basemodel ResNetonImageNet \
  --trainBatchSize 128 --epoch 50 
  --stage 2 --useXAI true 
```
#### 3. SemanticNN Testing
Change the path of your trained semanticNN in the `load_model` function for model testing. Note that the basemodel architecture `--basemodel` and the dataset `--datasets` must be consistent with the testing model.
```bash
python train_models_ours.py \
  --models semanticnn --datasets imagenet \
  --basemodel ResNetonImageNet \
  --testonly true 
```




<a id="baselines"></a>
## 📊 Baseline methods



This project implements various semantic communication models, mainly including:

1. [**ADJSCC**](https://arxiv.org/abs/2012.00533): You can add any ADJSCC structure in `DeepSC.py` and train it by defining `self.model` within *semanticnn*. We have also provided various ADJSCC versions (only support `ResNetonImageNet` and `ResNetoncifar100`) in DeepSC.py. 

2. [**DeepCOD**](https://dl.acm.org/doi/10.1145/3384419.3430898): Setting `--deepcodkd true` will enable the DeepCOD method.
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
