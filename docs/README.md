# Self-supervised Federated Learning (SSL-FL)

### Label-Efficient Self-Supervised Federated Learning for Tackling Data Heterogeneity in Medical Imaging
IEEE Transactions on Medical Imaging, 2023. [HTML](https://ieeexplore.ieee.org/document/10004993) | [ArXiv](https://arxiv.org/abs/2205.08576) | [Cite](#citation)

**TL;DR:** Pytorch implementation of the self-supervised federated learning framework proposed in [our paper](https://arxiv.org/pdf/2205.08576.pdf) for simulating self-supervised classification on multi-institutional medical imaging data using federated learning.

- Our framework employs masked image encoding as self-supervised task to learn efficient representations from images.
- Extensive experiments are performed on diverse medical datasets including retinal images, dermatology images and chest X-rays.
- In particular, we implement BEiT and MAE as the self-supervision learning module.

<!-- [<img src="figure1.png" width="300px" align="left" />] -->
<img src="figure2.png" width="800px" align="center" />

## Pre-requisites:
### Set Up Environment
```bash
git clone https://github.com/rui-yan/SSL-FL.git
cd SSL-FL
conda env create -f ssfl_conda.yml
conda activate ssfl
```
* NVIDIA GPU (Tested on Nvidia Tesla V100 32G x 4, and Nvidia GeForce RTX 2080 Ti x 8) on local workstations
* Python (3.8.12), torch (1.7.1), timm (0.3.2), numpy (1.21.2), pandas (1.4.2), scikit-learn (1.0.2), scipy (1.7.1), seaborn (0.11.2)
<!--* then ```pip install torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html```-->

### Data Preparation
Please refer to [SSL-FL/data](https://github.com/rui-yan/SSL-FL/tree/main/data) for information on the directory structures of data folders, download links to datasets, and instructions on how to train on custom datasets.

## Self-supervised Federated Learning for Medical Image Classification

In this paper, we selected ViT-B/16 as the backbone for all methods. The specifications for BEiT-B are as follows: #layer=12; hidden=768; FFN factor=4x; #head=12; patch=16x16 (#parameters: 86M).

Please refer to [SSL-FL/data](https://github.com/rui-yan/SSL-FL/tree/main/data) for access to the links to **pre-trained checkpoints** that were used to generate the results.

### Self-supervised Federated Pre-training and fine-tuning

Sample scripts for running Fed-BEiT and Fed-MAE pre-training and finetuning on the Retina dataset can be found in the following directories: [SSL-FL/code/fed_beit/script/retina](https://github.com/rui-yan/SSL-FL/blob/main/code/fed_beit/script/retina/retina_split1_ssl.sh) for Fed-BEiT and [SSL-FL/code/fed_mae/script/retina](https://github.com/rui-yan/SSL-FL/blob/main/code/fed_mae/script/retina/retina_split1_ssl.sh) for Fed-MAE.

To run Fed-BEiT, please download Dall-e tokenizers and save encoder.pkl and decoder.pkl to SSL-FL/data/tokenizer_weight: 
```bash
wget https://cdn.openai.com/dall-e/encoder.pkl
wget https://cdn.openai.com/dall-e/decoder.pkl
```

## Acknowledgements
We sincerely thank the authors of following open-source projects:
- [BEiT](https://github.com/microsoft/unilm/tree/master/beit)
- [MAE](https://github.com/facebookresearch/mae)
- [ViT-FL](https://github.com/Liangqiong/ViT-FL-main)

## Citation
If you find our work helpful in your research or if you use any source codes or datasets, please cite our paper.

```bibtex
@article{yan2023label,
  title={Label-efficient self-supervised federated learning for tackling data heterogeneity in medical imaging},
  author={Yan, Rui and Qu, Liangqiong and Wei, Qingyue and Huang, Shih-Cheng and Shen, Liyue and Rubin, Daniel and Xing, Lei and Zhou, Yuyin},
  journal={IEEE Transactions on Medical Imaging},
  year={2023},
  publisher={IEEE}
}
```
