<div align="center">
  <img src=".\resources\iMIX-LOGO.png" width="600"/>
</div>

## Introduction

English | [简体中文](README_zh-CN.md)

Inspur Multimodal Artificial Intelligence Framework (iMIX) is an open source multi-modal model building toolbox. This framework is based on the out-of-the-box design concept. It is compatible with rich multi-modal tasks, models and datasets. It is scalable, ease to use and in high performance.

The master branch works based on PyTorch.

Documentation: https://iMIX.readthedocs.io/

![demo_image](resources/0.png)

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog

v1.0.0 is our first release. iMIX v1.0.0 supports mainstream multi-modal datasets, models and mixed precision training. And it supports distribute training across multiple GPUs and multiple nodes.

iMIX's subsequent version will optimize the framework further. We will add more dual-stream and single-stream pre-training models, add more data process methods such as mask, back translation and unsupervised data enhancement, and support launch multiple jobs for training on a single machine simultaneously.

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/getstart/model_zoo.md).

All supported models and tasks are shown in the table below.

Supported backbones:

| task              | LXMERT | UNITER | ViLBERT | DeVLBert | Oscar | VinVL | MCAN | LCGN | HGL  | R2C  | VisDial-BERT |
| ----------------- | ------ | ------ | ------- | -------- | ----- | ----- | ---- | ---- | ---- | ---- | ------------ |
| VQA               | √      | √      | √       | √        | √     | √     | √    |      |      |      |              |
| GQA               | √      |        | √       |          | √     | √     |      | √    |      |      |              |
| NLVR              | √      | √      |         |          | √     | √     |      |      |      |      |              |
| VQA_large         |        |        |         |          | √     |       |      |      |      |      |              |
| NLVR_large        |        |        |         |          | √     | √     |      |      |      |      |              |
| GussWhatPointing  |        |        | √       |          |       |       |      |      |      |      |              |
| VisualEntailment  |        | √      | √       |          |       |       |      |      |      |      |              |
| GussWhat          |        |        | √       |          |       |       |      |      |      |      |              |
| VCR_QAR           |        |        |         | √        |       |       |      |      | √    | √    |              |
| VCR_QA            |        |        |         | √        |       |       |      |      | √    | √    |              |
| Visual7w          |        |        | √       |          |       |       |      |      |      |      |              |
| RetrivalFlickr30k |        |        | √       |          |       |       |      |      |      |      |              |
| GenomeQA          |        |        | √       |          |       |       |      |      |      |      |              |
| Retrivalcoco      |        |        | √       |          |       |       |      |      |      |      |              |
| refcocog          |        |        | √       |          |       |       |      |      |      |      |              |
| refcoco           |        |        | √       |          |       |       |      |      |      |      |              |
| refcoco+          |        |        | √       | √        |       |       |      |      |      |      |              |
| VisDial           |        |        |         |          |       |       |      |      |      |      | √            |

## Installation

Please refer to [get_started.md](docs/getstart/get_started.md) for installation.

## Getting Started

Please see [quickrun](docs/Quickrun/1_exist_data_model.md) for the basic usage of iMIX and visual interface for inference.
We provide basic introduction of iMIX core module [engine](docs/tutorials/Tutorial-engine.md), full guidance for [configuration](docs/tutorials/Tutorial1-config.md), and all the [results and model](docs/getstart/model_zoo.md).
There are also tutorials for [finetuning models](docs/tutorials/Tutorial6-finetune.md), [adding new dataset](docs/tutorials/Tutorial2-customize_dataset.md), [customizing models](docs/tutorials/Tutorial3-customize_models.md), [customizing runtime settings](docs/tutorials/Tutorial4-customize_Schedule_and_Runtime_Settings.md) and [useful tools](docs/log_visualization.md).

## Contributing

We appreciate all contributions to improve iMIX. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

iMIX is an open source project that is contributed by researchers and engineers from our company Inspur. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new detectors.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@misc{fan2021iMIX,
  author =       {Baoyu Fan, Liang Jin, Runze Zhang, Xiaochuan Li, Cong Xu, Hongzhi Shi, Jian Zhao, Yinyin Chao, Yingjie Zhang, Binqiang Wang, Zhenhua Guo, Yaqian Zhao, Rengang Li},
  title =        {iMIX: A multimodal framework for vision and language research},
  howpublished = {\url{https://github.com/inspur-hsslab/iMIX}},
  year =         {2021}
}
```
