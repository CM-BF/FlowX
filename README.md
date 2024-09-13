# FlowX: Towards Explainable Graph Neural Networks via Message Flows [TPAMI 2023]

[![TPAMI](https://img.shields.io/badge/TPAMI-2023.3347470-blue)]([https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10374255](https://ieeexplore.ieee.org/document/10374255))
[![arXiv](https://img.shields.io/badge/arXiv-2206.12987-b31b1b.svg)](https://arxiv.org/abs/2206.12987)
[![License][license-image]][license-url]

[license-url]: https://github.com/CM-BF/FlowX/blob/main/LICENSE
[license-image]:https://img.shields.io/badge/License-MIT-yellow.svg

## Installation

- Ubuntu 20.04
- PyTorch 1.10.1
- PyG 2.0.4
- Others: Please refer to environment.yml and requirements.txt

## Run FlowX

```shell
python -m xgraph.kernel.pipeline.py --task explain --model_name GCN_3l --dataset_name bbbp --target_idx 0 --explainer [FlowX_plus/FlowX_mius/...] --sparsity 0.7 --force_recalculate
```

- `task`: [`train`, `test`, `explain`] controls the phase of explanations. Please run `train` before explaining.
- `model_name`: [`GCN_3l`, `GIN_3l`] controls the GNN model waiting to be explained.
- `dataset_name` controls the dataset used to be explained.
- `target_idx` determines which task to explain when there are multiple tasks. When there is only 1 task, set it to be 0.
- `explainer` determines the explainer used to explain the model we choose. Options: [FlowX_plus, FlowX_minus, other baselines]
- `sparsity` is a metric & hyperparameter that we can control to determine what percentage of edges/nodes/flows we want the explainer to return.
- `force_recalculate` is generally used for debugging. When an explainer generate an explanation, this explanation will be saved and reused when needed. However, when this flag is set, explainers will never use saved explanations.

## License

This project is licensed under the terms of the MIT license.

## Citing FlowX

If you find FlowX useful in your research, please consider citing:

```bibtex
@ARTICLE{gui2024flowx,
  author={Gui, Shurui and Yuan, Hao and Wang, Jie and Lao, Qicheng and Li, Kang and Ji, Shuiwang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={FlowX: Towards Explainable Graph Neural Networks via Message Flows}, 
  year={2024},
  volume={46},
  number={7},
  pages={4567-4578},
  keywords={Graph neural networks;Task analysis;Predictive models;Electronic mail;Training;Philosophical considerations;Mutual information;Deep learning;explainability;graph neural networks;message passing neural networks},
  doi={10.1109/TPAMI.2023.3347470}}


```


