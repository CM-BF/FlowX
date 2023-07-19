# FlowX: Towards Explainable Graph Neural Networks via Message Flows


[![arXiv](https://img.shields.io/badge/arXiv-2206.12987-b31b1b.svg)](https://arxiv.org/abs/2206.12987)
[![License][license-image]][license-url]

[license-url]: https://github.com/CM-BF/FlowX/blob/main/LICENSE
[license-image]:https://img.shields.io/badge/License-MIT-yellow.svg

## Dependencies

See environment.yml and requirements.txt. The code is also tested on PyTorch 1.10.1, PyG 2.0.4.

## Run FlowX

```shell
python -m benchmark.kernel.pipeline.py --task explain --model_name GCN_3l --dataset_name bbbp --target_idx 0 --explainer FlowX_plus --sparsity 0.7 --force_recalculate
```

## License

This project is licensed under the terms of the MIT license.

## Citing FlowX

If you find FlowX useful in your research, please consider citing:

```bibtex
@article{gui2022flowx,
  title={Flowx: Towards explainable graph neural networks via message flows},
  author={Gui, Shurui and Yuan, Hao and Wang, Jie and Lao, Qicheng and Li, Kang and Ji, Shuiwang},
  journal={arXiv preprint arXiv:2206.12987},
  year={2022}
}
```


