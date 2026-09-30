# Third-party acknowledgments

The Massive Lion implementation and release-specific code use the repository's
[MIT license](LICENSE). The following notices apply to adapted components.

- **Cautious optimizers:** `massive_lion/baselines/cautious.py` adapts the
  C-Optim implementations. Copyright (c)
  2024 Kaizhao Liang. The original MIT notice is preserved in
  [C_OPTIM_LICENSE](massive_lion/baselines/C_OPTIM_LICENSE).
- **Language model and data pipeline:** `experiments/language_model.py` and
  the language preparation/loading code adapt the Transformer++ / plainLM setup,
  with credit to [plainLM](https://github.com/Niccolo-Ajroldi/plainLM) by Niccolò
  Ajroldi. Its [MIT notice](experiments/PLAINLM_LICENSE) is preserved.
- **Muon:** the comparison wrapper delegates the ordinary optimizer to
  `torch.optim.Muon`; the curvature-aware comparison uses the Newton–Schulz
  method attributed to Keller Jordan's [Muon](https://github.com/KellerJordan/Muon).
  Its [MIT notice](massive_lion/baselines/MUON_LICENSE) is preserved.
- **PyTorch and torchvision:** these are installed dependencies. The CIFAR
  model uses torchvision's ResNet-18 implementation; it is not vendored here.

Datasets, tokenizers, and model checkpoints are not distributed by this
repository. Data preparation downloads from the sources named in its scripts;
those resources retain their respective upstream terms.
