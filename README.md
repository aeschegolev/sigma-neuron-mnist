# Superconducting Σ-neuron perceptron on MNIST

Code accompanying the manuscript *"Digital–analogue concept for superconducting
perceptron-like neural networks"* (submitted to *Physical Review E*).

Two **separate** single-hidden-layer perceptrons classify MNIST digits. They share
the same architecture and training protocol and differ only in the hidden
activation:

| Script | Hidden activation |
|---|---|
| `script-SigmaMath.py` | ideal logistic sigmoid |
| `script-SigmaExp.py` | measured Σ-neuron transfer characteristic, read from `SigmaExp.csv` |

Common architecture and training: `784 → 500 → 10` fully connected layers without
bias, batch normalisation before the hidden activation and the read-out, dropout
0.2, a rectified-linear read-out with `argmax`, mean-squared-error loss on one-hot
targets, the Adam optimiser (learning rate 0.005 with a cosine schedule down to
1e-4), 20 epochs, batch size 128, and weights hard-clipped to `[-1/500, +1/500]`
after every update.

## Requirements

```
pip install -r requirements.txt
```

Python 3.9+; a CUDA GPU is used if available, otherwise the code runs on CPU.
MNIST is downloaded automatically by `torchvision` into `./data/` on first run.

## Run

```
python script-SigmaMath.py     # ideal-sigmoid network
python script-SigmaExp.py      # measured Σ-neuron (LUT) network
```

Each run prints per-epoch metrics and a classification report, and saves
`learning_curves.png` and `confusion_matrix.png`.

## The measured activation (`SigmaExp.csv`)

`script-SigmaExp.py` reads its hidden activation from `SigmaExp.csv`: two columns
`x,y` (that is, `φ_in, φ_out`). The loader (`CSVTabularActivation`) interpolates
**by index**, so it assumes a uniform `x` grid. The file shipped here is the
**experimentally measured** Σ-neuron transfer characteristic used in the paper
(192 points, domain ≈ [−0.6, 0.6], `y ∈ [0, 1]`). 

## Results (MNIST test set, 10,000 images)

| Metric | SigmaMath | SigmaExp |
|---|---|---|
| Test accuracy | 97.0 % | 91.9 % |
| Macro-averaged F1 | 0.970 | 0.918 |
| Best per-class F1 | 0.989 (digit 1) | 0.974 (digit 1) |
| Worst per-class F1 | 0.955 (digit 9) | 0.870 (digit 9) |

## License

No license has been assigned yet, so default copyright ("all rights reserved")
applies. Before publication we recommend adding a permissive license (e.g. MIT) so
that reviewers and readers can run and reuse the code; add a `LICENSE` file and
update this section.

## Citation

If you use this code, please cite the manuscript (see `CITATION.cff`).
