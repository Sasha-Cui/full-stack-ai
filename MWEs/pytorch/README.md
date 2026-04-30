# PyTorch Tutorial

This folder is the recommended starting point if you want to learn the rest of the repository without treating the frameworks as black boxes.

## Files

- `pytorch_tutorial.ipynb`: main notebook.
- `requirements.txt`: minimal dependency list for this folder.

## Topics

- Tensor basics and device movement
- Autograd and gradient flow
- `torch.no_grad()` vs `torch.inference_mode()`
- Custom autograd functions
- Optimizer configuration
- Mixed precision
- Debugging and profiling helpers
- Memory inspection

## Install

```bash
pip install -r requirements.txt
```

If you want GPU-backed PyTorch wheels, replace the default `torch` install with the command recommended on [pytorch.org](https://pytorch.org/get-started/locally/).

## Run

```bash
jupyter notebook pytorch_tutorial.ipynb
```

## Validation

This notebook was executed locally in a CPU-only Python 3.12 environment after installing `requirements.txt`.

## Suggested Order

1. Run the tensor and autograd sections.
2. Read the custom autograd example slowly.
3. Skip GPU-only cells if you are on CPU.
4. Return to the debugging and memory sections once you start training real models elsewhere in the repo.

## Good Companion Material

- [PyTorch docs](https://pytorch.org/docs/stable/)
- [Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
- [PyTorch tutorials](https://pytorch.org/tutorials/)
