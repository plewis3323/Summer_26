# Week 03 — Resources

## Docs

- **PyTorch docs.** https://pytorch.org/docs/stable/index.html. Key pages:
  - "Autograd mechanics": https://pytorch.org/docs/stable/notes/autograd.html
  - `torch.nn` modules: https://pytorch.org/docs/stable/nn.html
  - `torch.optim`: https://pytorch.org/docs/stable/optim.html
- **PyTorch recipes.** https://pytorch.org/tutorials/recipes/recipes_index.html. Bite-sized how-tos.

## Tutorials

- **PyTorch 60-minute Blitz.** https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html.
- **"What is `torch.nn` really?"** by Jeremy Howard. https://pytorch.org/tutorials/beginner/nn_tutorial.html. A guided tour from scratch to `nn.Module`. Complements Karpathy.
- **Lilian Weng's "Optimization" notes.** https://lilianweng.github.io/posts/2017-06-28-optimization/. Clean exposition of SGD → momentum → Adam.
- **distill.pub — "Why Momentum Really Works".** https://distill.pub/2017/momentum/. Interactive and rigorous.

## Videos

- **Karpathy Zero-to-Hero playlist.** https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ. Already assigned Lecture 1.
- **PyTorch Dev Day talks.** YouTube. The "PyTorch internals" talk by Edward Yang is worth an hour — gives you mental models for what `Tensor.grad_fn` really is.

## GitHub repos

- **`karpathy/micrograd`.** https://github.com/karpathy/micrograd. The canonical ~100 lines.
- **`karpathy/makemore`.** https://github.com/karpathy/makemore. Extension of micrograd's world to language models. Useful to peek at before Week 06.
- **`pytorch/examples`.** https://github.com/pytorch/examples. Clean reference implementations of common architectures.
- **`tinygrad/tinygrad`.** https://github.com/tinygrad/tinygrad. George Hotz's "micrograd, but GPU-capable, and kinda production-ish." Reading this after you've built micrograd is illuminating.

## Codecademy (Pro)

- "PyTorch for Deep Learning" — assigned modules.
- "Learn PyTorch: Classification and Regression" — optional continuation.

## Blogs and essays

- **Andrej Karpathy — *A Recipe for Training Neural Networks* (2019).** https://karpathy.github.io/2019/04/25/recipe/. Assigned primary reading; bookmark.
- **Chris Olah — various Distill.pub pieces on neural network interpretation.** Great for intuition, if interpretation is something you want to chase later.
- **Sebastian Raschka — *Understanding Deep Learning* videos and notebooks** on his GitHub (`rasbt/LLMs-from-scratch` starter). Good companion to Karpathy; more methodical pacing.

## Books (optional, for when Goodfellow feels dated)

- **Zhang, Lipton, Li, Smola — *Dive into Deep Learning* (D2L).** https://d2l.ai/. Free. Full code in PyTorch and other frameworks. Covers modern architectures. Use as a reference, not cover-to-cover.
- **Prince — *Understanding Deep Learning* (2023).** https://udlbook.github.io/udlbook/. Visual-first. Excellent bias/variance and generalization chapters.

## Tooling for this week

- **`torchinfo`.** `uv add torchinfo`. `torchinfo.summary(model, (1, 28, 28))` prints a Keras-style layer summary. Sanity check for parameter counts.
- **`lovely-tensors`.** https://github.com/xl0/lovely-tensors. `import lovely_tensors; lovely_tensors.monkey_patch()`. Prints tensors with shape, dtype, stats instead of just numbers. Magic for debugging.
- **`torchviz`.** Visualize the autograd graph of a loss. Useful once; then delete.

## Datasets

- `torchvision.datasets.MNIST` for E6, E7, E8.
- `sklearn.datasets.make_moons` for E4, E5.
- Synthetic for E10 and the project.

## When you get stuck

- **PyTorch Forums.** https://discuss.pytorch.org/. Often the canonical answer, indexed by Google.
- **PyTorch Discord.** Faster for nitty-gritty.
- **`torch.autograd.gradcheck`** — PyTorch's built-in gradient checker. When you're implementing custom autograd functions, use this.

## Further reading

- **Baydin et al. — *Automatic Differentiation in Machine Learning: a Survey* (JMLR 2018).** https://jmlr.org/papers/v18/17-468.html. Deep dive into forward vs reverse mode.
- **Griewank & Walther — *Evaluating Derivatives*.** The textbook. Reach for this if you want to implement AD in a different language or worry about higher-order derivatives.

---

Next: Week 04, CNNs and the Month-1 capstone.
