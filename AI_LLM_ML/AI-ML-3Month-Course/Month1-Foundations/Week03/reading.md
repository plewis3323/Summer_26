# Week 03 — Reading

## Primary

1. **Goodfellow, Bengio, Courville, *Deep Learning*, Ch. 4 "Numerical Computation".**
   - §4.1 Overflow and underflow (pp. 79–80). The log-sum-exp trick and why `log_softmax` is a thing.
   - §4.3 Gradient-based optimization (pp. 82–94). Calculus-refresher pace. Skim if comfortable; linger on §4.3.1.

2. **Goodfellow, Bengio, Courville, *Deep Learning*, Ch. 6 "Deep Feedforward Networks".**
   - §6.1 Example: XOR (pp. 168–171). Motivation.
   - §6.2 Gradient-based learning (pp. 172–183). Loss functions, output units.
   - §6.3 Hidden units (pp. 185–193). ReLU, tanh, sigmoid, GELU-era is still in the future for this text.
   - §6.4 Architecture design (pp. 193–197).
   - §6.5 Back-Propagation and Other Differentiation Algorithms (pp. 197–217). This is the single most important chapter in the book for this week. Read it twice. Work through Algorithms 6.1, 6.2, 6.4.

3. **Karpathy, *Neural Networks: Zero to Hero*, Lecture 1 — "The spelled-out intro to neural networks and backpropagation: building micrograd"** (~2h25m).
   - Read alongside the code at https://github.com/karpathy/micrograd.
   - 0:00–12:00 calc refresher; skim.
   - 12:00–55:00 Value class; follow along typing.
   - 55:00–1:25:00 backward(); pause and verify you can re-derive each local gradient.
   - 1:25:00–2:00:00 MLP class; neural net from Values.
   - 2:00:00–end training a tiny classifier.

4. **Karpathy blog — *A Recipe for Training Neural Networks* (2019).** https://karpathy.github.io/2019/04/25/recipe/.
   - Short. Read the whole thing. Re-read at the start of Week 04 and again in Week 07.

## Supplementary

5. **Bishop, *PRML*, §5.1–5.3.** Classical backprop derivation with the textbook notation. Good for seeing the math twice.

6. **Murphy, *Probabilistic Machine Learning: An Introduction*, Ch. 13 "Neural Networks for Unstructured Data".** Free online. Modern notation; covers optimizers and initialization updates the GBC book is silent on.

7. **Karpathy, *Neural Networks: Zero to Hero*, Lecture 2 — "The spelled-out intro to language modeling: building makemore"** (~1h57m). Only the first hour is relevant this week (bigram → MLP language model). The rest queues for Week 06.

8. **PyTorch official tutorial — "Deep Learning with PyTorch: A 60 Minute Blitz".** https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html. Skim. Good for nomenclature and code idioms.

9. **3Blue1Brown — *Backpropagation calculus*** (Video 4 of the Neural Networks playlist, ~10 min). If you didn't watch it in Week 01, now's the time.

## Codecademy (Pro)

- **"PyTorch for Deep Learning" — modules 1–3** ("Introduction to PyTorch", "Tensors", "Your First Neural Network"). Skim if pace is slow; use for finger-memory of the idioms.
- **"Learn PyTorch: Classification and Regression"** — optional but useful if you want more reps on the fit/eval loop before Week 04.

## Papers (skim only)

- **Rumelhart, Hinton, Williams — "Learning Representations by Back-Propagating Errors" (Nature 1986).** Historical. Read pp. 533–534 only; it's how the modern world started.
- **Hornik, Stinchcombe, White — "Multilayer Feedforward Networks are Universal Approximators" (Neural Networks 1989).** Skim the abstract and Theorem 2. That's enough — don't get lost in function-analytic details.

## Notes to take

Keep one personal reference document by end of week:

1. A derivation of the softmax+cross-entropy gradient in your own handwriting.
2. A table: op → forward → local gradient (for +, *, ReLU, tanh, exp, log, MatMul).
3. Your own Karpathy recipe, compressed to 10 bullet points that sound like you.
4. One paragraph: "What is the difference between a PyTorch `Tensor` and a numpy `ndarray`?" in words you'd say to a labmate.

These show up again in Week 04 (CNN architecture design) and Week 06 (transformer internals). Don't skip.
