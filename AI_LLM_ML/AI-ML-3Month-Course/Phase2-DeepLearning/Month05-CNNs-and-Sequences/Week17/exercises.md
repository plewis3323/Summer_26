# Week 17 — Exercises

Work top to bottom. Setup (imports, data loading, constants) is given by the
notebook; you write only the lines each exercise asks for. All exercises live in
notebook cells this week — no file-based deliverables.

## E1 — Convolution by hand (in NumPy)

Write `conv2d_loops(image, kernel, stride, pad)`: zero-pad the image, then two loops
over output positions, each computing one windowed multiply-and-sum. No channels, no
batch — a single 2D array in, a 2D array out.
Hint: compute the output shape first with the lesson §3 formula, allocate it with
`np.zeros`, then fill it; the window is `padded[r*stride : r*stride+k, c*stride : c*stride+k]`.
Accept when: output matches `torch.nn.functional.conv2d` to 1e-5 on random inputs for
all three provided settings: (k=3, s=1, p=0), (k=5, s=1, p=2), (k=3, s=2, p=1).

## E2 — Shape and parameter calculator

Write `layer_arithmetic(specs)` that walks a list of layer specs (each a dict with
`kind`, `k`, `s`, `p`, `c_out`) from a given input shape and returns the running
(channels, height, width) plus each conv layer's parameter count.
Hint: pooling changes shape by the same formula as conv but adds zero parameters.
Accept when: reproduces every row of the lesson §9 LeNet table, and its total
parameter count equals `sum(p.numel() for p in model.parameters())` for the provided
LeNet — 61,706.

## E3 — Hand-designed kernels

Apply three fixed kernels — box blur, vertical Sobel, horizontal Sobel — to one
FashionMNIST sneaker image using your E1 function, and plot the input plus three
feature maps in one figure.
Hint: kernels are given in setup; reuse `conv2d_loops` with s=1, p=1.
Accept when: the figure renders with the blur visibly smoothed and the two Sobel maps
lighting up on perpendicular edges, and one markdown line connects Sobel to what a
trained CNN's first layer learns.

## E4 — LeNet on FashionMNIST

Fill in the LeNet `forward` and the training step (the class skeleton, loaders, loss,
and optimizer are given) and train for 5 epochs.
Hint: this is lesson §9; write the forward pass from the shape table, not from memory.
Accept when: test accuracy ≥ 89% within 5 epochs on CPU or modest GPU, and your E2
calculator's shape trace matches the shapes printed from a forward hook in setup.

## E5 — Depth with and without skips

Build two 20-layer 3×3 convnets on a 10k-image CIFAR-10 subset (loader given): plain
stacked convs, and the same stack with an identity skip added around every pair of
conv layers. Train both with the identical loop and plot both training-loss curves on
one axis.
Hint: a skip around a pair is `x = x + block(x)`; keep channel counts constant (16)
so no 1×1 convs are needed.
Accept when: the plot shows the plain net training visibly slower or plateauing higher
than the skip net (the Week-17 degradation picture), and one line explains the gap via
the $(I + J_F)$ gradient term.

## E6 — Parameter audit (synthesis)

Count parameters of your E4 LeNet and of an MLP (784–256–128–10, given) trained with
the same loop and epochs on FashionMNIST. Report both counts, both test accuracies,
and the ratio of parameters per point of accuracy.
Hint: reuse E2 for the CNN count; the MLP count you can do on paper first.
Accept when: printed counts match `sum(p.numel())` for both models, the CNN matches or
beats the MLP's accuracy with fewer parameters, and a 2–3 line markdown cell states
which structural assumptions bought the savings.

## Review

1. (Week 13) Universal approximation says an MLP could learn any image function. Why
   build translation equivariance in anyway? Answer in terms of sample efficiency.
2. (Week 16) Which initialization does a ReLU conv layer want, and what is $n_{in}$
   for a 3×3 kernel with 64 input channels?
3. (Week 15) In the E4 training step, what goes wrong if `opt.zero_grad()` is dropped,
   and which Week-14 design decision explains it?
4. (Week 06) A convolution is a linear map. What kind of input images does a pure
   edge-detector kernel send to zero — i.e. what lives in its null space?
