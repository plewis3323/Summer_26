# Week 17 — Convolutions

~4 hrs of reading and derivation. Before starting you should be able to: build and
train an MLP in PyTorch with `nn.Module`, `DataLoader`, and a training loop (Week 15);
derive backprop for a two-layer network on paper (Week 13); state what Xavier/He
initialization preserves and read an activation histogram (Week 16); multiply matrices
and reason about a linear map's structure (Week 06).

## 1. Why flat vectors waste an image

Every network you have built so far eats a flat vector. To feed it a 28×28 grayscale
image you flatten the image into 784 numbers and connect every input to every hidden
unit. A first layer with 128 hidden units then holds

$$784 \times 128 + 128 = 100{,}480$$

parameters — and that is a small image. A 256×256 image would need 8.4 million
parameters in the first layer alone.

Worse than the count is what the flattening throws away. In an image, nearby pixels
are related: an edge, a corner, the rim of a shoe in FashionMNIST is a *local* pattern
of a few adjacent pixels. A dense layer has no idea which inputs are adjacent — permute
the 784 pixels the same way in every image and the MLP trains exactly as well, because
nothing in its structure knows about neighborhoods.

And there is a symmetry. A shoe in the top-left corner and the same shoe in the
bottom-right are the same shoe. The dense layer must learn "shoe detector, top-left"
and "shoe detector, bottom-right" as unrelated sets of weights, from separate examples.
Physics analogy (explained, as always): a particle detector's face is uniform — a
photon hitting one spot deposits the same pattern of energy as a photon hitting another
spot. A model of "what a photon looks like" should not depend on *where* it hit.

The fix is to build the two facts — locality and translation symmetry — into the layer
itself. That layer is the convolution.

## 2. The convolution operation

Start in one dimension. Take an input signal $x$ of length 7 and a small weight vector
$w$ of length 3, called a **kernel** (or **filter**). Slide the kernel along the input;
at each position, multiply elementwise and sum:

$$y_i = \sum_{j=0}^{2} w_j \, x_{i+j}.$$

Concretely, with $x = [1, 2, 3, 4, 5, 6, 7]$ and $w = [1, 0, -1]$:

$$y_0 = 1\cdot1 + 0\cdot2 + (-1)\cdot3 = -2, \qquad
y_1 = 1\cdot2 + 0\cdot3 + (-1)\cdot4 = -2, \quad \ldots$$

giving $y = [-2, -2, -2, -2, -2]$. This kernel computes "left neighbor minus right
neighbor" everywhere — a slope detector. Note the output is shorter than the input
(5 vs 7): the kernel has 5 valid starting positions on a length-7 input.

(Strictly, mathematicians call this operation *cross-correlation* and reserve
"convolution" for the version with a flipped kernel. Deep learning calls it
convolution and never flips, because the kernel is learned — a learned kernel and its
flip are the same family. We follow the deep-learning convention.)

In two dimensions the kernel is a small grid, say 3×3, and it slides over the image in
both directions:

$$y_{r,c} = \sum_{u=0}^{2}\sum_{v=0}^{2} w_{u,v}\, x_{r+u,\, c+v}.$$

The output $y$ is called a **feature map**: a map of "how strongly does the pattern
$w$ appear at each location". Classic hand-designed kernels make this concrete:

- a kernel of all $1/9$ averages a 3×3 patch — a blur;
- $[[-1,0,1],[-2,0,2],[-1,0,1]]$ (the Sobel kernel) responds to vertical edges;
- its transpose responds to horizontal edges.

A convolutional layer learns its kernels by gradient descent, exactly like every
weight you have trained so far. The first layer of a trained CNN almost always learns
edge and blob detectors on its own — Sobel filters rediscovered from data.

## 3. Padding, stride, and the output-size formula

Two knobs control how the kernel slides.

**Padding** $p$: add $p$ rings of zeros around the input border. Without padding, the
output shrinks by $k-1$ (kernel size $k$) at every layer, and border pixels appear in
fewer windows than central pixels. With $p = (k-1)/2$ (for odd $k$), the output is the
same size as the input — called "same" padding.

**Stride** $s$: move the kernel $s$ pixels at a time instead of 1. Stride 2 halves the
output resolution, a cheap way to shrink the map.

**Derivation of the output size.** Work along one dimension. The padded input has
length $n + 2p$. A window of length $k$ occupies positions $t, t+1, \ldots, t+k-1$,
where $t$ is the start index. The first start is $t = 0$; a start is valid while the
window fits, i.e. $t + k \le n + 2p$, so the last valid start is $t_{\max} = n + 2p - k$.
With stride $s$ the starts are $0, s, 2s, \ldots$ — every multiple of $s$ up to
$t_{\max}$. The number of multiples of $s$ in $[0, t_{\max}]$ is
$\lfloor t_{\max}/s \rfloor + 1$ (the $+1$ counts the start at 0). Therefore

$$n_{\text{out}} = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1.$$

Sanity checks: $n=7, k=3, p=0, s=1$ gives $\lfloor 4/1 \rfloor + 1 = 5$, matching the
example above. Same padding: $n=28, k=5, p=2, s=1$ gives $\lfloor 27/1 \rfloor + 1 = 28$.
Stride 2: $n=28, k=2, p=0, s=2$ gives $\lfloor 26/2 \rfloor + 1 = 14$.

Memorize the formula by re-deriving it once on paper — it is window-counting, nothing
more. You will use it every time you design a network.

## 4. Channels and parameter counting

Real inputs have **channels**: a color image has 3 (red, green, blue), a grayscale
image 1, and inside a CNN a layer's input is the previous layer's stack of feature
maps — often 32, 64, 256 channels. Convention: a batch of images is a tensor of shape
`(batch, channels, height, width)`.

A convolutional layer with $C_{\text{in}}$ input channels and $C_{\text{out}}$ output
channels works like this. Each output channel has its own filter of shape
$C_{\text{in}} \times k \times k$: it convolves *every* input channel with its own
$k \times k$ kernel, sums the results across channels, and adds one bias number. So:

- one filter: $C_{\text{in}} \cdot k^2$ weights $+ 1$ bias;
- the layer: $C_{\text{out}}$ such filters,

$$\text{params} = C_{\text{out}}\,(C_{\text{in}}\, k^2 + 1).$$

Worked count: a 3×3 conv from 32 channels to 64 channels has
$64 \times (32 \cdot 9 + 1) = 64 \times 289 = 18{,}496$ parameters — regardless of
whether the image is 28×28 or 4000×4000. Compare a dense layer between the same
feature maps at 28×28 resolution: $(32 \cdot 28^2) \times (64 \cdot 28^2) \approx
1.26 \times 10^9$ weights. The convolution is smaller by a factor of about 68,000.

Two ideas produce that saving, and it pays to name them separately:

- **Sparse connectivity**: each output value looks at only a $k \times k$ neighborhood,
  not the whole image.
- **Weight sharing**: the *same* kernel is used at every location. The shoe detector is
  learned once and applied everywhere.

## 5. Convolution is a constrained linear layer

A convolution is still a linear map — output values are weighted sums of input values —
so you could write it as one big matrix multiplying the flattened image. That matrix is
huge but almost entirely zeros (sparse connectivity), and the few nonzero entries
repeat the same $k^2$ numbers over and over, shifted one step per row (weight
sharing). On paper this week you will write this matrix out for a 3×3 kernel on a 5×5
input: a 9×25 matrix containing only the kernel's 9 distinct values.

So a conv layer is exactly a dense layer with two constraints bolted on: most weights
forced to zero, and the rest forced equal in a shifted pattern. Constraints reduce
capacity — a conv layer can represent fewer functions than the dense layer of the same
shape. Why is that a *good* trade? Week 09's bias–variance language answers it: the
constraint injects an assumption (patterns are local and position-independent) that is
true for images, so we spend our limited data estimating far fewer parameters and get a
lower-variance model without adding bias that matters.

The position-independence assumption has a name: **translation equivariance**. Shift
the input by one pixel and the feature map shifts by one pixel — the layer commutes
with translation. Formally, if $T$ shifts an image, a conv layer $f$ satisfies
$f(T(x)) = T(f(x))$. Detect-then-shift equals shift-then-detect.

## 6. Pooling and receptive fields

**Pooling** shrinks a feature map by summarizing small windows. Max pooling with a 2×2
window and stride 2 keeps the largest value in each 2×2 block, halving height and
width; average pooling takes the mean instead. Pooling has no parameters. It buys
three things: fewer values to process, a small amount of translation *invariance*
(a feature shifted by one pixel inside the window gives the same max), and — the
important one — a larger view for the layers above.

That "view" has a name. The **receptive field** of a unit is the patch of the original
input that can influence its value. A single 3×3 conv gives each output a 3×3
receptive field. Stack a second 3×3 conv: each of its outputs sees a 3×3 patch of the
first feature map, and each of *those* values saw a 3×3 input patch, so the second
layer's units see 5×5 of the input. Each extra 3×3 layer (stride 1) grows the
receptive field by 2: three stacked 3×3 convs see 7×7.

Why stack three 3×3s instead of using one 7×7? Count parameters per channel pair:
three 3×3 layers cost $3 \times 9\,C^2 = 27\,C^2$ weights ($C$ channels throughout),
one 7×7 layer costs $49\,C^2$. The stack is cheaper *and* has two extra nonlinearities
between the layers, so it computes a richer function. This is the argument the VGG
architecture is built on. Strides and pooling grow the receptive field much faster:
after a stride-2 stage, every kernel step above it covers 2 input pixels, so kernels
above pooling count double, quadruple, and so on.

## 7. From LeNet to ResNet

Four architectures, each solving the previous one's bottleneck:

- **LeNet-5 (1998).** The template: conv → pool → conv → pool → a few dense layers.
  Read digits on checks. Proof that learned kernels beat hand-designed features, at
  1998 scale.
- **AlexNet (2012).** LeNet's shape, scaled up (8 layers, 60M parameters), trained on
  GPUs with ReLU (Week 16's friend — no saturating tanh) and dropout (next week). Cut
  the ImageNet error nearly in half and started the deep-learning era. Bottleneck it
  exposed: architectures were ad hoc.
- **VGG (2014).** Radical uniformity: only 3×3 convs and 2×2 pools, 16–19 layers. The
  receptive-field argument of §6 made deep-and-thin the standard. Bottleneck it hit:
  making it *deeper* stopped helping.
- **ResNet (2015).** The fix for depth itself, and the subject of the next section:
  networks of 50, 101, 152 layers that train cleanly.

The problem ResNet solved is worth stating precisely, because it is counterintuitive.
Stacking more layers eventually made *training* error worse — not test error
(that would be overfitting), training error. This is called the **degradation
problem**. It cannot be a capacity problem: a 56-layer net could in principle imitate a
20-layer net exactly by making 36 layers compute the identity function (output = input).
It fails to because gradient descent cannot find that solution — a deep stack of
nonlinear layers initialized near zero is a hard place from which to *learn* the
identity, and (Week 16) signal and gradient statistics degrade layer by layer.

## 8. Skip connections, and why gradients survive them

A **residual block** wraps two or three conv layers, call them $F(x)$, and adds the
block's input back to its output:

$$y = x + F(x).$$

The added identity path is the **skip connection**. Now the layers inside $F$ no
longer have to learn the whole mapping — only the *residual*, the correction to the
identity. If the best thing a block can do is nothing, it just needs $F(x) \to 0$,
which gradient descent finds easily (weights near zero already give it).

That is the forward-pass story. The backward-pass story is why this week's derivation
matters. Take a loss $L$ computed downstream of $y$, and suppose we already know
$\partial L/\partial y$. Apply the chain rule (Week 13) to $y = x + F(x)$, treating
each component: $\partial y_i/\partial x_j = \delta_{ij} + \partial F_i/\partial x_j$,
where $\delta_{ij}$ is 1 when $i=j$ and 0 otherwise. In matrix form, with $J_F$ the
Jacobian of $F$ (the matrix of all partial derivatives of $F$'s outputs with respect
to its inputs, Week 13):

$$\frac{\partial L}{\partial x} \;=\; \frac{\partial L}{\partial y}\,(I + J_F)
\;=\; \underbrace{\frac{\partial L}{\partial y}}_{\text{unattenuated copy}}
\;+\; \frac{\partial L}{\partial y}\, J_F .$$

The gradient arriving at the block's input contains a *verbatim copy* of the gradient
at the block's output, plus a correction through the layers. Now chain $N$ blocks,
$x^{(i+1)} = x^{(i)} + F_i(x^{(i)})$:

$$\frac{\partial L}{\partial x^{(0)}}
= \frac{\partial L}{\partial x^{(N)}} \prod_{i=0}^{N-1}\bigl(I + J_{F_i}\bigr).$$

Multiply the product out and one of its $2^N$ terms is $I \cdot I \cdots I = I$: a
direct, unattenuated path from the loss to the very first layer. Compare a plain
(non-residual) network, whose gradient is $\prod_i J_{F_i}$ with no identity anywhere:
if each Jacobian shrinks its input a little — a typical layer at initialization has
$\lVert J \rVert < 1$ — the product shrinks *geometrically* with depth, e.g.
$0.9^{56} \approx 0.003$. Early layers of a deep plain net receive almost no learning
signal; early layers of a residual net always receive at least the identity term. This
is the sense in which people call skip connections "gradient highways".

Hold on to this derivation. In Week 19 the same product-of-Jacobians disease appears
in time instead of depth (recurrent networks), and the LSTM's fix is the same trick:
an additive path.

Two practical notes you will meet in code: when a block changes the number of channels
or the resolution, $x$ and $F(x)$ have different shapes, so the skip path gets a 1×1
convolution (a $k=1$ conv — per-pixel linear map across channels, cheap) to match
shapes. And ResNets put batch normalization (Week 16) inside every block; skips and
batchnorm together are what make 100+ layers routine.

## 9. Worked example: a LeNet-style CNN on FashionMNIST

FashionMNIST: 70,000 grayscale 28×28 images of clothing in 10 classes (T-shirt,
trouser, sneaker, ...), a drop-in replacement for MNIST that is not saturated at 99%.
`torchvision` downloads it for you (install with `uv add torchvision` if you have not).

First, the architecture arithmetic — do this table by hand for every network you ever
build. Formula from §3, parameter count from §4:

| layer | in shape | k | s | p | out shape | params |
|---|---|---|---|---|---|---|
| conv1 | 1×28×28 | 5 | 1 | 2 | 6×28×28 | $6(1\cdot25+1)=156$ |
| maxpool | 6×28×28 | 2 | 2 | 0 | 6×14×14 | 0 |
| conv2 | 6×14×14 | 5 | 1 | 0 | 16×10×10 | $16(6\cdot25+1)=2416$ |
| maxpool | 16×10×10 | 2 | 2 | 0 | 16×5×5 | 0 |
| flatten | 16×5×5 | | | | 400 | 0 |
| fc1 | 400 | | | | 120 | $400\cdot120+120=48120$ |
| fc2 | 120 | | | | 84 | $120\cdot84+84=10164$ |
| fc3 | 84 | | | | 10 | $84\cdot10+10=850$ |

Total: 61,706 parameters — check it against the model below with
`sum(p.numel() for p in model.parameters())`. Notice where the parameters live: the
convs do most of the *work* with 4% of the *weights*; the first dense layer is the hog.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(0)

to_tensor = transforms.ToTensor()
train_set = datasets.FashionMNIST("data", train=True, download=True, transform=to_tensor)
test_set = datasets.FashionMNIST("data", train=False, download=True, transform=to_tensor)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=256)

class LeNet(nn.Module):          # nn.Module subclass, exactly as in Week 15
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)   # in_ch, out_ch, kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)                # flatten all but batch dim
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = LeNet()
print("parameters:", sum(p.numel() for p in model.parameters()))   # 61706

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    for xb, yb in train_loader:
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
    print("epoch", epoch, "test acc", correct / len(test_set))
```

Five epochs of this reaches roughly 90% test accuracy on a laptop CPU in a few
minutes. For calibration: the same training budget on a same-size MLP lands around
87–88%, and the gap widens as you shrink the training set — the convolution's built-in
assumptions are doing real work. The exercises make you measure exactly this.

## Check yourself

1. A 3×3 conv, stride 1, padding 0, runs over a 32×32 input. Output size? Same
   question with padding 1, then with padding 1 and stride 2.
2. How many parameters does a 5×5 conv layer from 16 to 32 channels have (with
   biases)? Does the answer change for a 512×512 input instead of 32×32?
3. Name the two structural constraints that distinguish a conv layer from a dense
   layer, and the data assumption each encodes.
4. What is the receptive field of a unit after two stacked 3×3 convs (stride 1)?
   After 3×3 conv → 2×2 maxpool (stride 2) → 3×3 conv?
5. The degradation problem: what got worse as plain nets got deeper, and why does
   that rule out overfitting as the explanation?
6. For $y = x + F(x)$, write $\partial L/\partial x$ in terms of $\partial L/\partial y$
   and $J_F$. Which term survives even if $J_F = 0$?
7. Why is "translation equivariant" the right term for a conv layer rather than
   "translation invariant"? Which operation in a CNN adds a little invariance?
8. In the worked example, 78% of the parameters sit in `fc1`. What architectural
   change (seen in post-VGG nets) removes that dense bottleneck? (Hint: what could
   pool each feature map down to one number?)

## Answers

1. $(32-3)/1+1 = 30$. With $p=1$: $(32+2-3)/1+1 = 32$ (same padding). With $p=1,
   s=2$: $\lfloor 31/2 \rfloor + 1 = 16$.
2. $32\,(16 \cdot 25 + 1) = 12{,}832$. No — conv parameter counts are independent of
   input size; that is the point of weight sharing.
3. Sparse connectivity (patterns are local) and weight sharing (patterns mean the
   same thing everywhere, i.e. translation symmetry).
4. Two 3×3s: 5×5. Conv–pool–conv: the second conv's 3×3 window spans steps of 2 input
   pixels after the stride-2 pool, so $3 + (2-1) + (3-1)\cdot 2 = 8$; an 8×8 patch.
5. Training error itself got worse. Overfitting means low training error with high
   test error; here the optimizer could not even fit the training set, despite the
   deeper net containing the shallower one as a special case.
6. $\partial L/\partial x = \partial L/\partial y\,(I + J_F)$. The identity term:
   $\partial L/\partial y$ passes through untouched even if $J_F = 0$.
7. Shifting the input shifts the feature map — the output changes (equivariance), it
   just changes predictably. Pooling adds local invariance: small shifts within a
   pooling window leave the max unchanged.
8. Global average pooling: average each final feature map to a single number, then a
   small dense layer to the classes. ResNet does this; the giant flatten–dense layer
   disappears.

## New terms

- **kernel / filter** — the small learned weight grid slid over the input.
- **feature map** — one channel of a conv layer's output; the per-location response
  to one filter.
- **channel** — one 2D slice of a layer's input or output stack.
- **padding** — zeros added around the border before convolving.
- **stride** — step size of the sliding window.
- **sparse connectivity** — each output depends only on a local input patch.
- **weight sharing** — the same kernel weights reused at every location.
- **translation equivariance** — shift input ⇒ output shifts identically:
  $f(T(x)) = T(f(x))$.
- **pooling (max/average)** — parameter-free downsampling by window summary.
- **receptive field** — the input patch that can influence a given unit.
- **degradation problem** — deeper plain nets getting worse *training* error.
- **residual block / skip connection** — $y = x + F(x)$; the identity path around a
  block.
- **1×1 convolution** — per-pixel linear map across channels; used to match shapes on
  skip paths.
- **global average pooling** — collapsing each feature map to its mean; replaces the
  flatten–dense bottleneck.

## Going deeper

- CS231n course notes, "Convolutional Networks" — the classic shape-arithmetic
  treatment; do every calculation in it by hand against §3's formula.
- Prince, *Understanding Deep Learning* (free PDF), Ch. 10 (Convolutional networks)
  and Ch. 11 (Residual networks) — cleaner notation than most, and Ch. 11 extends §8's
  gradient argument.
- He et al., *Deep Residual Learning for Image Recognition* (arXiv:1512.03385) — read
  §1 and §3 for the degradation problem and the residual idea in the authors' own
  words; skim the results tables.
- Goodfellow, Bengio & Courville, *Deep Learning* (free online), Ch. 9 — optional; the
  formal treatment of equivariance if §5 left you wanting the general statement.
