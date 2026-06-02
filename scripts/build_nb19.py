#!/usr/bin/env python3
"""Build nb19_deep_learning_instructor.ipynb — full replication of the
10_deep_learning.qmd lecture deck + the PyTorch buildmodel lab.

Run: python scripts/build_nb19.py
Then generate the student copy with scripts/build_nb19_student (inline below).
"""
import json
from pathlib import Path

IMG = ("https://raw.githubusercontent.com/davi-moreira/"
       "2026Summer_predictive_analytics_purdue_MGMT474/main/notebooks/figures/")

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text})


def img(fname, width, alt="", caption=None):
    cap = f'\n<br>\n<small>{caption}</small>' if caption else ''
    a = f' alt="{alt}"' if alt else ''
    return (f'<center>\n<img src="{IMG}{fname}"{a} width="{width}"/>{cap}\n</center>')


def iframe(url, height=600):
    return (f'from IPython.display import IFrame\n'
            f'IFrame("{url}", width="100%", height={height})')


# ============================================================= HEADER
md(r"""# Special Topic — Deep Learning

<hr>

<center>
<div>
<img src="https://raw.githubusercontent.com/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/main/notebooks/figures/mgmt_474_ai_logo_02-modified.png" width="200"/>
</div>
</center>

# <center><a class="tocSkip"></center>
# <center>QM47400 Predictive Analytics</center>
# <center>Professor: Davi Moreira </center>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/nb19_deep_learning_student.ipynb)""")

# ============================================================= OBJECTIVES
md(r"""## Learning Objectives

By the end of this notebook, you will be able to:

1. Explain the historical arc that took neural networks from the 1980s rebrand into deep learning's 2010 resurgence, and name the three drivers that made it work (compute, data, frameworks).
2. Compare **PyTorch** and **TensorFlow** and say which one you would reach for and why.
3. Describe what a single neuron, a single hidden layer, and a multi-layer perceptron (MLP) compute, and read a single-layer-network diagram.
4. Explain how a neural network is **fit** — gradient descent, backpropagation, stochastic mini-batches — and the regularization tricks (dropout, data augmentation) that keep it from overfitting.
5. Distinguish the three structural inventions — fully-connected MLPs, **convolutional networks (CNNs)** for images, and **recurrent networks (RNNs)** for sequences — and name one problem class each is designed for.
6. **Build, inspect, and run a real neural network in PyTorch**, layer by layer, and use it to predict the label of a randomly chosen photo.
7. Decide whether deep learning is the right tool for a given business problem using a four-question rubric, and run one honest tabular comparison against gradient boosting.

> 📝 *This notebook's content is inspired by and replicates the material from [An Introduction to Statistical Learning (ISLP)](https://www.statlearning.com/) and the official [PyTorch "Learn the Basics" tutorials](https://pytorch.org/tutorials/beginner/basics/intro.html).*

> **📋 Participation Reminder:** This notebook contains **2 PAUSE-AND-DO exercises**. Complete both to receive participation credit.""")

# ============================================================= WHY THIS MATTERS
md(r"""## 💼 Why This Matters: The "What About AI?" Question Every Analyst Will Hear

The **VP of Strategy at TechCorp** sat in your Milestone 4 poster session, watched the gradient-boosting churn model demo, and asked one question:

> *"This is great. But shouldn't we be using deep learning?"*

This notebook is the answer you owe her. It does two things at once. First, it is a working analyst's **awareness module**: enough of the language and the structural ideas that you can recognize when deep learning **is** the right tool, recognize when it **is not** (most tabular business problems), and run a single fair comparison so the answer to her question is evidence-based, not vibes. Second — because awareness without hands-on contact is shallow — it walks you through **building and running a real neural network in PyTorch**, the same framework behind the LLMs you have used all course.

We follow the course's deep-learning lecture deck end to end: the history, the two big frameworks, the math of a single layer, how networks are fit, the CNN and RNN inventions, and a clear-eyed "when to use deep learning" rubric. Along the way you will embed the official PyTorch tutorial pages, define a network class, send a tensor through it, and read a prediction.""")

md(img("10_1_1-1.png", 480, "Deep learning pioneers",
       "The 2019 ACM Turing Award honored Yann LeCun, Geoffrey Hinton, and Yoshua Bengio for the work that became modern deep learning."))

# ============================================================= OVERVIEW
md(r"""## Overview

This notebook covers, in order:

| | |
|---|---|
| Deep Learning | Convolutional Neural Networks — CNN |
| PyTorch vs. TensorFlow | Document Classification |
| PyTorch (hands-on lab) | Recurrent Neural Networks — RNN |
| Neural Networks | RNN for Document Classification |
| Single Layer Neural Network | RNN for Time Series Forecasting |
| Fitting Neural Networks | When to Use Deep Learning |

Each block pairs a short conceptual explanation with the figure from the lecture deck. The PyTorch block is fully runnable — open it in Colab and execute the cells.""")

# ============================================================= DEEP LEARNING
md(r"""# 1. Deep Learning

Neural networks have lived two lives.

**Early rise (1980s).** Neural networks first gained popularity, with high levels of excitement and dedicated conferences (NeurIPS, Snowbird).

**1990s shift.** Other methods emerged — SVMs, Random Forests, Boosting — and neural networks receded into the background.

**Resurgence (2010).** Rebranded and refined under the banner of *Deep Learning*. By the 2020s it had become extremely successful and widely adopted.

**Key drivers of success:**

1. **Compute** — rapid increases in computing power (GPUs, parallel computing). GPUs designed for video games turned out to be exactly the right hardware for the matrix multiplications a network is built from, and a \$1,000 graphics card replaced a \$100,000 cluster.
2. **Data** — the availability of large-scale labeled datasets (ImageNet, with 14 million labeled images, gave networks enough training signal to outperform every classical method).
3. **Frameworks** — user-friendly deep-learning libraries (TensorFlow, PyTorch) cut the boilerplate from thousands of lines of low-level CUDA to a few lines of Python.

Much of the credit goes to three pioneers and their research teams — **Yann LeCun**, **Geoffrey Hinton**, and **Yoshua Bengio** — who received the 2019 ACM Turing Award for their work on neural networks.

> **A question that often comes up here:** *"If the math is mostly unchanged from the 1980s, why did it take 25 years?"* Because the math was *almost* unchanged. Three small but load-bearing additions — ReLU activations replacing sigmoid (which trains better in deep networks), dropout regularization, and batch normalization — combined with the new compute and data to make training networks with hundreds of layers practical for the first time. Each addition is a "small" idea, but together they crossed the threshold.""")

md(r"""## AI Visionaries: Interviews

Three short interviews with the pioneers — worth watching once to put faces and voices to the names.

<table><tr>
<td width="33%"><center>
<iframe width="100%" height="220" src="https://www.youtube.com/embed/Ah6nR8YAYF4" title="Yann LeCun" frameborder="0" allowfullscreen></iframe>
<a href="https://www.youtube.com/watch?v=Ah6nR8YAYF4" target="_blank"><b>Yann LeCun</b><br>The Future of AI</a>
</center></td>
<td width="33%"><center>
<iframe width="100%" height="220" src="https://www.youtube.com/embed/qrvK_KuIeJk" title="Geoffrey Hinton" frameborder="0" allowfullscreen></iframe>
<a href="https://www.youtube.com/watch?v=qrvK_KuIeJk" target="_blank"><b>Geoffrey Hinton</b><br>60 Minutes Interview</a>
</center></td>
<td width="33%"><center>
<iframe width="100%" height="220" src="https://www.youtube.com/embed/5LgDUqCbBwo" title="Yoshua Bengio" frameborder="0" allowfullscreen></iframe>
<a href="https://www.youtube.com/watch?v=5LgDUqCbBwo" target="_blank"><b>Yoshua Bengio</b><br>Path to Human-Level AI</a>
</center></td>
</tr></table>""")

# ============================================================= PYTORCH vs TENSORFLOW
md(r"""# 2. PyTorch vs. TensorFlow""")

md(img("pytorch_vs_tensorflow.png", 720, "PyTorch vs TensorFlow"))

md(r"""## What Are Deep Learning Frameworks?

Deep learning frameworks are software libraries designed to streamline the creation, training, and deployment of neural networks. They reduce boilerplate code, handle tensor operations efficiently, and make it easier to prototype and iterate on new architectures.

- They provide **pre-built functions**, **automatic differentiation**, and **GPU/TPU** support.
- **Why they are necessary:** they let researchers and developers focus on **model design** rather than low-level implementation details.

**What both give you (and why neither is just `numpy`):**

- **Tensor operations.** A tensor is a multi-dimensional array — `numpy.ndarray` is a 1- or 2-D version; tensor frameworks scale to 4-D image batches and 5-D video batches without breaking a sweat.
- **Automatic differentiation.** When you write `loss = something(weights)`, the framework remembers the chain of operations and computes `dloss/dweights` for you with one call (`loss.backward()` in PyTorch). That is the magic that makes training a network with millions of parameters tractable.
- **GPU support.** Move a tensor to the GPU with `.to('cuda')` (PyTorch) and every subsequent operation runs there. The same code on CPU "just" runs \~50× slower.
- **Pre-built layers.** Convolution, recurrence, attention — none need to be implemented from scratch.""")

md(r"""## PyTorch and TensorFlow

### What is PyTorch?

- Developed primarily by **Facebook (Meta)** and released in September 2016.
- Emphasizes a **dynamic computation graph** (eager execution).
- Highly **"Pythonic"**: feels natural for Python developers.
- Strong community presence in **academia** and research.
- Active ecosystem: **torchvision**, **torchaudio**, and more.
- Get started: <https://pytorch.org/tutorials/beginner/basics/intro.html>

### What is TensorFlow?

- Developed primarily by **Google** and released in November 2015.
- Historically used a **static graph** approach (an "eager mode" was added later).
- Comes with **extensive tools** for deployment (mobile, web, production).
- Large ecosystem with well-integrated components (**TensorBoard**, **TFX**, **TensorFlow Lite**).
- Get started: <https://www.tensorflow.org/tutorials>""")

md(r"""## Key Differences

| Aspect | **PyTorch** | **TensorFlow** |
|:---|:---|:---|
| **Computation Graph** | Dynamic graph (eager execution by default). | Historically static graph with a build-and-execute phase (now supports eager execution). |
| **Debugging & Development Style** | Straightforward for Python developers, immediate error feedback. | Trickier to debug in graph mode; eager mode helps but is newer. |
| **Deployment & Production** | TorchServe and growing enterprise support. | TensorFlow Serving, TensorFlow Lite, easy Google Cloud integration. |

## Similarities

| Similarity | Description |
|:---|:---|
| **Wide range of layers** | Convolutional, recurrent, transformers — both maintain robust libraries for standard and advanced layers. |
| **Auto-differentiation** | No need to manually compute gradients; backpropagation is handled automatically. |
| **GPU acceleration** | Both leverage CUDA (NVIDIA GPUs) or other backends to speed up training. |
| **Rich communities** | Abundant tutorials, example code, pretrained models, and Q&A forums. |""")

md(r"""## Which Should You Choose?

### Choose **PyTorch** if:

- Your focus is **rapid experimentation** and **academic research**.
- You prioritize a **Pythonic workflow** and easy debugging.
- You prefer a **dynamic graph** approach.
- You value seamless interaction with Python libraries.

### Choose **TensorFlow** if:

- You need **robust production** and **deployment pipelines**.
- You plan to integrate with **Google Cloud** services.
- You require support for **mobile/edge devices** (TensorFlow Lite).
- You want an **end-to-end ecosystem** (TFX, TensorBoard, Serving).

> **A question that often comes up here:** *"If I can only learn one, which one?"* For a working business analyst, pick **PyTorch**. Research code on GitHub, the Hugging Face model hub, and the modern LLM ecosystem are all PyTorch-first. TensorFlow is still common in production at Google-scale companies, but the gap closes every year — and the rest of this notebook is a PyTorch lab for exactly this reason.""")

# ============================================================= PYTORCH LAB
md(r"""# 3. PyTorch — Hands-On

The slides for this section embed the official PyTorch *"Learn the Basics"* tutorial pages so you can read them in place, then we **run** the network-building tutorial ourselves. The cells below are runnable in Colab.

> 💡 **Pro tip:** Use a GPU runtime to speed things up — *Runtime > Change runtime type > GPU*. Everything here also runs on CPU in well under a minute.

Colab already ships with `torch` and `torchvision`. If you ever need them locally, uncomment the install line in the next cell.""")

code(r"""# Colab already has PyTorch installed. Locally, uncomment:
# !pip install torch torchvision -q

import torch
print("PyTorch version:", torch.__version__)""")

md(r"""## [Tensors in PyTorch](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)

Tensors are PyTorch's core data structure — multi-dimensional arrays that can live on a GPU and remember the operations applied to them (for automatic differentiation). The official page is embedded below.""")
code(iframe("https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html"))

md(r"""## [Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

`Dataset` stores the samples and their labels; `DataLoader` wraps an iterable around the `Dataset` to serve mini-batches during training. We will use the **FashionMNIST** dataset — 28×28 grayscale images of clothing in 10 classes.""")
code(iframe("https://pytorch.org/tutorials/beginner/basics/data_tutorial.html"))

md(r"""**Let's load FashionMNIST ourselves** so the snippets below have real data to work with. The first run downloads \~30 MB.""")
code(r"""from torchvision import datasets
from torchvision.transforms import ToTensor

# Training and test sets, with images converted to tensors scaled to [0, 1].
training_data = datasets.FashionMNIST(root="data", train=True,  download=True, transform=ToTensor())
test_data     = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

# Human-readable class names for FashionMNIST.
labels_map = {0:"T-Shirt", 1:"Trouser", 2:"Pullover", 3:"Dress", 4:"Coat",
              5:"Sandal", 6:"Shirt", 7:"Sneaker", 8:"Bag", 9:"Ankle Boot"}

print("Training images:", len(training_data), "| Test images:", len(test_data))""")

md(r"""**Inspecting a single image-tensor.** Indexing the dataset returns `(image, label)` with the transform already applied. Set `idx` to any integer in `range(len(training_data))` to see a different image.""")
code(r"""import matplotlib.pyplot as plt

idx = 0  # change as desired
image_tensor, label = training_data[idx]   # image_tensor is a 1x28x28 tensor

print("Shape :", image_tensor.shape)        # torch.Size([1, 28, 28])
print("Label :", label, "->", labels_map[label])
print("Tensor (first 5 rows):\n", image_tensor[0, :5, :])

plt.imshow(image_tensor.squeeze(), cmap="gray")
plt.title(f"FashionMNIST class {label} ({labels_map[label]})")
plt.axis("off")
plt.show()""")

md(r"""## [Transforms](https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html)

Transforms manipulate the data into the form a model expects — here, `ToTensor()` turns a PIL image into a `[0,1]`-scaled tensor. The official page is embedded below.""")
code(iframe("https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html"))

md(r"""## [Build the Neural Network](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)

Neural networks are made of layers/modules that perform operations on data. The `torch.nn` namespace provides all the building blocks you need. Every module in PyTorch subclasses `nn.Module`. A neural network is itself a module that consists of other modules (layers) — this nested structure lets you build complex architectures easily.

The official page is embedded below; then **we build the same network ourselves**, cell by cell.""")
code(iframe("https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html"))

md(r"""### Get the device for training

We want to train on an accelerator (CUDA, MPS, ...) if one is available; otherwise we use the CPU.""")
code(r"""import os
from torch import nn

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")""")

md(r"""### Define the class

We define our neural network by subclassing `nn.Module`, and initialize the layers in `__init__`. Every `nn.Module` subclass implements the operations on input data in the `forward` method.

**What are we doing?** We are defining a network for 28×28 grayscale images (like FashionMNIST). The network outputs **10 values**, one per class.

- `class NeuralNetwork(nn.Module):` — a new class that inherits from PyTorch's base class for all models.
- `super().__init__()` — runs the parent's initialization so PyTorch can track everything inside the model.
- `self.flatten = nn.Flatten()` — turns a **2D image (28×28)** into a **1D vector (784 values)** that linear layers can handle.
- `self.linear_relu_stack` — the main body: **three fully-connected (`Linear`) layers with ReLU activations in between**.
    1. `nn.Linear(28*28, 512)` — takes the 784 pixel values and projects them into 512 new values. A `Linear(784, 512)` layer computes `output = x · W + b`, where `W` has shape `[784 × 512]` and `b` has length 512. Each of the 512 outputs is a linear combination of all 784 inputs.
    2. `nn.ReLU()` — keeps positive numbers, turns negatives into zero; adds the non-linearity.
    3. `nn.Linear(512, 512)` — a hidden layer that helps the model learn more complex patterns.
    4. `nn.ReLU()` — another non-linear transformation.
    5. `nn.Linear(512, 10)` — produces **10 output values** called **logits**, one per class.
- `forward(self, x)` — the forward pass: flatten the image, push it through the stack, return the logits.""")
code(r"""class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits""")

md(r"""We create an instance of `NeuralNetwork`, move it to the `device`, and print its structure.""")
code(r"""model = NeuralNetwork().to(device)
print(model)""")

md(r"""### Use the model on input data

To use the model, we pass it the input data, which runs the model's `forward`. **Do not** call `model.forward()` directly — PyTorch manages hooks and gradients when you call `model(X)`.

Calling the model returns a 2-D tensor: `dim=0` indexes the batch, `dim=1` holds the 10 raw values per class. We get probabilities by passing the logits through `nn.Softmax`, and the predicted class with `argmax`.

- `torch.rand(1, 28, 28, device=device)` — a **random image** with shape `[1, 28, 28]`: batch size 1, 28×28 pixels, placed on the model's device.
- `pred_probab = nn.Softmax(dim=1)(logits)` — softmax turns logits into probabilities (values in `[0,1]` that sum to 1); `dim=1` applies it across the 10 class values.
- `y_pred = pred_probab.argmax(1)` — the index of the largest probability, i.e. the predicted class.""")
code(r"""X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# Visualize the random input image (remove the batch dim: (1,28,28) -> (28,28))
image = X[0].cpu()
plt.imshow(image, cmap="gray")
plt.title("Random 28x28 Image")
plt.axis("off")
plt.show()""")

md(r"""### Model layers, broken down

Let's break down the layers using a sample mini-batch of **3 images** of size 28×28 and watch what happens as it flows through the network.""")
code(r"""input_image = torch.rand(3, 28, 28)
print(input_image.size())""")

md(r"""**`nn.Flatten`** converts each 2D 28×28 image into a contiguous array of 784 pixel values (the batch dimension at `dim=0` is maintained).""")
code(r"""flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())""")

md(r"""**`nn.Linear`** applies a linear transformation using its stored weights and biases.""")
code(r"""layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())""")

md(r"""**`nn.ReLU`** introduces non-linearity, helping the network learn a wide variety of phenomena. It is applied between linear layers.""")
code(r"""print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")""")

md(r"""**`nn.Sequential`** is an ordered container of modules; data passes through them in order. Handy for assembling a quick network.""")
code(r"""seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)
print(logits.size())""")

md(r"""**`nn.Softmax`** scales the last layer's logits (raw values in `[-∞, ∞]`) to `[0, 1]` so they represent predicted probabilities. `dim` indicates the dimension along which the values must sum to 1.""")
code(r"""softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
print(pred_probab)""")

md(r"""**Model parameters.** Many layers are *parameterized* — they have weights and biases optimized during training. Subclassing `nn.Module` tracks all fields automatically and exposes them via `parameters()` / `named_parameters()`.""")
code(r"""print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")""")

# ---- RANDOM PHOTO PREDICTION (requirement #4 add-on) ----
md(r"""### 🎲 Randomly select a photo and read its predicted label

Now let's close the loop: pull a **random photo** from the FashionMNIST test set, run it through the model, and display the image with the label the network predicts. Re-run the cell to draw a different photo each time.""")
code(r"""# Pick a random photo from the test set
rand_idx = torch.randint(len(test_data), size=(1,)).item()
photo, true_label = test_data[rand_idx]   # photo: 1x28x28 tensor

# Run the model in evaluation mode (no gradient tracking needed for prediction)
model.eval()
with torch.no_grad():
    logits_photo = model(photo.unsqueeze(0).to(device))   # add batch dim -> 1x1x28x28
    probs = nn.Softmax(dim=1)(logits_photo)
    pred_label = probs.argmax(1).item()
    confidence = probs[0, pred_label].item()

# Show the photo with predicted vs. actual label
plt.imshow(photo.squeeze(), cmap="gray")
plt.title(f"Predicted: {labels_map[pred_label]} ({confidence:.0%})  |  Actual: {labels_map[true_label]}")
plt.axis("off")
plt.show()

print(f"Random photo #{rand_idx}")
print(f"Predicted class: {pred_label} -> {labels_map[pred_label]}")
print(f"Actual class   : {true_label} -> {labels_map[true_label]}")""")

md(r"""> **A question that often comes up here:** *"Why is the prediction usually wrong?"* Because this model has **not been trained yet** — its weights are still the random values PyTorch assigned at initialization, so it is guessing. The point of this cell is the **mechanics**: select a photo → add a batch dimension → forward pass → softmax → `argmax` → label. Training the weights so the predictions become accurate is exactly what the **Optimizing Model Parameters** tutorial (embedded below) covers; after that optimization loop, this same cell would land the correct label the large majority of the time.""")

md(r"""## [Automatic Differentiation with `torch.autograd`](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)

`torch.autograd` is PyTorch's automatic differentiation engine — it records operations and computes the gradients that gradient descent needs. The official page is embedded below.""")
code(iframe("https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html"))

md(r"""## [Optimizing Model Parameters](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)

This is the training loop: pick a loss function and optimizer, then iterate over mini-batches, calling `loss.backward()` and `optimizer.step()`. Run this tutorial and the random-photo cell above will start predicting correctly.""")
code(iframe("https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html"))

md(r"""## [Save and Load the Model](https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html)

How to persist a trained model's weights (`state_dict`) to disk and reload them for inference later.""")
code(iframe("https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html"))

md(r"""## Your turn! [Introduction to PyTorch — YouTube Series](https://pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html)

If you want to go deeper, the official YouTube series walks through the same material with narration. Embedded below.""")
code(iframe("https://pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html"))

# ============================================================= NEURAL NETWORKS
md(r"""# 4. Neural Networks

Before the math, watch this 19-minute visual introduction — the single best intuition-builder for what a neural network actually *is*.

<center>
<iframe width="800" height="450" src="https://www.youtube.com/embed/aircAruvnKk" title="But what is a neural network?" frameborder="0" allowfullscreen></iframe>
<br>
<a href="https://www.youtube.com/watch?v=aircAruvnKk" target="_blank">But what is a neural network? (3Blue1Brown)</a>
</center>""")

# ============================================================= SINGLE LAYER NN
md(r"""# 5. Single Layer Neural Network

A single-layer network with $K$ hidden units computes:

$$
\begin{align*}
f(X) &= \beta_0 + \sum_{k=1}^{K} \beta_k h_k(X) \\
     &= \beta_0 + \sum_{k=1}^{K} \beta_k\, g\!\left(w_{k0} + \sum_{j=1}^{p} w_{kj} X_j \right).
\end{align*}
$$""")

md(img("10_1-1.png", 560, "Single layer neural network diagram",
       "Network diagram of a single-layer neural network."))

md(r"""**Reading the diagram.** Neural networks are often drawn as **network diagrams**:

- **Input layer (orange circles):** $X_1, X_2, X_3, X_4$ — observed features from the dataset.
- **Hidden layer (blue circles):** $A_1, \ldots, A_5$ — transformations (activations) computed from the inputs.
- **Output layer (pink circle):** $f(X) \to Y$ — also observed (a label or continuous response).

**Observed vs. latent.** The $X_j$ are observed (inputs) and $Y$ is observed (the response). The **hidden units** $A_k$ are **not** observed — they are learned transformations.

**The hidden layer as transformations.** Each activation is

$$
A_k = h_k(X) = g\!\Bigl(w_{k0} + \sum_{j=1}^{p} w_{kj} X_j\Bigr),
$$

where $g(\cdot)$ is a non-linear function (ReLU, sigmoid, tanh) and the $w_{kj}$ are weights learned during training. Each hidden unit has a *different* set of weights, hence a different transformation.

**Training.** The network learns all the weights $w_{kj}, w_{k0}, \beta_k, \beta_0$ during training, with the goal of predicting $Y$ from $X$ accurately. The key insight: the hidden layer learns useful transformations on the fly to help approximate the true mapping from $X$ to $Y$.""")

md(r"""## Single Layer Neural Network: Details""")
md(img("10_2-1.png", 620, "Activation function details"))
md(r"""- $A_k = h_k(X) = g\!\left(w_{k0} + \sum_{j=1}^{p} w_{kj} X_j\right)$ are the **activations** in the *hidden layer* — a non-linear transformation of a linear function.
- $g(z)$ is the **activation function**. Two popular choices: the **sigmoid** and the **rectified linear unit (ReLU)**.
- Activation functions in hidden layers are typically non-linear; otherwise the whole model collapses to a linear model.
- So activations are like **derived features** — non-linear transformations of linear combinations of the inputs.
- For regression, the model is fit by minimizing $\sum_{i=1}^{n} (y_i - f(x_i))^2$.""")

md(r"""## NN Example: MNIST Digits""")
md(img("10_3a-1.png", 360, "MNIST digit examples"))
md(img("10_3b-1.png", 420, "MNIST two-layer network"))
md(r"""- **Handwritten digits:** 28×28 grayscale images, 60K train / 10K test. Features are the 784 pixel grayscale values $\in (0, 255)$; labels are the digit class $0\text{–}9$.
- **Goal:** build a classifier to predict the image class.
- A two-layer network with **256 units** at the first layer, **128 units** at the second, and **10 units** at the output layer has — along with the biases — **235,146 parameters** (the weights).

> **A question that often comes up here:** *"235,146 parameters for a tiny 28×28 image — isn't that wildly overfit?"* It would be for a noisy tabular problem with a few thousand rows. But image recognition has a very **high signal-to-noise ratio** and a huge training set, so the network has enough data to pin those parameters down. This is the same theme you will see again under "double descent" — over-parameterization is far less dangerous when the signal is strong and the data is large.""")

# ============================================================= FITTING NN
md(r"""# 6. Fitting Neural Networks

Two short videos build the intuition for *how* a network learns, then we make it precise.

<center>
<iframe width="760" height="430" src="https://www.youtube.com/embed/IHZwWFHWa-w" title="Gradient descent" frameborder="0" allowfullscreen></iframe>
<br><a href="https://www.youtube.com/watch?v=IHZwWFHWa-w" target="_blank">Gradient descent, how neural networks learn (3Blue1Brown)</a>
<br><br>
<iframe width="760" height="430" src="https://www.youtube.com/embed/Ilg3gGewQ5U" title="Backpropagation" frameborder="0" allowfullscreen></iframe>
<br><a href="https://www.youtube.com/watch?v=Ilg3gGewQ5U" target="_blank">Backpropagation, intuitively (3Blue1Brown)</a>
</center>""")

md(r"""## The optimization problem

We fit the network by minimizing squared error:

$$
\min_{\{w_k\}_{1}^K,\, \beta}\; \frac{1}{2} \sum_{i=1}^n \bigl(y_i - f(x_i)\bigr)^2,
\qquad
f(x_i) = \beta_0 + \sum_{k=1}^K \beta_k\, g\!\left(w_{k0} + \sum_{j=1}^p w_{kj} x_{ij}\right).
$$

This problem is hard because the objective is **non-convex**. Despite that, effective algorithms have evolved that optimize complex neural networks efficiently.

## Non-convex functions and gradient descent

Let $R(\theta) = \frac{1}{2}\sum_{i=1}^n (y_i - f_\theta(x_i))^2$ with $\theta = (\{w_k\}_{1}^K, \beta)$.""")
md(img("10_17-1.png", 560, "Gradient descent on a non-convex surface"))
md(r"""1. Start with a guess $\theta^0$ and set $t = 0$.
2. Iterate until $R(\theta)$ stops decreasing:
    - (a) Find a small change $\delta$ so that $\theta^{t+1} = \theta^t + \delta$ **reduces** the objective.
    - (b) Set $t \gets t + 1$.

**How do we find a downhill direction $\delta$?** We compute the **gradient vector**

$$
\nabla R(\theta^t) = \frac{\partial R(\theta)}{\partial \theta}\bigg|_{\theta = \theta^t},
$$

the vector of partial derivatives at the current guess. The gradient points *uphill*, so our update steps the other way:

$$
\theta^{t+1} \gets \theta^t - \rho\, \nabla R(\theta^t),
$$

where $\rho$ is the **learning rate** (typically small, e.g. $\rho = 0.001$). If we had started a little to the left, we might have descended into a **local** rather than the global minimum — in high dimensions it is hard to tell which.""")

md(r"""## Gradients and backpropagation

Because $R(\theta) = \sum_{i=1}^n R_i(\theta)$ is a sum, its gradient is the sum of gradients. With $z_{ik} = w_{k0} + \sum_{j=1}^p w_{kj} x_{ij}$, backpropagation applies the **chain rule**:

$$
\frac{\partial R_i(\theta)}{\partial \beta_k}
= -(y_i - f_\theta(x_i)) \cdot g(z_{ik}),
$$

$$
\frac{\partial R_i(\theta)}{\partial w_{kj}}
= -(y_i - f_\theta(x_i)) \cdot \beta_k \cdot g'(z_{ik}) \cdot x_{ij}.
$$

This is exactly the calculation `loss.backward()` runs for you in PyTorch.""")

md(r"""## Tricks of the trade

- **Slow learning.** Gradient descent is slow, and a small $\rho$ slows it further. With **early stopping**, this slowness becomes a form of regularization.
- **Stochastic gradient descent (SGD).** Rather than use *all* the data each step, use a small **mini-batch** drawn at random. For MNIST ($n = 60{,}000$) we might use mini-batches of 128.
- **Epoch.** One epoch is the number of mini-batch updates needed to process $n$ samples once — e.g. $60{,}000/128 \approx 469$ updates for MNIST.
- **Regularization.** Ridge and lasso shrink the weights at each layer; two other popular forms are **dropout** and **augmentation**, next.""")

md(r"""## Dropout learning""")
md(img("10_1_4-1.png", 560, "Dropout"))
md(r"""- At each SGD update, randomly **remove** units with probability $\phi$, and scale up the weights of those retained by $1/(1-\phi)$ to compensate.
- In simple cases (linear regression) this is equivalent to **ridge** regularization.
- As in ridge, the other units *stand in* for those temporarily removed, and their weights are drawn closer together.
- Similar in spirit to randomly omitting variables when growing trees in random forests.

## Ridge and data augmentation""")
md(img("10_1_5-1.png", 560, "Data augmentation as ridge"))
md(r"""- Make many copies of each $(x_i, y_i)$ and add a small amount of Gaussian noise to the $x_i$ — a little cloud around each observation — but leave the copies of $y_i$ alone.
- This makes the fit robust to small perturbations in $x_i$, and is equivalent to ridge regularization in an OLS setting.

## Data augmentation on the fly""")
md(img("10_1_6-1.png", 560, "Image data augmentation"))
md(r"""- Data augmentation is especially effective with SGD, here for a CNN and image classification.
- Natural transformations are made of each training image as it is sampled, creating a cloud of images around each original.
- The label is left unchanged — in each case still **tiger**.
- Improves CNN performance and is similar to ridge.""")

md(r"""## Double descent""")
md(r"""- With neural networks, it seems better to have **too many** hidden units than too few — and more hidden layers better than few.
- Running SGD until *zero* training error often gives good out-of-sample error; adding even more units and again training to zero error sometimes gives **even better** out-of-sample error.
- What happened to overfitting and the usual bias-variance trade-off?""")
md(img("10_20-1.png", 560, "The double-descent error curve"))
md(r"""**The double-descent error curve.** When $d \le 20$ the model is OLS and we see the usual bias-variance trade-off. When $d > 20$ we revert to the minimum-norm solution; as $d$ increases above 20, $\sum_{j=1}^d \hat\beta_j^2$ **decreases** (it is easier to achieve zero error), giving less wiggly solutions.""")
md(img("10_21-1.png", 560, "Less wiggly solutions for larger d"))
md(r"""- Achieving a zero-residual solution with $d = 20$ is a real stretch; it is easier for larger $d$.
- By analogy, deep and wide networks fit by SGD down to zero training error often generalize well — *especially* in high signal-to-noise problems like image recognition, where the zero-error solution is mostly signal.""")

# ============================================================= CNN
md(r"""# 7. Convolutional Neural Network — CNN

Neural networks rebounded around 2010 with big successes in **image classification**, just as massive databases of labeled images were being accumulated.

## The CIFAR-100 database""")
md(img("cifar100.png", 720, "CIFAR-100 sample images"))
md(r"""- The figure shows 75 images from the **CIFAR-100** database.
- It consists of 60,000 images labeled into 20 superclasses (e.g. *aquatic mammals*), with five classes per superclass (*beaver, dolphin, otter, seal, whale*).
- Each image is 32×32 pixels, with three 8-bit numbers per pixel (red, green, blue). The numbers are organized in a 3-D array called a **feature map**: the first two axes are spatial (both 32-dimensional), and the third is the **channel** axis.
- There is a designated training set of 50,000 images and a test set of 10,000.""")

md(r"""## The convolutional network hierarchy""")
md(img("10_1_2-1.png", 560, "CNN feature hierarchy"))
md(r"""- CNNs mimic, to a degree, how humans classify images — recognizing specific features or patterns anywhere in the image.
- The network first identifies **low-level features** (small edges, patches of color), then combines them into **higher-level features** (parts of ears or eyes), and finally into class probabilities.
- This hierarchy is built from two specialized hidden-layer types: **convolution layers** (search for instances of small patterns) and **pooling layers** (downsample to keep a prominent subset).
- State-of-the-art architectures stack many convolution and pooling layers.""")

md(r"""## Convolution layer

A convolution layer is made of many **convolution filters**, each a template that detects whether a particular local feature is present. The convolution operation repeatedly multiplies matching matrix elements and sums them:

$$
\text{Input} =
\begin{bmatrix} a & b & c \\ d & e & f \\ g & h & i \\ j & k & l \end{bmatrix},
\quad
\text{Filter} =
\begin{bmatrix} \alpha & \beta \\ \gamma & \delta \end{bmatrix}
\;\Rightarrow\;
\text{Convolved} =
\begin{bmatrix}
a\alpha + b\beta + d\gamma + e\delta & b\alpha + c\beta + e\gamma + f\delta \\
d\alpha + e\beta + g\gamma + h\delta & e\alpha + f\beta + h\gamma + i\delta \\
g\alpha + h\beta + j\gamma + k\delta & h\alpha + i\beta + k\gamma + l\delta
\end{bmatrix}.
$$

- The filter is applied to every 2×2 submatrix of the image.
- If a submatrix resembles the filter, it produces a large value — so the convolved image **highlights regions that resemble the filter**.
- The filter is itself a small image (an edge, a shape), and the filters are **learned** during training.""")
md(r"""## Convolution example""")
md(img("10_7-1.png", 640, "Convolution with vertical/horizontal filters"))
md(r"""- Convolution finds common patterns that occur in different parts of the image.
- The two filters shown highlight vertical and horizontal stripes.
- Since images have three color channels, each filter does too: one filter per channel, with dot-products summed.
- The weights in the filters are **learned** by the network.""")

md(r"""## Pooling layer

A pooling layer condenses a large image into a smaller summary image:

$$
\text{Max pool}
\begin{bmatrix}
1 & 2 & 5 & 3 \\ 3 & 0 & 1 & 2 \\ 2 & 1 & 3 & 4 \\ 1 & 1 & 2 & 0
\end{bmatrix}
\rightarrow
\begin{bmatrix} 3 & 5 \\ 2 & 4 \end{bmatrix}
$$

- Each non-overlapping 2×2 block is replaced by its maximum.
- This sharpens feature identification, allows **locational invariance**, and reduces the dimension by a factor of 4.

## Architecture of a CNN""")
md(img("10_8-1.png", 640, "Full CNN architecture"))
md(r"""- Many convolve + pool layers.
- Filters are typically small, e.g. 3×3 per channel.
- Each filter creates a new channel in the convolution layer.
- As pooling reduces size, the number of filters/channels is typically increased.
- The number of layers can be very large — **resnet50**, trained on **imagenet** (1000 classes), has 50 layers.""")

md(r"""## CNN example: pretrained networks to classify images""")
md(img("10_1_3-1.png", 460, "resnet50 classifying photographs"))
md(r"""Here a 50-layer **resnet50** network trained on the 1000-class **imagenet** corpus classifies photographs. This is the everyday way analysts use CNNs: **download a pretrained network** and apply it, rather than train one from scratch. The `book_images/` folder in the lecture lab (cats, hawks, flamingos) is exactly the kind of input you would feed such a model.""")

# ============================================================= DOC CLASSIFICATION
md(r"""# 8. Document Classification

The **IMDB** corpus consists of user movie reviews, each labeled for **sentiment** (positive or negative). Here is the start of a negative review:

> *This has to be one of the worst films of the 1990s. When my friends & I were watching this film ... we just sat & watched the first half an hour with our jaws touching the floor at how bad it really was...*

There are labeled training and test sets, each with 25,000 reviews, balanced on sentiment. **Goal:** predict the sentiment of a review.

## Featurization: bag-of-words

Documents have different lengths and are sequences of words. How do we build features $X$?

- From a dictionary, take the 10K most frequent words.
- Represent each document as a binary vector of length $p = 10\text{K}$, scoring a 1 in every position whose word occurs.
- With $n$ documents, we get an $n \times p$ **sparse** feature matrix $\mathbf{X}$.
- Bag-of-words are **unigrams**; we can instead use **bigrams** (adjacent word pairs) or, in general, **m-grams**.""")
md(img("10_11-1.png", 380, "Lasso vs neural network on IMDB"))
md(r"""**Lasso vs. neural network.** On these IMDB reviews, a simpler **lasso logistic regression** works as well as a two-hidden-layer neural network — an early hint of the "when to use deep learning" lesson at the end of this notebook.""")

# ============================================================= RNN
md(r"""# 9. Recurrent Neural Networks — RNN

Often data arrive as **sequences**:

- **Documents** are sequences of words, whose relative positions carry meaning.
- **Time series** such as weather data or financial indices.
- **Recorded speech or music.**

RNNs take this sequential nature into account and build a **memory** of the past. The feature for each observation is a **sequence** of vectors $X = \{X_1, X_2, \ldots, X_L\}$; the target $Y$ is often a single variable (e.g. *Sentiment*) but can also be a sequence (e.g. the same document in another language).

## Simple RNN architecture""")
md(img("10_12-1.png", 560, "Simple RNN architecture"))
md(r"""- The hidden layer is a sequence of vectors $A_\ell$, each receiving input $X_\ell$ and the previous hidden state $A_{\ell-1}$, and producing an output $O_\ell$.
- The **same** weights $\mathbf{W}, \mathbf{U}, \mathbf{B}$ are used at every step — hence **recurrent**.
- The $A_\ell$ sequence is an evolving model for the response, updated as each $X_\ell$ is processed.

## RNN in detail

With $X_\ell = (X_{\ell 1}, \ldots, X_{\ell p})$ and $A_\ell = (A_{\ell 1}, \ldots, A_{\ell K})$, the $k$-th hidden component is

$$
A_{\ell k} = g\!\left(w_{k0} + \sum_{j=1}^{p} w_{kj} X_{\ell j} + \sum_{s=1}^{K} u_{ks} A_{\ell-1,s}\right),
\qquad
O_\ell = \beta_0 + \sum_{k=1}^{K} \beta_k A_{\ell k}.
$$

Often we care only about the final prediction $O_L$. For squared-error loss over $n$ sequence/response pairs we minimize $\sum_{i=1}^{n} (y_i - o_{iL})^2$.

**Three variants you will hear named:**

- **Vanilla RNN** — the original; trains poorly on long sequences because gradients vanish.
- **LSTM** (Long Short-Term Memory) — adds gates that let the network learn what to remember and forget. The default for sequence problems from \~2014 to \~2018.
- **Transformer** — replaces recurrence with attention, processes the whole sequence in parallel, and now dominates language and increasingly time series. ChatGPT, Claude, and every modern LLM are transformers.""")

md(r"""# 10. RNN for Document Classification

- A document is a sequence of words $\{\mathcal{W}_\ell\}_{1}^{L}$. We truncate/pad to a common length $L$ (here $L = 500$).
- Each word is a **one-hot** binary vector $X_\ell$ of length 10K — extremely sparse, and it would not work well directly.
- Instead we use a lower-dimensional pretrained **word embedding** matrix $\mathbf{E}$ ($m \times 10\text{K}$), reducing each length-10K binary vector to a real vector of dimension $m \ll 10\text{K}$ (e.g. in the low hundreds).""")
md(img("10_13a-1.png", 500, "Word embedding (one-hot)"))
md(img("10_13b-1.png", 500, "Word embedding (dense)"))
md(r"""Embeddings are pretrained on very large corpora using methods similar to principal components; **word2vec** and **GloVe** are popular. This embedding idea is the direct ancestor of the token embeddings inside today's LLMs.""")

# ============================================================= RNN TIME SERIES
md(r"""# 11. RNN for Time Series Forecasting""")
md(img("10_14-1.png", 720, "New York Stock Exchange data"))
md(r"""**New York Stock Exchange data.** Three daily time series for December 3, 1962 to December 31, 1986 (6,051 trading days):

- **Log trading volume** — the fraction of outstanding shares traded that day, relative to a 100-day moving average, on the log scale.
- **Dow Jones return** — the difference between the log Dow Jones index on consecutive days.
- **Log volatility** — based on the absolute values of daily price movements.

**Goal:** predict **Log trading volume** tomorrow, given its past values plus those of **Dow Jones return** and **Log volatility**.

## Autocorrelation""")
md(img("10_15-1.png", 560, "Autocorrelation function"))
md(r"""- The **autocorrelation** at lag $\ell$ is the correlation of all pairs $(v_t, v_{t-\ell})$ that are $\ell$ trading days apart.
- Sizable correlations give us confidence that past values help predict the future.
- A curious feature: the response $v_t$ is also a feature $v_{t-\ell}$ — exactly the lag-feature idea from nb16.

## RNN forecaster

We only have one long series. We extract many short mini-series of length $L$ (the **lag**):

$$
X_1 = \begin{pmatrix} v_{t-L} \\ r_{t-L} \\ z_{t-L} \end{pmatrix},
\;\cdots,\;
X_L = \begin{pmatrix} v_{t-1} \\ r_{t-1} \\ z_{t-1} \end{pmatrix},
\qquad Y = v_t.
$$

With $T = 6{,}051$ and $L = 5$, we get 6,046 such $(X, Y)$ pairs — the first 4,281 for training, the next 1,770 for testing. We fit an RNN with 12 hidden units per lag step.""")
md(img("10_16-1.png", 560, "RNN forecasts vs truth"))
md(r"""**Results.** $R^2 = 0.42$ for the RNN, versus $R^2 = 0.18$ for the naive "use yesterday's value" approach.

## Autoregression forecaster

The RNN forecaster is structurally similar to a traditional **autoregression**. Fitting an OLS regression of $\mathbf{y}$ on lagged values gives

$$
\hat{v}_t = \hat\beta_0 + \hat\beta_1 v_{t-1} + \cdots + \hat\beta_L v_{t-L},
$$

an **order-$L$ autoregression**, $AR(L)$. For NYSE we can include lagged **DJ_return** and **log_volatility**, giving $3L+1$ columns.

**Autoregression results for NYSE:**

- $R^2 = 0.41$ for $AR(5)$ (16 parameters)
- $R^2 = 0.42$ for the RNN (205 parameters)
- $R^2 = 0.42$ for $AR(5)$ fit by a neural network
- $R^2 = 0.46$ for all models if we add **day_of_week** of the day being predicted

> **A question that often comes up here:** *"The RNN has 205 parameters and the AR(5) has 16, and they tie — so why ever use the RNN here?"* You wouldn't, for this series. That is the whole point of the next section: on noisy, low-volume series the simpler model wins on every practical axis (speed, interpretability, deployment). The RNN earns its 205 parameters only when the dependence is strongly non-linear *and* you have enough data to estimate them.""")

# ============================================================= WHEN TO USE DL
md(r"""# 12. When to Use Deep Learning

- **CNNs** have had enormous success in image classification and are entering medical diagnosis (digital mammography, ophthalmology, MRI, digital X-rays).
- **RNNs** (and transformers) have had big wins in speech, language translation, and forecasting.

**Should we always use deep learning?** No.

- The big successes occur when the **signal-to-noise ratio** is high (image recognition, language translation), datasets are large, and overfitting is not the main worry.
- For **noisier** data, simpler models often work better:
    - On the **NYSE** data, the **AR(5)** model is much simpler than an RNN and performed as well.
    - On the **IMDB** reviews, a linear model (glmnet) did as well as the neural network — and better than the RNN.""")
md(img("2_7-1.png", 480, "Flexibility vs interpretability",
       "The flexibility-interpretability trade-off (ISLR Figure 2.7)."))
md(r"""**Occam's razor.** Prefer simpler models when they work as well — they are more interpretable. This is the same trade-off curve from the very start of the course; deep learning sits at the far flexible-but-opaque end.""")

# ---- The honest tabular demo + rubric ----
md(r"""## A four-question rubric

Before you reach for deep learning, run a problem through these four questions:

1. **Data shape.** Is the input an image, audio, long text sequence, or a high-dimensional structured object (e.g., a graph)? If yes, lean DL. If it is a tabular row, lean classical ML.
2. **Sample size.** Do you have at least tens of thousands of labeled examples (ideally millions)? If yes, DL has the data it needs. With hundreds or low thousands, classical ML almost always wins.
3. **Compute budget.** Can you spend \~10× the training time and \~50× the inference cost of a tree ensemble? If yes, DL is in scope. If you need a CPU model that predicts in milliseconds, prefer classical ML.
4. **Interpretability requirement.** Does the stakeholder need a feature-by-feature explanation for every prediction? If yes, DL adds friction (post-hoc explainers like SHAP exist but are imperfect). If predictions only need to be accurate, DL is in scope.

**Verdict rule:** answer "yes" to **at least three** questions for DL to be the right *primary* tool. Otherwise, classical ML first.""")

md(r"""## 📝 PAUSE-AND-DO Exercise 1 — When Is Deep Learning the Right Tool? (10 minutes)

**Task:** Apply the four-question rubric to **two** problems: the Bank Churn case competition (nb18) and the DemandCo monthly forecast (nb16). For each, answer the four questions and produce one verdict — *deep learning is / is not the right primary tool*.""")

md(r"""### YOUR RUBRIC ANSWERS HERE:

**Problem A — Bank Churn case competition (tabular, \~10K rows, churn probability for the retention team):**

1. Data shape: *[tabular / sequential / image — and DL lean?]*
2. Sample size: *[count + lean]*
3. Compute budget: *[lean]*
4. Interpretability: *[lean]*
5. **Verdict:** *[DL primary / Classical ML primary]*  *[one sentence why]*

**Problem B — DemandCo monthly demand forecast (60 months, procurement target):**

1. Data shape: *[lean]*
2. Sample size: *[lean]*
3. Compute budget: *[lean]*
4. Interpretability: *[lean]*
5. **Verdict:** *[DL primary / Classical ML primary]*  *[one sentence why]*""")

md(r"""## One honest demo — `MLPClassifier` vs. Gradient Boosting on tabular data

Now the comparison the VP of Strategy actually wants: a feed-forward neural network on a Bank-Churn-style classification problem, evaluated on the **same CV folds**, against gradient boosting from nb13. We use scikit-learn's `MLPClassifier` (an off-the-shelf MLP) so it runs on a CPU in under a minute — no PyTorch needed. We report the mean ROC-AUC with the same Student's *t* 95% CI from nb08.""")
code(r"""import pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

RANDOM_SEED = 474
np.random.seed(RANDOM_SEED)

# Synthetic tabular classification — a realistically noisy business table:
# modest class separation and 10% label noise, the regime where trees and
# nets tie. (same shape as nb15: 5000 rows, 20 features)
X_tab, y_tab = make_classification(
    n_samples=5000, n_features=20, n_informative=6, n_redundant=4,
    n_classes=2, flip_y=0.10, class_sep=0.7, random_state=RANDOM_SEED,
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

mlp_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                          max_iter=200, random_state=RANDOM_SEED)),
])
gbm = GradientBoostingClassifier(
    n_estimators=300, max_depth=3, learning_rate=0.05, random_state=RANDOM_SEED,
)

mlp_aucs = cross_val_score(mlp_pipe, X_tab, y_tab, scoring="roc_auc", cv=cv, n_jobs=-1)
gbm_aucs = cross_val_score(gbm,      X_tab, y_tab, scoring="roc_auc", cv=cv, n_jobs=-1)

k = 5
t_crit = stats.t.ppf(0.975, df=k - 1)  # ~2.776 — same Student's t constant from nb08

def _row(scores):
    m = float(scores.mean()); sd = float(scores.std(ddof=1))
    half_w = t_crit * sd / np.sqrt(k)
    return {'AUC_mean': m, 'AUC_sd': sd, 'AUC_half_w': half_w,
            'AUC_ci_low': m - half_w, 'AUC_ci_high': m + half_w}

summary = pd.DataFrame({
    'MLPClassifier (64,32)': _row(mlp_aucs),
    'GradientBoosting':      _row(gbm_aucs),
}).T
print(summary.round(4))

mlp = summary.loc['MLPClassifier (64,32)']
gbm_row = summary.loc['GradientBoosting']
overlap = not (mlp['AUC_ci_high'] < gbm_row['AUC_ci_low'] or gbm_row['AUC_ci_high'] < mlp['AUC_ci_low'])
if not overlap:
    winner = 'MLPClassifier' if mlp['AUC_mean'] > gbm_row['AUC_mean'] else 'GradientBoosting'
    print(f'\nVerdict: CIs do NOT overlap -> {winner} wins outright (CI-clear margin).')
else:
    print('\nVerdict: CIs overlap -> statistical tie -> simpler / faster model wins by default.')""")

md(r"""**Reading the output:** On a 5,000-row tabular problem with 20 features, the MLP and the gradient-boosted ensemble usually land within a few thousandths of each other — well inside the CV 95% CI. That overlapping interval is the empirical answer to the VP's question: deep learning **does not** outperform a tuned tree ensemble on this kind of data, and the ensemble is faster, more interpretable, and easier to deploy. The story flips when the input is an image, a long text passage, or a high-frequency sensor stream — there a CNN's or transformer's structural priors start paying for themselves.

> **A question that often comes up here:** *"What if I give the MLP more layers?"* You can — try `hidden_layer_sizes=(128, 64, 32)`. On this data the AUC will not move outside the CI; what *will* move is training time and the chance of training instability. Bigger is not better when the data structure does not need bigger. The next exercise has you confirm this yourself.""")

md(r"""## 📝 PAUSE-AND-DO Exercise 2 — Compare and Decide (10 minutes)

**Task:** Add a third candidate to the comparison — a deeper MLP with `hidden_layer_sizes=(128, 64, 32)` — and rerun the CV on the **same folds**. Then write three sentences answering the VP of Strategy's question.

**Hints:**
- Reuse the `mlp_pipe` pattern; only the `hidden_layer_sizes` argument changes.
- Do **not** change the CV splits — identical folds are what makes the comparison honest.""")

code(r"""# YOUR SOLUTION CODE HERE

# Hints:
# deep_pipe = Pipeline([
#     ("scaler", StandardScaler()),
#     ("clf", MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu",
#                           max_iter=300, random_state=RANDOM_SEED)),
# ])
# deep_aucs = cross_val_score(deep_pipe, X_tab, y_tab, scoring="roc_auc", cv=cv, n_jobs=-1)
# Build a 3-row summary table and compare the CIs.""")

md(r"""### INSTRUCTOR SOLUTION — Exercise 2""")
code(r"""# INSTRUCTOR SOLUTION
deep_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu",
                          max_iter=300, random_state=RANDOM_SEED)),
])
deep_aucs = cross_val_score(deep_pipe, X_tab, y_tab, scoring="roc_auc", cv=cv, n_jobs=-1)

summary3 = pd.DataFrame({
    'MLPClassifier (64,32)':      _row(mlp_aucs),
    'MLPClassifier (128,64,32)':  _row(deep_aucs),
    'GradientBoosting':           _row(gbm_aucs),
}).T
print(summary3.round(4))

# All three CIs overlap -> statistical tie -> the deeper net buys nothing here.
print("\nAll three 95% CIs overlap: the deeper MLP does not beat the simpler MLP or"
      "\nthe gradient-boosted ensemble on this tabular data. Bigger is not better when"
      "\nthe data structure does not demand it.")""")

md(r"""<!-- INSTRUCTOR SOLUTION -->
**Reading the output:** Adding a third hidden layer leaves the AUC inside the same confidence interval — the deeper MLP is a statistical tie with both the shallow MLP and gradient boosting. The honest three-sentence answer to the VP writes itself: (1) on this tabular data all three models tie within the CV 95% CI; (2) this generalizes to Bank Churn because that problem is also a few-thousand-row tabular table, exactly where classical ML wins; (3) deep learning would earn its keep at TechCorp on a genuinely different data asset — say, classifying product photos or parsing free-text support tickets — not on the churn table.""")

md(r"""### YOUR ANSWER TO THE VP HERE:

**Three sentences for the VP of Strategy:**

1. *[Empirical comparison — which model won on this data, and by how much in CV CI terms?]*
2. *[Why this generalizes to the Bank Churn problem — data-shape and sample-size argument from the rubric]*
3. *[Where deep learning **would** earn its keep at TechCorp — name one concrete use case from the company's likely data assets]*""")

# ============================================================= LLM
md(r"""# 13. Special Topic: Large Language Models (LLM)

The LLMs you have used all course (Gemini, ChatGPT, Claude) are **transformers** — the successor to the RNN. Four short videos build the intuition, from what an LLM is to how attention and memory work.

<center>
<iframe width="760" height="430" src="https://www.youtube.com/embed/LPZh9BOjkQs" title="LLMs explained briefly" frameborder="0" allowfullscreen></iframe>
<br><a href="https://www.youtube.com/watch?v=LPZh9BOjkQs" target="_blank">Large Language Models explained briefly</a>
<br><br>
<iframe width="760" height="430" src="https://www.youtube.com/embed/wjZofJX0v4M" title="Transformers" frameborder="0" allowfullscreen></iframe>
<br><a href="https://www.youtube.com/watch?v=wjZofJX0v4M" target="_blank">Transformers, the tech behind LLMs</a>
<br><br>
<iframe width="760" height="430" src="https://www.youtube.com/embed/eMlx5fFNoYc" title="Attention" frameborder="0" allowfullscreen></iframe>
<br><a href="https://www.youtube.com/watch?v=eMlx5fFNoYc" target="_blank">Attention in transformers, step-by-step</a>
<br><br>
<iframe width="760" height="430" src="https://www.youtube.com/embed/9-Jl0dxWQs8" title="LLM memory" frameborder="0" allowfullscreen></iframe>
<br><a href="https://www.youtube.com/watch?v=9-Jl0dxWQs8" target="_blank">How might LLMs store facts</a>
</center>""")

# ============================================================= ADDITIONAL + SUMMARY
md(r"""## Additional Material

- [3Blue1Brown: Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [*Deep Learning*, by Goodfellow, Bengio, and Courville](https://www.deeplearningbook.org/)
- [Welch Labs: Neural Networks Demystified](https://www.youtube.com/watch?v=bxe2T-V8XRs&list=PLiaHhY2iBX9hdHaRr6b7XevZtgZRa1PoU)
- [Distill: A Gentle Introduction to Graph Neural Networks](https://distill.pub/)
- [Neural Networks and Deep Learning, by Michael Nielsen](http://neuralnetworksanddeeplearning.com/)
- [Deep Learning with PyTorch: Step-by-Step](https://github.com/dvgodoy/PyTorchStepByStep/tree/master)""")

md(r"""## 14. Wrap-Up — Key Takeaways

1. **Deep learning is the *engineering stack* around old neural-network math.** Compute + data + frameworks were the bottleneck; the math was mostly already there in the 1980s.
2. **The three structural inventions are MLP, CNN, and RNN/Transformer.** MLP for tabular extras, CNN for images, RNN/Transformer for sequences. Name the right one for a problem and you have \~80% of what an analyst needs in this conversation.
3. **You can build and run a network in PyTorch.** Subclass `nn.Module`, define `forward`, send a tensor through `model(X)`, softmax the logits, `argmax` for the label — you did exactly this above, including predicting a random photo.
4. **For tabular business problems with thousands of rows, gradient boosting almost always wins.** The MLP is a credible candidate but rarely the champion; the four-question rubric keeps you honest.
5. **PyTorch is the safer default to learn next** — the Hugging Face hub, almost every research paper, and the modern LLM ecosystem are PyTorch-first.

> **A question that often comes up here:** *"What about ChatGPT, Claude, and the LLMs we use every day?"* They are transformers (a successor to RNNs) trained on enormous text corpora. As an analyst you will mostly **use** them through APIs (you have done this all course via Gemini prompts) rather than train them. The "know the structural ideas" bar from this notebook is the right bar — it lets you read the docs and the announcement papers without getting lost.

**Next stop — nb20: Course End and Reflection.** Tomorrow is delivery day: M4 poster, Kaggle final submission, intra-group peer evaluation, and the reflection survey that closes the course. Today's awareness gives you the language for the "what's next?" line on your poster, and one credible, evidence-based answer when a colleague asks "should we be doing deep learning?".""")

md(r"""## Participation Assignment Submission Instructions

1. **Complete both PAUSE-AND-DO exercises.**
2. **Run all cells** (the PyTorch lab downloads FashionMNIST on first run — give it a moment).
3. **Save with output** and submit `nb19_deep_learning_<your_lastname>.ipynb` to Brightspace.

### Next Step

- **Notebook 20** — Final submission + peer review (Day 20)

**Bibliography**
- ISLP, Chapter 10: Deep Learning (the textbook chapter behind this notebook's lecture slides).
- PyTorch "Learn the Basics": <https://pytorch.org/tutorials/beginner/basics/intro.html>.
- *Deep Learning* by Goodfellow, Bengio, and Courville (free online): <https://www.deeplearningbook.org/>.
- 3Blue1Brown's neural-network video series (the best visual intuition online).""")

md(r"""<center>

# Thank you!

</center>""")

# ============================================================= WRITE
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

# normalize source into list-of-lines with trailing newlines preserved
for c in nb["cells"]:
    s = c["source"]
    lines = s.splitlines(keepends=True)
    c["source"] = lines

out = Path("notebooks/nb19_deep_learning_instructor.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("Wrote", out, "with", len(cells), "cells")
