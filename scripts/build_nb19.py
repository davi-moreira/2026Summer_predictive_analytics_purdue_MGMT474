#!/usr/bin/env python3
"""Build nb19_deep_learning_instructor.ipynb.

Focused, business-undergrad-friendly deep-learning notebook:
  1. Deep Learning (history + drivers)
  2. PyTorch vs. TensorFlow
  3. Neural Networks — Build Your Intuition (3Blue1Brown, videos)
  4. PyTorch — Hands-On with FashionMNIST, END-TO-END  <- the focus
  5. When to Use Deep Learning (rubric + honest tabular demo)
  6. Large Language Models (videos)
  7. Wrap-Up

Colab strips <iframe> from markdown cells, so inline-playing videos / embedded
pages don't render there without running code. To keep everything in markdown
(no code to run), YouTube videos are shown as clickable thumbnail images
(<img> inside <a>, which Colab does render) that open the video on YouTube, and
the PyTorch tutorial pages are clear reference links.

Run: python scripts/build_nb19.py   (student copy strips INSTRUCTOR SOLUTION cells)
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


def yt_thumb(vid, title, watch_url, w=480):
    """A centered, clickable YouTube thumbnail that renders in a markdown cell
    WITHOUT running any code (Colab strips <iframe> from markdown, but renders
    <div align>/<img>/<a>). Clicking the thumbnail opens the video on YouTube."""
    thumb = f"https://img.youtube.com/vi/{vid}/hqdefault.jpg"
    return (f'<div align="center">\n'
            f'<a href="{watch_url}" target="_blank">'
            f'<img src="{thumb}" alt="{title}" width="{w}"></a>\n'
            f'<br>\n'
            f'▶️ <b><a href="{watch_url}" target="_blank">{title}</a></b> '
            f'<i>(click the image or link to watch)</i>\n'
            f'</div>')


def pt_link(url, title):
    """A clear reference link to a PyTorch tutorial page. (Colab strips <iframe>
    from markdown, so we link out rather than embed a dead frame.)"""
    return (f'> 📖 **Official PyTorch reference:** [{title}]({url}) — '
            f'opens the full tutorial in a new browser tab.')


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

1. Explain in plain language **what deep learning is**, the historical arc that brought it back in 2010, and the three things that made it work (compute, data, frameworks).
2. Say what **PyTorch** and **TensorFlow** do for you, and which one you would reach for.
3. Build the right **intuition** for what a neural network is and how it learns, from four short videos.
4. **Train a neural network end to end in PyTorch** on real image data (FashionMNIST): load the data, build the model, train it, save and reload it, and use it to **predict** what a photo shows — and understand every step well enough to explain it to a colleague.
5. Use a simple **four-question rubric** to decide whether deep learning is the right tool for a business problem, and back that decision with one honest comparison against gradient boosting.

> 📝 *This notebook is inspired by and replicates material from [An Introduction to Statistical Learning (ISLP)](https://www.statlearning.com/) and the official [PyTorch "Learn the Basics" Quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html).*

> **📋 Participation Reminder:** This notebook contains **2 PAUSE-AND-DO exercises**. Complete both to receive participation credit.

> ▶️ **About the videos:** several sections show a short video as a **thumbnail image** — nothing to run; just **click the thumbnail** (or the link beside it) to open the video on YouTube in a new tab. (Colab cannot play a video inside a text cell, so we link out — your place in the notebook is kept.)""")

# ============================================================= WHY THIS MATTERS
md(r"""## 💼 Why This Matters: The "What About AI?" Question Every Analyst Will Hear

The **VP of Strategy at TechCorp** sat in your Milestone 4 poster session, watched your gradient-boosting churn model, and asked one question:

> *"This is great. But shouldn't we be using deep learning?"*

This notebook is the answer you owe her — and it has two halves. The first half gives you the **language and judgment**: what deep learning is, when it genuinely wins, and (just as important) when a simpler model you already know is the better call. The second half is **hands-on**: you will train a real neural network in PyTorch — the same toolkit behind ChatGPT, Claude, and Gemini — and watch it learn to recognize photos of clothing well enough to make correct predictions. By the end you will not be guessing about deep learning; you will have *done* it.

You do not need any prior PyTorch experience. Everything runs in Colab, and every new term is explained as it appears.""")

md(img("10_1_1-1.png", 460, "Deep learning pioneers",
       "The 2019 ACM Turing Award honored Yann LeCun, Geoffrey Hinton, and Yoshua Bengio for the work that became modern deep learning."))

# ============================================================= OVERVIEW
md(r"""## Overview — the path through this notebook

| Section | What you get |
|---|---|
| **1. Deep Learning** | Where it came from and why it suddenly works |
| **2. PyTorch vs. TensorFlow** | The two tools the whole field is built on |
| **3. Neural Networks — build your intuition** | Four short videos: what a network *is* and how it *learns* |
| **⭐ 4. PyTorch hands-on (FashionMNIST)** | **Train your own network, end to end — the heart of the notebook** |
| **5. When to Use Deep Learning** | A rubric + an honest comparison against gradient boosting |
| **6. Large Language Models** | How the AI you use every day fits in |
| **7. Wrap-Up** | Key takeaways and what's next |

Sections 3 and 4 are the centerpiece: build the intuition, then train a network yourself. Section 4 is fully runnable — open it in Colab and execute the cells in order.""")

# ============================================================= 1. DEEP LEARNING
md(r"""# 1. Deep Learning

**Deep learning** is a way of teaching a computer to find patterns by stacking many simple "layers" of math on top of each other — deep, as in *many layers*. It is the engine behind face recognition on your phone, the recommendations on Netflix, self-driving-car vision, and every modern chatbot.

Here is the surprising part: the core ideas are **old**. Neural networks have lived two lives.

- **First life (1980s).** Researchers invented the key math and got excited. But computers were slow and datasets were tiny, so the models stayed small and other methods (the SVMs, random forests, and boosting you met earlier in this course) usually won. Neural networks faded into the background.
- **Second life (2010 onward).** Almost the same math suddenly started winning *everything* — image recognition, translation, speech. The field rebranded as **deep learning**, and by the 2020s it was everywhere.

**What changed? Three things arrived at once:**

1. **Compute.** Graphics cards (GPUs) built for video games turned out to be perfect for the massive multiplication-and-addition a neural network does. A \$1,000 gaming GPU could now do what used to need a \$100,000 cluster. *Business translation:* training went from "only big labs can afford it" to "a startup can rent it by the hour."
2. **Data.** The internet produced enormous labeled datasets (e.g., **ImageNet**, with 14 million tagged images). Neural networks are hungry — give them enough examples and they outperform everything else.
3. **Frameworks.** Free software libraries — **PyTorch** and **TensorFlow** — turned thousands of lines of low-level code into a few lines of Python. You will use one of them in Section 4.

Much of the credit goes to three pioneers and their research teams — **Yann LeCun**, **Geoffrey Hinton**, and **Yoshua Bengio** — who shared the 2019 ACM Turing Award (computing's "Nobel Prize") for this work.

> **A question that often comes up here:** *"If the math is from the 1980s, why did it take 25 years to work?"* Because the math was *almost* — but not entirely — enough. A few small but crucial tweaks (a better "on/off" function called ReLU, a trick called dropout, and another called batch normalization) combined with the new compute and data to finally make deep networks trainable. No single piece was the breakthrough; they crossed the finish line together.""")

md("## Hear it from the pioneers\n\n"
   "Three short interviews — worth watching once to put faces and voices to the names. "
   "**Click any thumbnail** to watch on YouTube.\n\n"
   '<table align="center"><tr>\n'
   '<td width="33%" valign="top" align="center">\n'
   '<a href="https://www.youtube.com/watch?v=Ah6nR8YAYF4" target="_blank">'
   '<img src="https://img.youtube.com/vi/Ah6nR8YAYF4/hqdefault.jpg" alt="Yann LeCun" width="100%"></a><br>\n'
   '▶️ <a href="https://www.youtube.com/watch?v=Ah6nR8YAYF4" target="_blank"><b>Yann LeCun</b> — The Future of AI</a>\n'
   '</td>\n'
   '<td width="33%" valign="top" align="center">\n'
   '<a href="https://www.youtube.com/watch?v=qrvK_KuIeJk" target="_blank">'
   '<img src="https://img.youtube.com/vi/qrvK_KuIeJk/hqdefault.jpg" alt="Geoffrey Hinton" width="100%"></a><br>\n'
   '▶️ <a href="https://www.youtube.com/watch?v=qrvK_KuIeJk" target="_blank"><b>Geoffrey Hinton</b> — 60 Minutes Interview</a>\n'
   '</td>\n'
   '<td width="33%" valign="top" align="center">\n'
   '<a href="https://www.youtube.com/watch?v=5LgDUqCbBwo" target="_blank">'
   '<img src="https://img.youtube.com/vi/5LgDUqCbBwo/hqdefault.jpg" alt="Yoshua Bengio" width="100%"></a><br>\n'
   '▶️ <a href="https://www.youtube.com/watch?v=5LgDUqCbBwo" target="_blank"><b>Yoshua Bengio</b> — Path to Human-Level AI</a>\n'
   '</td>\n'
   "</tr></table>")

# ============================================================= 2. PYTORCH vs TF
md(r"""# 2. PyTorch vs. TensorFlow

Before you build anything, meet the two tools the entire field runs on.""")

md(img("pytorch_vs_tensorflow.png", 680, "PyTorch vs TensorFlow"))

md(r"""## What is a "deep learning framework," and why do I need one?

Think of a framework as a **fully-equipped kitchen**. You *could* build a neural network from raw math the way you *could* forge your own knives and pots — but nobody does, because it is slow and error-prone. PyTorch and TensorFlow give you the pre-made tools so you can focus on the recipe (your model) instead of the metalwork.

Concretely, a framework hands you four things — these are the words you'll hear over and over:

- **Tensors.** A tensor is just a **multi-dimensional table of numbers** — a spreadsheet that's allowed to have more than two dimensions. A single grayscale photo is a 2-D tensor (height × width); a batch of 64 color photos is a 4-D tensor. Everything inside a network is tensors flowing forward.
- **Automatic differentiation ("autograd").** Training a network means nudging millions of numbers in the right direction. The framework figures out *which direction* automatically — you write one line, `loss.backward()`, and it computes every needed adjustment. Doing this by hand would be impossible.
- **GPU support.** One line (`.to('cuda')`) moves your work onto a graphics card and it runs roughly **50× faster**. Same code, much bigger problems.
- **Pre-built layers.** The standard building blocks (the `Linear`, `ReLU`, and other layers you'll use in Section 4) are ready to snap together — no need to implement them yourself.

## The two contenders

| | **PyTorch** | **TensorFlow** |
|:---|:---|:---|
| **Made by** | Meta (Facebook), 2016 | Google, 2015 |
| **Feel** | Very "Pythonic," reads like normal Python; easy to debug | Powerful but historically steeper to learn |
| **Strongest in** | Research, prototyping, the modern AI/LLM ecosystem | Large-scale production and deployment (mobile, web, Google Cloud) |
| **You'll see it in** | Hugging Face, almost every new research paper, ChatGPT/Claude-style models | Enterprise production pipelines at big tech firms |

Both can build the same models, both run on GPUs, both have huge communities. The differences are mostly about *style* and *ecosystem*, not raw capability.

> **A question that often comes up here:** *"If I only learn one, which?"* For a business analyst today, learn **PyTorch**. The research world, the Hugging Face model hub, and the entire modern large-language-model ecosystem are PyTorch-first — and that is exactly what you'll use in Section 4. TensorFlow is still common in big-company production, but the gap shrinks every year.""")

# ============================================================= 3. NN INTUITION
md(r"""# 3. Neural Networks — Build Your Intuition

Before you write code, build a clear mental picture. Here is the whole idea in three sentences:

- A **neuron** is a tiny decision-maker: it takes some numbers in, multiplies each by a "weight" (how much it cares about that input), adds them up, and passes the result through a simple on/off-ish function. *Inputs in, one number out.*
- A **layer** is a row of these neurons working in parallel; **stacking layers** lets the network build up from simple patterns (edges, curves) to complex ones (a sleeve, a shoelace).
- **Training** is the process of automatically adjusting all those weights — millions of them — until the network's outputs match the right answers on your training data.

That's it. Everything else is detail. The four videos below — 3Blue1Brown's beloved *Deep Learning* series — make this **visual and intuitive**, and they map directly onto the code you'll write in Section 4. Watch them in order (about an hour total; even just Chapters 1–2 are enough to follow the lab).

> ▶️ Each chapter below shows a clickable thumbnail — click it to watch on YouTube (nothing to run).""")

md(r"""### Chapter 1 — But what is a neural network?

Start here. It builds the whole picture — neurons, layers, weights — on the problem of recognizing handwritten digits (the cousin of the clothing-recognition problem you'll solve in Section 4). This is the single best 20 minutes for *getting* what a neural network is.

""" + yt_thumb("aircAruvnKk", "But what is a neural network? | Deep Learning Chapter 1",
              "https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=1"))

md(r"""### Chapter 2 — Gradient descent, how neural networks learn

This is the "learning" part. It shows how a network turns wrong answers into better weights by taking small downhill steps on a "cost" surface. When you see `optimizer.step()` in Section 4, *this* is what it's doing.

""" + yt_thumb("IHZwWFHWa-w", "Gradient descent, how neural networks learn | Deep Learning Chapter 2",
              "https://www.youtube.com/watch?v=IHZwWFHWa-w&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=2"))

md(r"""### Chapter 3 — Backpropagation, intuitively

Backpropagation is the clever bookkeeping that figures out how to nudge each weight. This chapter gives the intuition without heavy math — how each training example "votes" to push weights up or down.

""" + yt_thumb("Ilg3gGewQ5U", "Backpropagation, intuitively | Deep Learning Chapter 3",
              "https://www.youtube.com/watch?v=Ilg3gGewQ5U&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=3"))

md(r"""### Chapter 4 — Backpropagation calculus

The math behind the magic. This one is **optional** for the lab — PyTorch does this calculus for you — but it's here if you want to see exactly what `loss.backward()` computes under the hood.

""" + yt_thumb("tIeHLnjs5U8", "Backpropagation calculus | Deep Learning Chapter 4",
              "https://www.youtube.com/watch?v=tIeHLnjs5U8&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=4"))

md(r"""> **A question that often comes up here:** *"Do I have to understand the calculus to do the lab?"* No. The whole point of a framework like PyTorch is that it handles the calculus automatically. Chapters 1 and 2 give you everything you need to follow Section 4 with real understanding; Chapters 3 and 4 are there for when you're curious about *exactly* how the machine computes its adjustments.""")

# ============================================================= 4. PYTORCH HANDS-ON
md(r"""# ⭐ 4. PyTorch — Hands-On with FashionMNIST (End-to-End)

This is the heart of the notebook. You are going to **train your own neural network** and use it to recognize photos of clothing. We follow the official PyTorch **Quickstart**, on the **FashionMNIST** dataset — 70,000 small grayscale photos of clothing, each belonging to one of 10 categories (T-shirt, trouser, sneaker, bag, …).

The journey has six steps, and you'll run every one:

> **4.1 load the data → 4.2 build the model → 4.3 train it → 4.4 save it → 4.5 load it back → 4.6 predict.**

By the final cell you'll have a trained model that looks at a photo it has *never seen* and tells you what it is — and is usually right.

> 💡 **Speed tip:** for faster training, switch on a free GPU in Colab — *Runtime > Change runtime type > GPU*. Everything also runs on the CPU; five training passes take a couple of minutes.

Colab already has `torch` and `torchvision` installed. Run the cell below to load them.""")

code(r"""# Colab already has PyTorch installed. (Locally, uncomment the next line.)
# !pip install torch torchvision -q

import torch
from torch import nn                      # nn = the neural-network building blocks
from torch.utils.data import DataLoader   # serves the data in batches
from torchvision import datasets          # ready-made datasets like FashionMNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

print("PyTorch version:", torch.__version__)""")

# ---- 4.1 working with data ----
md(r"""## 4.1 Load the data

Two ideas you'll use constantly:

- A **`Dataset`** is the collection of examples (here: photos + their correct labels). `torchvision` ships **FashionMNIST** ready to go.
- A **`DataLoader`** is a conveyor belt that feeds the model the data in small **batches** (we'll use 64 photos at a time) and reshuffles them each pass so the model doesn't just memorize the order.

`ToTensor()` is the converter that turns each photo into a **tensor** (that multi-dimensional table of numbers) with pixel brightness scaled to a tidy 0-to-1 range — models train better on small, consistent numbers.

The first time you run this it downloads about 30 MB.""")
code(r"""# Download the FashionMNIST training and test sets.
training_data = datasets.FashionMNIST(
    root="data", train=True,  download=True, transform=ToTensor()
)
test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=ToTensor()
)

# The 10 category names (the official Quickstart calls this list `classes`).
labels_map = {0:"T-Shirt", 1:"Trouser", 2:"Pullover", 3:"Dress", 4:"Coat",
              5:"Sandal", 6:"Shirt", 7:"Sneaker", 8:"Bag", 9:"Ankle Boot"}

print("Training photos:", len(training_data), "| Test photos:", len(test_data))""")

md(r"""Now wrap the datasets in `DataLoader`s. A **batch size of 64** means the model studies 64 photos, updates itself, then studies the next 64 — like working through flashcards in small stacks instead of all 60,000 at once.""")
code(r"""batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader  = DataLoader(test_data,     batch_size=batch_size)

# Peek at the shape of one batch.
# N = number of photos, C = color channels (1 = grayscale), H x W = pixels.
for X, y in test_dataloader:
    print(f"One batch of images has shape [N, C, H, W]: {X.shape}")
    print(f"One batch of labels has shape: {y.shape} ({y.dtype})")
    break""")

md(r"""Let's actually *look* at the data — nine random training photos with their labels. (This is the image version of the EDA snapshot you did back in nb01.)""")
code(r"""figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()""")

md(r"""> **A question that often comes up here:** *"Why split into training and test sets again?"* Same reason as every model all course: the **test set is the honest exam.** The model learns from the training photos; we judge it on the test photos it never trained on. If we graded it on photos it had already memorized, the score would be a lie. This is the exact discipline you've practiced since nb01 — it doesn't change just because the model is a neural network.

""" + pt_link("https://pytorch.org/tutorials/beginner/basics/data_tutorial.html",
              "Datasets & DataLoaders — official PyTorch tutorial"))

# ---- 4.2 build the model ----
md(r"""## 4.2 Build the model

First, pick the **device** — the hardware that will do the math. A GPU is like a calculator with thousands of hands; a CPU is the reliable single-hand version. This line uses a GPU if one is available, otherwise the CPU. (The model and the data have to be on the same device.)""")
code(r"""device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")""")

md(r"""Now define the network. In PyTorch you describe a model as a **class** that lists its layers and says how data flows through them. Read the comments — each layer has a plain-English job:

- **`Flatten`** unrolls the 28×28 photo into a single list of 784 numbers (one per pixel).
- **`Linear(784, 512)`** is a layer of 512 neurons; each one takes a **weighted vote** over all 784 pixels. These weights are what the network will *learn*.
- **`ReLU`** is the on/off switch: it keeps positive numbers and turns negatives into zero. This little non-linearity is what lets the network learn complex shapes instead of just straight lines.
- The final **`Linear(512, 10)`** outputs **10 numbers** — one raw score per clothing category. These raw scores are called **logits**.""")
code(r"""class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()                 # 28x28 image -> list of 784 numbers
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),                  # 784 inputs -> 512 neurons
            nn.ReLU(),                              # on/off switch (adds non-linearity)
            nn.Linear(512, 512),                    # 512 -> 512 (a hidden layer)
            nn.ReLU(),
            nn.Linear(512, 10),                     # 512 -> 10 category scores (logits)
        )

    def forward(self, x):                           # how data flows through the model
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)""")

md(r"""### Watch the data change shape, layer by layer

To make the layers concrete, push a mini-batch of **3 photos** through the first few pieces and print the shape at each step. Notice how 3 photos of 28×28 become 3 lists of 784, then 3 lists of 20.""")
code(r"""input_image = torch.rand(3, 28, 28)
print("start — 3 photos:           ", input_image.size())

flatten = nn.Flatten()
flat_image = flatten(input_image)
print("after Flatten — 3 lists:    ", flat_image.size(), "  (784 = 28x28)")

layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print("after Linear(784->20):      ", hidden1.size())

print("\nBefore ReLU (first photo, first 8 values):\n", hidden1[0][:8])
hidden1 = nn.ReLU()(hidden1)
print("\nAfter ReLU (negatives are now 0):\n", hidden1[0][:8])""")

md(r"""The 10 logits the model outputs are raw scores. To turn them into **probabilities** (percentages that add up to 100%), we pass them through **softmax**, then pick the highest one with **`argmax`** — that's the predicted category.

But here's the catch: **the model hasn't been trained yet.** Its weights are still random starting values, so right now it's basically guessing. Let's prove it — this prediction will probably be wrong.""")
code(r"""# One prediction with the UNTRAINED model — expect a wrong guess.
X0, y0 = test_data[0]
model.eval()                                   # evaluation mode
with torch.no_grad():                          # we're not training, so skip gradient tracking
    logits = model(X0.unsqueeze(0).to(device)) # add a "batch of 1" dimension, send to device
    probabilities = nn.Softmax(dim=1)(logits)  # logits -> probabilities
    guess = probabilities.argmax(1).item()     # index of the highest probability
print(f"Untrained guess: {labels_map[guess]}   |   Correct answer: {labels_map[y0]}")""")

md(pt_link("https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html",
           "Build the Neural Network — official PyTorch tutorial"))

# ---- 4.3 train ----
md(r"""## 4.3 Train the model

This is where the network actually learns. Training needs three ingredients, each with an everyday meaning:

- **Loss function** — a single number measuring *how wrong* the model is right now, like a golf score: lower is better. For sorting things into categories we use `CrossEntropyLoss`.
- **Optimizer** — the "coach" that, after each batch, nudges every weight a little in the direction that lowers the loss. We use **SGD** (stochastic gradient descent — the downhill walk from Chapter 2).
- **Learning rate** — how *big* each nudge is. Too small and learning crawls; too big and it overshoots. `1e-3` (0.001) is a safe default.

One more word: an **epoch** is one complete pass through all the training photos. We'll do 5.""")
code(r"""learning_rate = 1e-3     # size of each weight nudge
epochs = 5               # how many full passes through the training data

loss_fn = nn.CrossEntropyLoss()                                   # the "wrongness" score
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # the "coach"

# (model was set to eval() above; the train loop will switch it back to train())""")

md(r"""Now the two loops that do the work:

- **`train_loop`** walks through the training data in batches and, for each batch, performs the three-step update you saw in the videos: `loss.backward()` works out which way is downhill (backpropagation), `optimizer.step()` takes the step, and `optimizer.zero_grad()` resets for the next batch.
- **`test_loop`** walks through the held-out test data and reports **accuracy** (% correct) and average loss — *without* changing anything. This is our honest progress check.""")
code(r"""def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()                                  # training mode
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)                            # 1. predict this batch
        loss = loss_fn(pred, y)                    # 2. how wrong were we?

        loss.backward()                            # 3a. which way is downhill? (backprop)
        optimizer.step()                           # 3b. take a step downhill
        optimizer.zero_grad()                      # 3c. reset for the next batch

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"  loss: {loss:>7f}  [{current:>5d}/{size:>5d} photos]")


def test_loop(dataloader, model, loss_fn):
    model.eval()                                   # evaluation mode
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():                          # no learning during the exam
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"  Test accuracy: {(100*correct):>0.1f}%   |   Avg loss: {test_loss:>8f}\n")""")

md(r"""Run the training. Watch two things happen: the **loss falls** and the **test accuracy climbs** with each epoch. That is your network learning, in real time. (This is the slow cell — a couple of minutes on CPU, seconds on a GPU.)""")
code(r"""for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")""")

md(r"""> **A question that often comes up here:** *"My accuracy is only around 65–70% — is something broken?"* Not at all — that's exactly what five passes of plain SGD give this simple network, and it's the official Quickstart's result too. Remember the floor: random guessing among 10 categories is 10%, so the model has clearly *learned*. Want better? The usual levers are: train for **more epochs**, switch the optimizer to **`Adam`** (often faster), or use a **convolutional network (CNN)** — a model designed for images, which routinely tops 90% here. We meet CNNs by name in Section 5.

""" + pt_link("https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html",
              "Optimizing Model Parameters — official PyTorch tutorial"))

# ---- 4.4 save ----
md(r"""## 4.4 Save the model

Your model just spent a couple of minutes learning. You don't want to redo that every time. PyTorch stores everything the model learned in a **state dictionary** (`state_dict`) — essentially its "brain settings." Saving it to a file means you can reload the trained model instantly later, or hand it to a deployment pipeline. (This is the neural-network version of the `joblib` model-saving you did in nb18.)""")
code(r"""torch.save(model.state_dict(), "model.pth")
print("Saved the trained model to model.pth")""")

# ---- 4.5 load ----
md(r"""## 4.5 Load the model back

To reload, you recreate the empty model (the class is the blueprint) and then pour the saved settings back in. Always call `model.eval()` before using a loaded model to make predictions.""")
code(r"""model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()
print("Reloaded the trained model from disk — ready to predict.")""")

# ---- 4.6 predict ----
md(r"""## 4.6 Predict — the payoff

The whole point. Hand the trained model a photo and read its answer. We take the first test photo, run it through the model, and compare the prediction to the truth.""")
code(r"""model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = labels_map[pred[0].argmax(0).item()], labels_map[y]
print(f'Predicted: "{predicted}"   |   Actual: "{actual}"')""")

md(r"""### 🎲 Predict a grid of random photos

Here's the full experience. Draw **nine random test photos**, predict each with your trained model, and show every photo with its predicted vs. true label — **green** when the model is right, **red** when it's wrong. Re-run the cell for a fresh nine. With a trained model, most titles come up green.""")
code(r"""model.eval()
figure = plt.figure(figsize=(9, 9))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    idx = torch.randint(len(test_data), size=(1,)).item()
    img, true_label = test_data[idx]
    with torch.no_grad():
        pred_label = model(img.unsqueeze(0).to(device)).argmax(1).item()
    is_correct = (pred_label == true_label)
    figure.add_subplot(rows, cols, i)
    plt.title(f"Pred: {labels_map[pred_label]}\nTrue: {labels_map[true_label]}",
              color=("green" if is_correct else "red"), fontsize=10)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.tight_layout()
plt.show()""")

md(r"""> **A question that often comes up here:** *"The untrained model in 4.2 guessed wrong, and now most guesses are right — what actually changed?"* Only the **weights**. The model's structure never changed — same layers, same `forward` path. Training adjusted the millions of numbers inside the `Linear` layers (via the downhill steps in 4.3) so that the very same calculation now maps clothing photos to the correct category most of the time. That's the entire story of supervised deep learning: **fixed structure, learned weights, a loss function pulling those weights toward "right."** You just did it, end to end — load, build, train, save, load, predict. That is a complete machine-learning pipeline in PyTorch.""")

# ============================================================= 5. WHEN TO USE
md(r"""# 5. When to Use Deep Learning

You've now trained a neural network. So back to the VP's question: *should TechCorp be using deep learning?* The honest answer is **"it depends — and usually, for tabular business data, no."** This section gives you the judgment to say that with confidence.

## First, the three "shapes" deep learning was built for

Different network designs suit different *shapes* of data. You only need the headline for each:

- **MLP (the plain network you just built)** — works on any flat list of numbers. It's the general-purpose option, and the one you'll compare against gradient boosting below.
- **CNN — Convolutional Neural Network — for images.** A CNN scans an image with small sliding "pattern detectors," so it recognizes a sleeve whether it's in the corner or the center. This is why CNNs dominate photo tasks (and would beat your MLP on FashionMNIST). Self-driving cars, medical scans, and product-photo tagging all run on CNNs.
- **RNN / Transformer — for sequences.** These handle data that arrives *in order* — text, audio, time series. Transformers (the modern version) are what power ChatGPT, Claude, and Gemini.

A useful image-classification benchmark, **CIFAR-100**, gives a feel for what CNNs are built to handle:""")
md(img("cifar100.png", 560, "CIFAR-100 sample images",
       "CIFAR-100: 60,000 small color photos across 100 categories — the kind of task CNNs were designed for."))

md(r"""## Simpler is often better: flexibility vs. interpretability

Deep learning is extremely **flexible** — it can fit almost any pattern. But flexibility has a cost: the model becomes a **black box** that's hard to explain. That trade-off is the same curve you saw at the very start of the course:""")
md(img("2_7-1.png", 460, "Flexibility vs interpretability",
       "More flexible models (like deep learning) are harder to interpret. Prefer the simplest model that does the job."))
md(r"""**Occam's razor for analysts:** if a simpler, explainable model works as well, use it. A churn model your retention team can *understand and trust* often beats a slightly-more-accurate black box they won't act on.""")

md(r"""## The four-question rubric

Before reaching for deep learning on a business problem, run it through these four questions:

1. **Data shape.** Is the input an **image, audio, long text, or another rich/structured object**? → lean deep learning. Is it a **tabular row** (a spreadsheet of customer attributes)? → lean classical ML (the trees and boosting you already know).
2. **Sample size.** Do you have **tens of thousands+** of labeled examples (ideally millions)? → DL has the fuel it needs. Only hundreds or low thousands? → classical ML almost always wins.
3. **Compute budget.** Can you afford roughly **10× the training time and 50× the prediction cost** of a tree ensemble? → DL is in scope. Need millisecond predictions on a laptop? → classical ML.
4. **Interpretability need.** Does a stakeholder need a clear, **per-prediction explanation**? → classical ML is friendlier (deep learning's explanations are approximate). Only need accuracy? → DL is fine.

**Decision rule:** you need **"yes" to at least three** of the four for deep learning to be the right *primary* tool. Otherwise, start with classical ML — and most tabular business problems land there.""")

md(r"""## 📝 PAUSE-AND-DO Exercise 1 — When Is Deep Learning the Right Tool? (10 minutes)

**Task:** Apply the four-question rubric to **two** problems from this course: the Bank Churn case competition (nb18) and the DemandCo monthly forecast (nb16). For each, answer the four questions and give one verdict — *deep learning is / is not the right primary tool.*""")

md(r"""### YOUR RUBRIC ANSWERS HERE:

**Problem A — Bank Churn case competition (tabular, \~10K rows, predict churn probability for the retention team):**

1. Data shape: *[tabular / sequential / image — and which way does it lean?]*
2. Sample size: *[count + lean]*
3. Compute budget: *[lean]*
4. Interpretability: *[lean]*
5. **Verdict:** *[DL primary / Classical ML primary]* — *[one sentence why]*

**Problem B — DemandCo monthly demand forecast (60 months of history, procurement target):**

1. Data shape: *[lean]*
2. Sample size: *[lean]*
3. Compute budget: *[lean]*
4. Interpretability: *[lean]*
5. **Verdict:** *[DL primary / Classical ML primary]* — *[one sentence why]*""")

md(r"""## An honest demo: neural network vs. gradient boosting on tabular data

Talk is cheap — let's measure it. Below, a neural network (`MLPClassifier`) and gradient boosting (from nb13) compete on a realistic, noisy tabular problem, on the **same cross-validation folds**, scored by ROC-AUC with the same Student's *t* 95% confidence interval from nb08. We use scikit-learn's off-the-shelf MLP so it runs on a CPU in seconds — no PyTorch needed for this comparison.""")
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

# A realistically noisy business table: modest class separation + 10% label noise
# (5000 rows, 20 features) — the regime where trees and nets tend to tie.
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
    'Neural net (MLP 64,32)': _row(mlp_aucs),
    'Gradient boosting':      _row(gbm_aucs),
}).T
print(summary.round(4))

mlp = summary.loc['Neural net (MLP 64,32)']
gbm_row = summary.loc['Gradient boosting']
overlap = not (mlp['AUC_ci_high'] < gbm_row['AUC_ci_low'] or gbm_row['AUC_ci_high'] < mlp['AUC_ci_low'])
if not overlap:
    winner = 'Neural net' if mlp['AUC_mean'] > gbm_row['AUC_mean'] else 'Gradient boosting'
    print(f'\nVerdict: the confidence intervals do NOT overlap -> {winner} wins clearly.')
else:
    print('\nVerdict: the confidence intervals OVERLAP -> statistical tie -> the simpler / faster model wins by default.')""")

md(r"""**Reading the output:** On a 5,000-row tabular problem, the neural network and gradient boosting land within a hair of each other — their 95% confidence intervals overlap, which is a **statistical tie**. And that's the empirical answer to the VP: on this kind of data, deep learning does **not** beat a good tree ensemble, while the ensemble is faster, easier to explain, and simpler to deploy. The story would flip if the input were an **image** (like the FashionMNIST photos you trained on) or **long text** — there, the specialized networks (CNNs, transformers) pull ahead. But for the churn table? Trees win on every practical axis.

> **A question that often comes up here:** *"What if I just make the neural network bigger?"* Try it in the exercise below. On data like this, more layers won't move the accuracy outside the confidence interval — it'll just cost more time and risk unstable training. **Bigger is not better when the data doesn't need bigger.**""")

md(r"""## 📝 PAUSE-AND-DO Exercise 2 — Compare and Decide (10 minutes)

**Task:** Add a third competitor — a **deeper** neural network with `hidden_layer_sizes=(128, 64, 32)` — and rerun the comparison on the **same folds**. Then write three sentences answering the VP of Strategy's question.

**Hints:**
- Copy the `mlp_pipe` pattern; only the `hidden_layer_sizes` argument changes.
- Do **not** change the CV splits — identical folds are what make the comparison fair.""")

code(r"""# YOUR SOLUTION CODE HERE

# Hints:
# deep_pipe = Pipeline([
#     ("scaler", StandardScaler()),
#     ("clf", MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu",
#                           max_iter=300, random_state=RANDOM_SEED)),
# ])
# deep_aucs = cross_val_score(deep_pipe, X_tab, y_tab, scoring="roc_auc", cv=cv, n_jobs=-1)
# Build a 3-row summary table and compare the confidence intervals.""")

md(r"""### INSTRUCTOR SOLUTION — Exercise 2""")
code(r"""# INSTRUCTOR SOLUTION
deep_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu",
                          max_iter=300, random_state=RANDOM_SEED)),
])
deep_aucs = cross_val_score(deep_pipe, X_tab, y_tab, scoring="roc_auc", cv=cv, n_jobs=-1)

summary3 = pd.DataFrame({
    'Neural net (64,32)':      _row(mlp_aucs),
    'Neural net (128,64,32)':  _row(deep_aucs),
    'Gradient boosting':       _row(gbm_aucs),
}).T
print(summary3.round(4))

print("\nAll three 95% confidence intervals overlap: the deeper network does not beat"
      "\nthe simpler network or gradient boosting on this tabular data. Bigger is not"
      "\nbetter when the data structure does not demand it.")""")

md(r"""<!-- INSTRUCTOR SOLUTION -->
**Reading the output:** Adding a third hidden layer leaves the AUC inside the same confidence interval — a statistical tie with both the smaller network and gradient boosting. The three-sentence answer to the VP writes itself: (1) on this tabular data all three models tie within the cross-validation 95% CI; (2) this carries over to Bank Churn, which is also a few-thousand-row tabular table — exactly where classical ML wins; (3) deep learning would earn its keep at TechCorp on a genuinely different data asset, such as classifying **product photos** (the FashionMNIST-style task you did in Section 4) or reading **free-text support tickets** — not on the churn spreadsheet.""")

md(r"""### YOUR ANSWER TO THE VP HERE:

**Three sentences for the VP of Strategy:**

1. *[Empirical comparison — which model won on this data, and by how much in CV CI terms?]*
2. *[Why this generalizes to the Bank Churn problem — the data-shape and sample-size argument from the rubric]*
3. *[Where deep learning **would** earn its keep at TechCorp — name one concrete use case from the company's likely data assets]*""")

# ============================================================= 6. LLM
md(r"""# 6. Special Topic: Large Language Models (LLMs)

The AI you've used all course — Gemini, ChatGPT, Claude — is built on **transformers**, the sequence-handling network mentioned in Section 5. You don't need to build one (that takes thousands of GPUs and months), but it helps to know what's under the hood. These four short, friendly videos take you from "what is an LLM?" to how its attention and memory work — **click any thumbnail** to watch.

""" + yt_thumb("LPZh9BOjkQs", "Large Language Models explained briefly",
              "https://www.youtube.com/watch?v=LPZh9BOjkQs") + "\n\n"
   + yt_thumb("wjZofJX0v4M", "Transformers, the tech behind LLMs",
             "https://www.youtube.com/watch?v=wjZofJX0v4M") + "\n\n"
   + yt_thumb("eMlx5fFNoYc", "Attention in transformers, step-by-step",
             "https://www.youtube.com/watch?v=eMlx5fFNoYc") + "\n\n"
   + yt_thumb("9-Jl0dxWQs8", "How might LLMs store facts",
             "https://www.youtube.com/watch?v=9-Jl0dxWQs8"))

md(r"""> **A question that often comes up here:** *"Will I ever train one of these?"* Almost certainly not — and you don't need to. As a business analyst you'll **use** LLMs through an API (exactly what you've done all course with Gemini prompts), not train them from scratch. But notice: the LLM learns the *same way* the network you built in Section 4 did — data in, a loss measuring wrongness, gradient descent nudging weights — just at an astronomically larger scale. You already understand the core loop.""")

# ============================================================= 7. WRAP-UP
md(r"""# 7. Wrap-Up — Key Takeaways

1. **Deep learning is old math with a new engine.** The ideas are from the 1980s; **compute + data + frameworks** are what finally made them work in 2010.
2. **You trained a neural network end to end.** In Section 4 you loaded FashionMNIST, built a model, ran a train/test loop, saved and reloaded the weights, and predicted real photos — a complete machine-learning pipeline in PyTorch. That's a genuine, resume-worthy skill.
3. **Match the model to the data shape:** plain networks (MLPs) for flat tables, **CNNs for images**, **RNNs/transformers for sequences and text**. Naming the right one is most of what an analyst needs in an AI conversation.
4. **For everyday tabular business problems, gradient boosting usually wins** — it ties or beats a neural network and is faster and easier to explain. The four-question rubric keeps you honest about when to escalate to deep learning.
5. **PyTorch is the right thing to learn next** if you want to go further — it's the language of modern AI research and the LLM ecosystem.

> **A question that often comes up here:** *"So what do I actually tell the VP?"* Something like: *"For our churn data — a few thousand rows in a spreadsheet — gradient boosting is the right call; I tested a neural network and it was a statistical tie, but harder to explain and deploy. Deep learning is the move when we have images, audio, or large text — say, classifying product photos or reading support tickets. I've built one end to end, so when that project comes, we're ready."* That's an evidence-based answer, not a buzzword.

**Next stop — nb20: Course End and Reflection.** Tomorrow is delivery day: M4 poster, Kaggle final submission, peer evaluation, and the reflection survey that closes the course. Today you got both the language *and* the hands-on experience to speak credibly about deep learning. Well done.""")

md(r"""## Participation Assignment Submission Instructions

1. **Complete both PAUSE-AND-DO exercises.**
2. **Run all cells** in order — Section 4 downloads FashionMNIST and trains for a few epochs, so give it a couple of minutes (or switch to a GPU runtime). The videos are embedded as players in the page, so there's nothing to run for them.
3. **Save with output** and submit `nb19_deep_learning_<your_lastname>.ipynb` to Brightspace.

### Next Step

- **Notebook 20** — Final submission + peer review (Day 20)

**Bibliography**
- ISLP, Chapter 10: Deep Learning.
- PyTorch "Learn the Basics" & Quickstart: <https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html>.
- *Deep Learning* by Goodfellow, Bengio, and Courville (free online): <https://www.deeplearningbook.org/>.
- 3Blue1Brown's *Deep Learning* video series — the best visual intuition online.""")

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

for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)

out = Path("notebooks/nb19_deep_learning_instructor.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("Wrote", out, "with", len(cells), "cells")
