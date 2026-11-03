"""Implementation points for Assignment 3 Part 1.

Edit this file while working through `PytorchTraining.ipynb`. The autograder
imports these functions/classes directly, so keep the public names unchanged.
"""

import torch
from torch import nn


def make_three_layer_convnet(input_shape=(1, 28, 28), num_classes=3):
    """Return the ConvNet required in Part 1(a).

    Architecture:
      conv 5x5, 32 filters, padding 2, bias
      ReLU
      conv 3x3, 16 filters, padding 1, bias
      ReLU
      max pool 2x2 stride 2
      flatten
      fully connected layer to `num_classes`
    """

    in_channels, height, width = input_shape
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=5, padding=2, bias=True),
        nn.ReLU(),
        nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=True),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * (height // 2) * (width // 2), num_classes, bias=True)
    )


class Tokenizer(nn.Module):
    """Convert a batch of images into a sequence of patch tokens."""

    def __init__(self, in_channels=3, patch_size=8, embed_dim=64):
        super().__init__()
        self.T = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x):
        """Return tokens with shape B x num_patches x embed_dim."""
        x = self.T(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class SingleHeadSelfAttention(nn.Module):
    """Single-headed scaled dot-product self-attention."""

    def __init__(self, embed_dim):
        super().__init__()
        self.token_dim = embed_dim
        self.Q = nn.Linear(embed_dim, embed_dim)
        self.K = nn.Linear(embed_dim, embed_dim)
        self.V = nn.Linear(embed_dim, embed_dim)

    def forward(self, tokens, return_attn=False):
        """Apply self-attention to B x num_tokens x embed_dim tokens."""
        Q = self.Q(tokens)
        K = self.K(tokens)
        V = self.V(tokens)

        d = self.token_dim
        attn = Q @ K.transpose(-2, -1) / (d ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        T = attn @ V
        if return_attn:
            return T, attn
        return T


class TokenwiseMLP(nn.Module):
    """Apply a 1x1 convolution independently to each token."""

    def __init__(self, token_dim):
        super().__init__()
        self.token_dim = token_dim
        self.W = nn.Conv1d(token_dim, token_dim, kernel_size=1)

    def forward(self, tokens):
        """Return a tensor with the same shape as `tokens`."""
        x_trans = tokens.transpose(1, 2)
        conv = self.W(x_trans)
        mlp_out = conv.transpose(1, 2)
        return mlp_out


def per_class_accuracy(logits, labels, num_classes):
    """Return per-class accuracy as percentages."""
    preds = logits.argmax(dim=1)
    labels = labels.reshape(-1)
    correct = [0] * num_classes
    total = [0] * num_classes

    y_list = labels.detach().cpu().tolist()
    preds_list = preds.detach().cpu().tolist()
    for i in range(len(y_list)):
        true_label = y_list[i]
        pred_label = preds_list[i]
        total[true_label] += 1
        if true_label == pred_label:
            correct[true_label] += 1

    accs = []
    for c in range(num_classes):
        accs.append(100 * correct[c] / total[c] if total[c] > 0 else 0.0)

    return torch.tensor(accs, dtype=torch.float32, device=logits.device)
