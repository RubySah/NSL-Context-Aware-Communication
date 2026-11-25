# src/training/calibrate.py
"""Fit one temperature on validation logits so confidence matches accuracy."""

import torch
import torch.nn.functional as F


def fit_temperature(logits, labels):
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp())