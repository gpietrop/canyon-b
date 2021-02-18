"""
definition of the train function of the neural network
"""

import torch
from IPython import display

losses = []


def train(model, ep, data, target, top, optimizer):

    for t in range(ep):
        output = model(data)
        criterion = torch.nn.L1Loss()  # criterion=torch.nn.L1Loss()
        loss = criterion(output, target)
        losses.append(loss)

        print(f"[MODEL]: {top + 1}, [EPOCH]: {t}, [LOSS]: {loss.item():.6f}")
        display.clear_output(wait=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def mylosses():
    return losses
