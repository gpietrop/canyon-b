"""
main code of the canyon-b implementation
Reading and preparation of the dataset, training of the Bayesian MLP and plot of the approximated solutions
"""

import torch.optim as optim
from error_plot import *
from preparation_data import *
from MLP_Bayesian_NN import MLP_Bayesian
from train import *
from topology import best_topo

data, target = preparation_function("CT", "tcarbn")
target.resize_(4145, 1)

print(data[0,:])
models = list()
models_loss = []
epoch = 5000

for top in range(len(best_topo)):
    model_mlp = MLP_Bayesian(top)
    models.append(model_mlp)
    optimizer = optim.Adam(model_mlp.parameters(), lr=0.0001)  # , momentum=0.5)
    train(model_mlp, epoch, data, target, top, optimizer)

#model = models[0]
#result = model(data)


losses = mylosses()

for top in range(len(best_topo)):
    model_selected = models[top]
    result = model_selected(data)
    get_all_plot(target, result, epoch, losses, top)
