"""
main code of the canyon-b implementation
Reading and preparation of the dataset, training of the Bayesian MLP and plot of the approximated solutions
"""

import torch.optim as optim
from error_plot import *
from preparation_data import *
from MLP_Bayesian_NN import MLP_Bayesian
from train import *
from validation import *
from topology import best_topo

data, target, data_validation, target_validation = preparation_function("CT", "tcarbn")
target.resize_(4145, 1)
target_validation.resize_(1033, 1)
n = len(data_validation)

print(data[0, :])
models = list()
models_loss = []
epoch = 5000

for top in range(len(best_topo)):
    model_mlp = MLP_Bayesian(top)
    models.append(model_mlp)
    optimizer = optim.Adam(model_mlp.parameters(), lr=0.0001)  # , momentum=0.5)
    train(model_mlp, epoch, data, target, top, optimizer)

# model = models[0]
# result = model(data)
losses = mylosses()

for top in range(len(best_topo)):
    model_selected = models[top]
    result_validation = model_selected(data_validation)  # result = model_selected(data)
    get_all_plot(target_validation, result_validation, epoch, losses, top)

mae_list = []
rmse_list = []
output_validation_list = []

for top in range(len(best_topo)):
    model_selected = models[top]
    output_validation, mae, rmse = validation_MAE_RMSE(data_validation, target_validation, model_selected)

    output_validation_list.append(output_validation)
    mae_list.append(mae)
    rmse_list.append(rmse)
    print(f"[MODEL]: {top + 1}, [RMSE]: {rmse}, [MAE]: {mae}")
