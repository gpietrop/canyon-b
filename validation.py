"""
definition of the two different function for the computation of the effectiveness of the method over the validation set,
i.e. the aim is to test if our method has capabilities also over new samples
"""
import torch
import numpy as np


def validation_MAE_RMSE(data_validation, target_validation, model_selected):
    output_validation = model_selected(data_validation)
    rmse = torch.square(1 / len(data_validation) * (torch.sum((output_validation - target_validation) ** 2)))
    mae = (1 / len(data_validation)) * torch.sum(torch.abs((output_validation - target_validation)))
    return output_validation, mae, rmse
