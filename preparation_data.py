"""
reading_data(element)                                  : as input must be inserted the unknown that we want to estimate,
                                                         the corrispondent .csv file with all the data is read
split_into_dataset_and_validation(data)                : the data are splitted into training set (used for training the
                                                         network) and validation set (used for testing the quality of
                                                         the network)
get_decimal_year(dataset)                              : the first input of the date is transformed in a decimal year
get_prespivot(presgrid)                                : get the prespivot matrix, used to prepare the data relatives
                                                         to time
get_data_target_prespivot(element, name_column_element): read the data, split it into dataset and validation set, fix
                                                         input and output
preparation_dataset(data, prespivot)                   : the data are adjusted in order to improve computation
norm_fun_data(data)                                       : normalization of the input data
preparation_function(element, name_column_element)     : application of all the functions above
"""

import numpy as np
import pandas as pd
import torch


def reading_data(element):
    data = pd.read_csv("dataset/data_" + element + ".csv")
    return data


mont_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10,
             'Nov': 11, 'Dec': 12}


def split_into_dataset_and_validation(data):
    dataset = data[data.categ == 'training']
    validation = data[data.categ == 'validation']
    return dataset, validation


def get_decimal_year(dataset):
    for i in range(len(dataset[:, 0])):
        dataset[i, 0] = int(dataset[i, 0][7:11]) + (mont_dict[dataset[i, 0][3:6]]) / 12  # fix data input into
        # decimal year


def get_data_target(element, name_column_element):
    data = reading_data(element)
    dataset, validation = split_into_dataset_and_validation(data)

    out_d = dataset[str(name_column_element)].to_numpy()
    in_d = dataset[['date', 'latitude', 'longitude', 'longitude', 'pres', 'temp', 'doxy', 'psal']].to_numpy()
    out_v = validation[str(name_column_element)].to_numpy()
    in_v = validation[['date', 'latitude', 'longitude', 'longitude', 'pres', 'temp', 'doxy', 'psal']].to_numpy()

    get_decimal_year(in_d)
    get_decimal_year(in_v)
    in_d = in_d.astype('float64')
    in_v = in_v.astype('float64')

    data = torch.from_numpy(in_d)
    target = torch.from_numpy(out_d)
    data = data.float()
    target = target.float()
    return data, target


def preparation_dataset(data):
    for i in range(len(data[:, 0])):  # iteration on the rows (i.e. the samples)
        data[i, 2] = np.abs(1 - np.mod(data[i, 2] - 110, 360) / 180)  # fix longitude input
        data[i, 3] = np.abs(1 - np.mod(data[i, 3] - 20, 360) / 180)  # fix longitude input
        data[i, 4] = data[i, 4] / 20000 + (1 / ((1 + np.exp(-data[i, 4]/300)) ** 3))  # fix pressure input

    return data


def norm_fun_data(data):
    mean = [data[:, i].mean() for i in range(data.size()[1])]  # mean of the columns
    std = [data[:, i].std() for i in range(data.size()[1])]  # std of the columns
    for i in range(data.size()[1]):  # iterations over the columns
        data[:, i] = 2 / 3 * (data[:, i] - mean[i]) / std[i]

    return data, mean, std

def norm_fun_target(target):
    mean, std = target.mean(), target.std()
    target = 2 / 3 * (target - mean) / std

    return target, mean, std


def preparation_function(element, name_column_element):
    data, target = get_data_target(element, name_column_element)
    data = preparation_dataset(data)
    data, mean_data, std_data = norm_fun_data(data)
    target, mean_target, std_target = norm_fun_target(target)

    return data, target
