"""
This file contains the 10 best topology for the network, according to the article.
An operation of selection of how many topology must be consider is performed here.
"""

i = 8  # input number
best_topo_all = [[i, 31, 23, 1], [i, 20, 8, 1], [i, 18, 13, 1] , [i, 35, 21, 1], [i, 35, 9, 1], [i, 36, 21, 1],
                 [i, 39, 26, 1], [i, 44, 27, 1], [i, 47, 21, 1], [i, 47, 29, 1]]


def top_select(number_of_selected_topology):
    best_topo = best_topo_all[0:number_of_selected_topology]
    return best_topo


number_of_selected_topology = 10
best_topo = top_select(number_of_selected_topology)
