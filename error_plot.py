"""
function that shows significant plot for the analysis of the solution:
plot_truesol_approxsol   :   plot of the true value of the unknowns and the estimated value of the unknowns
comparison_plot          :   plot of the comparison of the true and approx solution, the aim is to get near to a 45°line
error_plot               :   plot of the errors and its decreasing (we hope) within the epochs of training
"""


import matplotlib.pyplot as plt

path = "fig/"


def plot_truesol_approxsol(sol, prev, top):
    plt.plot(sol.detach().numpy(), 'ro', label='true solution')
    plt.plot(prev.detach().numpy(), 'go', label='nn solution')
    plt.ylabel('C_t')
    plt.xlabel('samples')
    plt.title('Graph of samples and their approx')
    plt.legend()

    plt.savefig(path + "model_" + str(top + 1) + "_comparison_truesol_approxsol.png")
    plt.close()


def comparison_plot(sol, prev, top):
    plt.scatter(sol.detach().numpy(), prev.detach().numpy())
    plt.plot(sol, sol)
    plt.xlabel('C_t in situ')
    plt.ylabel('C_t CANYON-MED')
    plt.title('Comparison between true solution and approx')
    plt.legend()

    plt.savefig(path + "model_" + str(top + 1) + "_comparison_plot.png")
    plt.close()


def error_plot(ep, losses, top):
    ep_vect = [i for i in range(ep)]
    plt.plot(ep_vect, losses, label='loss during epochs')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Errors of Bayesian MLP during epochs')

    plt.savefig(path + "model_" + str(top + 1) + "_error_plot.png")
    plt.close()


def get_all_plot(sol, prev, ep, losses, top):
    plot_truesol_approxsol(sol, prev, top)
    comparison_plot(sol, prev, top)
    #error_plot(ep, losses, top)
