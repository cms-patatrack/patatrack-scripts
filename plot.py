import matplotlib.pyplot as plt

def bar_graph(X_axis, Y_axis, fig_name=None, save_fig=False, show_fig=True):

    if len(X_axis) == 0 or len(Y_axis) == 0:

        print("\nOne of the lists that you have provided is  empty.")
        return
    
    plt.bar(X_axis, Y_axis)

    if show_fig:
        plt.show()

    if save_fig:
        plt.savefig(f"./plots/{fig_name}.png")

    plt.close()

def line_graph(X_axis, Y_axis, fig_name=None, save_fig=False, show_fig=True):

    if len(X_axis) == 0 or len(Y_axis) == 0:

        print("\nOne of the lists that you have provided is empty.")
        return
    
    plt.plot(X_axis, Y_axis)

    if show_fig:
        plt.show()
    
    if save_fig:
        plt.savefig(f"./plots/{fig_name}.png")

    plt.close()
