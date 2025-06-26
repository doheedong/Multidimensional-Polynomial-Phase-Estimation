import matplotlib.pyplot as plt
from modules.plotting import plot_figure

def main():
    """
    Plots the figures based on the specified figure number.
    Figure list: '5', '6', '9a', '9b', '9c',
                 '10a', '10b', '10c', '10d', '10e', '10f', '10g', '10h', '10i', '10j', '10k', '10l',
                 '11', '12a', '12b', '12c', '13a', '13b', '13c', '14a', '14b', '14c'.
    """

    figure_list = ['15f']
    for i_figure in figure_list:
        plot_figure(i_figure)
    plt.show()

if __name__ == "__main__":
    main()