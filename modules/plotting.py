import numpy as np
import matplotlib.pyplot as plt
from modules.config import get_parameter
from modules.simulation import run_monte_carlo
from modules.estimator import compute_cramer_rao_bound

def configure_plot_style(ax, xlabel=None, ylabel=None, xticks=None, yticks=None, ylim=None, legend_labels=None, legend_loc='best', legend_ncol=1):
    """
    Applies common plotting styles to a matplotlib axes object.

    Args:
        ax (matplotlib.axes.Axes): The axes object to configure.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
        legend_labels (list of str, optional): Labels for the legend.
        xticks (list or np.ndarray, optional): Custom x-axis tick locations.
        yticks (list or np.ndarray, optional): Custom y-axis tick locations.
        ylim (tuple, optional): y-axis limits as (ymin, ymax).
        legend_loc (str, optional): Location of the legend (default 'best').
    """
    label_fontsize = 10
    tick_fontsize = 7
    legend_fontsize = 9

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    if legend_labels:
        ax.legend(
            legend_labels,
            fontsize=legend_fontsize,
            frameon=False,
            loc=legend_loc,
            ncol=legend_ncol,
            handlelength=1.5,
            handletextpad=0.5,
            columnspacing=0.5,
            borderaxespad=0.3,  # space between legend and axes
            labelspacing=0.2    # vertical space between legend entries (rows)
        )


def plot_figure(i_figure):
    ieeetran_column_width = 3.45
    ieeetran_page_width = 7.16
    if i_figure == '5':
        my_width = ieeetran_page_width/3
        algorithm_parameter, simulation_parameter = get_parameter(i_figure)
        mse, mse_wrapping, mse_no_wrapping, probability_wrapping = run_monte_carlo(algorithm_parameter, simulation_parameter)
        mse_crb = compute_cramer_rao_bound(algorithm_parameter['signal_basis'], simulation_parameter['snrdb'])
        
        # Plot Figure 5a
        plt.figure(figsize=(my_width, my_width))
        plt.semilogy(simulation_parameter['snrdb'], mse, 'k-+', linewidth=1)
        plt.semilogy(simulation_parameter['snrdb'], mse_wrapping, 'k--', linewidth=1, markersize=2)
        plt.semilogy(simulation_parameter['snrdb'], mse_no_wrapping, 'k:', linewidth=1)
        plt.semilogy(simulation_parameter['snrdb'], mse_crb, 'k-', linewidth=1)
        ax = plt.gca()
        configure_plot_style(
            ax,
            xlabel='SNR [dB]',
            ylabel='Reconstruction MSE',
            legend_labels=['Overall', 'Wrapping', 'No wrapping', 'CRB'],
            xticks=np.arange(0, 11, 5),
            ylim=(1e-3, None),
            legend_loc='lower left'
        )
        plt.tight_layout()

        # Plot Figure 5b
        plt.figure(figsize=(my_width, my_width))
        plt.semilogy(simulation_parameter['snrdb'], probability_wrapping, 'k--', linewidth=1, markersize=2)
        ax = plt.gca()
        configure_plot_style(
            ax,
            xlabel='SNR [dB]',
            ylabel='Probability',
            xticks=np.arange(0, 11, 5)
        )
        plt.tight_layout()

        # Plot Figure 5c
        proportion_wrapping = (mse_wrapping * probability_wrapping)/(mse_wrapping * probability_wrapping + mse_no_wrapping * (1 - probability_wrapping))
        proportion_no_wrapping = 1 - proportion_wrapping
        plt.figure(figsize=(my_width, my_width))
        plt.plot(simulation_parameter['snrdb'], proportion_wrapping, 'k--', linewidth=1, markersize=2)
        plt.plot(simulation_parameter['snrdb'], proportion_no_wrapping, 'k:', linewidth=1, markersize=2)
        ax = plt.gca()
        configure_plot_style(
            ax,
            xlabel='SNR [dB]',
            ylabel='Proportion',
            xticks=np.arange(0, 11, 5)
        )
        plt.tight_layout()
        
    elif i_figure == '6':
        my_width = ieeetran_column_width
        plt.figure(figsize=(my_width, my_width))
        for i_curve in range(4):
            algorithm_parameter, simulation_parameter = get_parameter(i_figure, i_curve)
            mse = run_monte_carlo(algorithm_parameter, simulation_parameter)
            plt.semilogy(simulation_parameter['snrdb'], mse, color='k', marker='+', markersize=5, markerfacecolor='k', linewidth=1)
        mse_crb = compute_cramer_rao_bound(algorithm_parameter['signal_basis'], simulation_parameter['snrdb'])
        plt.semilogy(simulation_parameter['snrdb'], mse_crb, color='k', linewidth=1)
        ax = plt.gca()
        configure_plot_style(
            ax,
            xlabel='SNR [dB]',
            ylabel='Reconstruction MSE',
            legend_labels=['Alg. 3', '_nolegend_', '_nolegend_', '_nolegend_', 'CRB'],
            xticks=np.arange(0, 31, 10)
        )
        plt.tight_layout()

    elif i_figure in ['9a', '9b', '9c']:
        my_width = ieeetran_page_width/3
        plt.figure(figsize=(my_width, my_width))
        line_styles = ['k-', 'k-', 'k--', 'k-.']
        markers = ['+', '.', '', '']
        for i_curve in range(4):
            if i_curve == 3:
                # Add an invisible curve as a placeholder for legend alignment
                plt.semilogy([], [], color='none', label='')
            algorithm_parameter, simulation_parameter = get_parameter(i_figure, i_curve)
            mse = run_monte_carlo(algorithm_parameter, simulation_parameter)
            plt.semilogy(
            simulation_parameter['snrdb'],
            mse,
            line_styles[i_curve],
            marker=markers[i_curve],
            markersize=5,
            linewidth=1
            )

        mse_crb = compute_cramer_rao_bound(algorithm_parameter['signal_basis'], simulation_parameter['snrdb'])
        plt.semilogy(simulation_parameter['snrdb'], mse_crb, color='k', linewidth=1)
        ax = plt.gca()
        legend_labels = ['Linear', 'Circular', 'Kay', '', 'L&W', 'CRB'] if i_figure == '9a' else None
        ylim = (1e-5, None) if i_figure == '9a' else None
        configure_plot_style(
            ax,
            xlabel='SNR [dB]',
            ylabel='Reconstruction MSE',
            legend_labels=legend_labels,
            xticks=np.arange(0, 31, 10),
            ylim=ylim,
            legend_loc='lower left',
            legend_ncol=2
        )
        plt.tight_layout()

    elif i_figure in ['10a', '10b', '10c', '10d', '10e', '10f', '10g', '10h', '10i', '10j', '10k', '10l']:
        my_width = ieeetran_page_width/3
        plt.figure(figsize=(my_width, my_width))
        for i_curve in range(4):
            algorithm_parameter, simulation_parameter = get_parameter(i_figure, i_curve)
            mse = run_monte_carlo(algorithm_parameter, simulation_parameter)
            plt.semilogy(simulation_parameter['snrdb'], mse, color='k', marker='.', markersize=5, markerfacecolor='k', linewidth=1)
        mse_crb = compute_cramer_rao_bound(algorithm_parameter['signal_basis'], simulation_parameter['snrdb'])
        plt.semilogy(simulation_parameter['snrdb'], mse_crb, color='k', linewidth=1)
        ax = plt.gca()
        legend_labels = ['Circular', '_nolegend_', '_nolegend_', '_nolegend_', 'CRB'] if i_figure == '10a' else None
        configure_plot_style(
            ax,
            xlabel='SNR [dB]',
            ylabel='Reconstruction MSE',
            legend_labels=legend_labels,
            xticks=np.arange(0, 31, 10)
        )
        plt.tight_layout()
    
    elif i_figure in ['11']:
        my_width = ieeetran_column_width
        plt.figure(figsize=(my_width, my_width))
        for i_curve in range(4):
            algorithm_parameter, simulation_parameter = get_parameter(i_figure, i_curve)
            algorithm_parameter['is_naive'] = True
            mse = run_monte_carlo(algorithm_parameter, simulation_parameter)
            plt.semilogy(simulation_parameter['snrdb'], mse, color='k', marker='.', markersize=5, markerfacecolor='k', linewidth=1)
        for i_curve in range(4):
            algorithm_parameter, simulation_parameter = get_parameter(i_figure, i_curve)
            algorithm_parameter['is_naive'] = False
            mse = run_monte_carlo(algorithm_parameter, simulation_parameter)
            plt.semilogy(simulation_parameter['snrdb'], mse, color='k', marker='*', markersize=5, markerfacecolor='k', linewidth=1)
        mse_crb = compute_cramer_rao_bound(algorithm_parameter['signal_basis'], simulation_parameter['snrdb'])
        plt.semilogy(simulation_parameter['snrdb'], mse_crb, color='k', linewidth=1)
        ax = plt.gca()
        legend_labels = ['Naïve', '_nolegend_', '_nolegend_', '_nolegend_', 'Proposed', '_nolegend_', '_nolegend_', '_nolegend_', 'CRB']
        configure_plot_style(
            ax,
            xlabel='SNR [dB]',
            ylabel='Reconstruction MSE',
            legend_labels=legend_labels,
            xticks=np.arange(0, 31, 10)
        )
        plt.tight_layout()

    elif i_figure in ['12a', '12b', '12c']:
        my_width = ieeetran_page_width/3
        plt.figure(figsize=(my_width, my_width))
        for i_curve in range(5):
            algorithm_parameter, simulation_parameter = get_parameter(i_figure, i_curve)
            mse = run_monte_carlo(algorithm_parameter, simulation_parameter)
            plt.semilogy(simulation_parameter['snrdb'], mse, color='k', marker='.', markersize=5, markerfacecolor='k', linewidth=1)
        mse_crb = compute_cramer_rao_bound(algorithm_parameter['signal_basis'], simulation_parameter['snrdb'])
        plt.semilogy(simulation_parameter['snrdb'], mse_crb, color='k', linewidth=1)
        ax = plt.gca()
        legend_labels = ['Lag', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', 'CRB'] if i_figure == '12a' else None
        configure_plot_style(
            ax,
            xlabel='SNR [dB]',
            ylabel='Reconstruction MSE',
            legend_labels=legend_labels,
            xticks=np.arange(-10, 31, 10)
        )
        plt.tight_layout()

    elif i_figure in ['13a', '13b', '13c']:
        my_width = ieeetran_page_width/3
        plt.figure(figsize=(my_width, my_width))

        # First curve (single lag)
        algorithm_parameter, simulation_parameter = get_parameter(i_figure, 0)
        mse = run_monte_carlo(algorithm_parameter, simulation_parameter)
        plt.semilogy(simulation_parameter['snrdb'], mse, color='k', marker='.', markersize=5, markerfacecolor='k', linewidth=1)
        # Second curve (multiple lags)
        algorithm_parameter, simulation_parameter = get_parameter(i_figure, 1)
        mse = run_monte_carlo(algorithm_parameter, simulation_parameter)
        plt.semilogy(simulation_parameter['snrdb'], mse, color='k', marker='*', markersize=5, markerfacecolor='k', linewidth=1)

        mse_crb = compute_cramer_rao_bound(algorithm_parameter['signal_basis'], simulation_parameter['snrdb'])
        plt.semilogy(simulation_parameter['snrdb'], mse_crb, color='k', linewidth=1)
        ax = plt.gca()
        legend_labels = ['Single lag', 'Multiple lags', 'CRB'] if i_figure == '13a' else None
        configure_plot_style(
            ax,
            xlabel='SNR [dB]',
            ylabel='Reconstruction MSE',
            legend_labels=legend_labels,
            xticks=np.arange(-10, 31, 10)
        )
        plt.tight_layout()

    elif i_figure in ['14a', '14b', '14c']:
        my_width = ieeetran_page_width/3
        plt.figure(figsize=(my_width, my_width))
        
        algorithm_parameter, simulation_parameter = get_parameter(i_figure, i_curve=0)
        mse = run_monte_carlo(algorithm_parameter, simulation_parameter)
        plt.semilogy(simulation_parameter['snrdb'], mse, color='k', marker='.', markersize=5, markerfacecolor='k', linewidth=1)
        
        algorithm_parameter, simulation_parameter = get_parameter(i_figure, i_curve=1)
        mse = run_monte_carlo(algorithm_parameter, simulation_parameter)
        plt.semilogy(simulation_parameter['snrdb'], mse, color='k', marker='*', markersize=5, markerfacecolor='k', linewidth=1)
        
        mse_crb = compute_cramer_rao_bound(algorithm_parameter['signal_basis'], simulation_parameter['snrdb'])
        plt.semilogy(simulation_parameter['snrdb'], mse_crb, color='k', linewidth=1)

        ax = plt.gca()
        legend_labels = ['Single lag', 'Multiple lags', 'CRB'] if i_figure == '14a' else None
        configure_plot_style(
            ax,
            xlabel='SNR [dB]',
            ylabel='Reconstruction MSE',
            legend_labels=legend_labels,
            xticks=np.arange(-20, 21, 10)
        )
        plt.tight_layout()
    
    elif i_figure in ['15a', '15d']:
        my_width = ieeetran_page_width/3
        plt.figure(figsize=(my_width, my_width))
        
        algorithm_parameter, simulation_parameter = get_parameter(i_figure, i_curve=0)
        mse = run_monte_carlo(algorithm_parameter, simulation_parameter)
        plt.semilogy(simulation_parameter['snrdb'], mse, color='k', marker='.', markersize=5, linewidth=1)

        algorithm_parameter, simulation_parameter = get_parameter(i_figure, i_curve=1)
        mse = run_monte_carlo(algorithm_parameter, simulation_parameter)
        plt.semilogy(simulation_parameter['snrdb'], mse, color='k', linestyle='--', linewidth=1)

        mse_crb = compute_cramer_rao_bound(algorithm_parameter['signal_basis'], simulation_parameter['snrdb'])
        plt.semilogy(simulation_parameter['snrdb'], mse_crb, color='k', linewidth=1)

        ax = plt.gca()
        legend_labels = ['Proposed', 'DPT', 'CRB'] if i_figure == '15a' else None
        configure_plot_style(
            ax,
            xlabel='SNR [dB]',
            ylabel='Reconstruction MSE',
            legend_labels=legend_labels,
            xticks=np.arange(0, 31, 10)
        )
        plt.tight_layout()
    
    elif i_figure in ['15b', '15c', '15e', '15f']:
        my_width = ieeetran_page_width/3
        plt.figure(figsize=(my_width, my_width))
        
        algorithm_parameter, simulation_parameter = get_parameter(i_figure, i_curve=0)
        mse = run_monte_carlo(algorithm_parameter, simulation_parameter)
        plt.semilogy(simulation_parameter['snrdb'], mse, color='k', marker='.', markersize=5, linewidth=1)

        algorithm_parameter, simulation_parameter = get_parameter(i_figure, i_curve=1)
        mse = run_monte_carlo(algorithm_parameter, simulation_parameter)
        plt.semilogy(simulation_parameter['snrdb'], mse, color='k', linestyle='--', linewidth=1)

        algorithm_parameter, simulation_parameter = get_parameter(i_figure, i_curve=2)
        mse = run_monte_carlo(algorithm_parameter, simulation_parameter)
        plt.semilogy(simulation_parameter['snrdb'], mse, color='k', linestyle='--', linewidth=1)

        mse_crb = compute_cramer_rao_bound(algorithm_parameter['signal_basis'], simulation_parameter['snrdb'])
        plt.semilogy(simulation_parameter['snrdb'], mse_crb, color='k', linewidth=1)

        ax = plt.gca()
        configure_plot_style(
            ax,
            xlabel='SNR [dB]',
            ylabel='Reconstruction MSE',
            xticks=np.arange(0, 31, 10)
        )
        plt.tight_layout()

    else:
        print(f"No specific execution path defined for figure: {i_figure}")
