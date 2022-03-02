import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from auxiliary_funcs import autocor


def plot_with_period(df_data_aggregated, diffs, other_tolerances, best_tolerance, lag_list, r_list, p_list, corfunc, signal_data, model_data, norm_diff_between_singal_and_model, norm_model, norm_signal, plot_tolerances,level_of_significance_for_pearson,consider_only_significant_correlation, minimum_number_of_datapoints_for_correlation_test):
    # Plots and parameter outputs in case at least one period could be suggested
    signal_subtracted_model = signal_data - model_data
    df_data_difference_signal_model = pd.DataFrame(data=signal_subtracted_model, columns=["value"])
    r_list_diff, p_list_diff, corfunc_diff, lag_list_diff = autocor(df_data_difference_signal_model["value"], list(range(0, int((df_data_difference_signal_model["value"].size) - minimum_number_of_datapoints_for_correlation_test))),level_of_significance_for_pearson,consider_only_significant_correlation)
    print('Norm of difference between signal and model: ' + str(norm_diff_between_singal_and_model))
    print('Norm model: ' + str(norm_model))
    print('Norm signal: ' + str(norm_signal))

    #plt.plot(df_data_aggregated["date"], df_data_aggregated["value"], label="signal")
    #plt.plot(df_data_aggregated["date"], model_data, label="model")
    plt.plot(df_data_aggregated["value"], label="data")
    plt.plot(model_data, label="model")
    plt.legend()
    plt.title("Data and fitted model")
    plt.xlabel('Time in time points')
    plt.savefig('data_model_trend.png',format='png')
    plt.show()

    plt.plot(list(np.array(diffs)))
    plt.title('Norm of difference between unshited and shifted autocorrelation function')
    plt.xlabel('Shift in time points')
    if plot_tolerances == 1:
        plt.hlines(y=other_tolerances, xmin=0, xmax=len(diffs), colors='green')
        plt.hlines(y=[best_tolerance], xmin=0, xmax=len(diffs), colors='red')
    plt.savefig('shift_acf_trend.png',format='png')
    plt.show()

    plt.plot(lag_list, r_list, label="r")
    plt.plot(lag_list, p_list, label="p")
    plt.plot(lag_list, corfunc, label="acf")
    plt.legend()
    plt.title("Autocorrelation function of data")
    plt.xlabel('Shift of the data in time points')
    plt.savefig('acf_trend.png',format='png')
    plt.show()

    plt.plot(df_data_difference_signal_model["value"])
    plt.title("Difference of data and model")
    plt.show()

    plt.plot(lag_list_diff, r_list_diff, label="r")
    plt.plot(lag_list_diff, p_list_diff, label="p")
    plt.plot(lag_list_diff, corfunc_diff, label="acf")
    plt.legend()
    plt.title("Autocorrelation function of data minus model")
    plt.xlabel('Shift of the data minus model in time points')
    plt.savefig('acf_data_model_trend.png',format='png')
    plt.show()

def plot_without_period(df_data_aggregated, diffs, lag_list, r_list, p_list, corfunc):
    # Plots and parameter outputs in case no period could be suggested

    #plt.plot(df_data_aggregated["date"], df_data_aggregated["value"], label="signal")
    plt.plot(df_data_aggregated["value"], label="data")
    plt.legend()
    plt.title("time series data")
    plt.show()

    plt.plot(list(np.array(diffs)))
    plt.title('Norm of difference between unshited and shifted autocorrelation function')
    plt.xlabel('Shift')
    plt.show()

    plt.plot(lag_list, r_list, label="r")
    plt.plot(lag_list, p_list, label="p")
    plt.plot(lag_list, corfunc, label="acfunc")
    plt.xlabel('Time in data points')
    plt.legend()
    plt.title("Autocorrelation function of data")
    plt.show()
