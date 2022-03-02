import numpy as np
import functools
from scipy.signal import find_peaks
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR



def calculate_autocorrelation(x,i,level_of_significance_for_pearson,consider_only_significant_correlation):
    # Auxiliary function for autocor.
    # Calculated the autocorrelation function for the vector x for a given shift given in indices of the vector x
    # output: r , p, value autocorrelation function, shift in index of vector
    if i == 0:
        return 1,0,1,0
    else:
        r, p = stats.pearsonr(x[i:], x[:-i])
        if np.isnan(r):
            return 0,0,0,i
        # consider only correlations that are significant concerning the p-value for the corresponding shift if only significant correlations are supposed to be taken selectable via consider_only_significant_correlation
        if p > level_of_significance_for_pearson and (consider_only_significant_correlation==1):
            return r,p,0,i
        else:
            return r,p,r,i

def autocor(x,list_of_lags,level_of_significance_for_pearson,consider_only_significant_correlation):
    # Calculate the autocorrelation function for a time series given via the vector x at the shifts as indices for the vector x given in the list of lags list_of_lags
    list_of_results=list(zip(*list(map(functools.partial(calculate_autocorrelation,x,level_of_significance_for_pearson=level_of_significance_for_pearson,consider_only_significant_correlation=consider_only_significant_correlation),list_of_lags))))
    list_of_results
    r_list=list(list_of_results[0])
    p_list=list(list_of_results[1])
    func=list(list_of_results[2])
    lag_list=list(list_of_results[3])
    return r_list, p_list, func, lag_list

def shift_diff(i, corfunc):
    # Calculates the L^1 norm of the difference between the autocorrelation and its shifted version for a given shift as the index of the vector corfunc
    if i==0:
        return 0  # Zero because the subtraction without no shift is subtracting the same function
    else:
        w1 = np.array(corfunc)[i:]
        w2 = np.array(corfunc)[:-i]
        return float(sum(abs(w1 - w2)) / w1.size)

def fit_model(df_data_aggregated):
    # The routine to fit a model based on the periodic information/phase of a date concerning the period to the original time series to test the hypothesis how well the suggested period fits the original time series
    X = df_data_aggregated["date_modulo"].to_numpy().reshape(df_data_aggregated["date_modulo"].size, 1)
    y = df_data_aggregated["value"].to_numpy().reshape(df_data_aggregated["value"].size)
    # In principle any model here is possible, basically we only need to test if there exists one model such that based on the suggested period the correlation at the relevant sites vanishes.
    mlp = RandomForestRegressor()

    mlp.fit(X, y)
    y_model = mlp.predict(X).reshape(X.size, 1)
    return y_model, mlp

def get_relevant_diffs(diffs):
    # Get position of the local minima of the function mapping the L^1 norm of the difference between the autocorrelation function and its shifted versions against the shifts.
    # Returning its positions in peaks as indices of the vector diffs and the corresponding values of the local minimum is due to small terms being subtracted or due to a high self-similarity of autocorrelation function with its shifted version.
    stop_calculation=0
    peaks, _ = find_peaks(-np.array(diffs))
    if peaks.size <= 0:
        print('No local minima in the inner domain of the function mapping the L^1 norm of the difference between the autocorrelation function and its shifted versions against the shifts! No period detection possible!')
        stop_calculation = 1
    # For shift=0, two identical functions are subtracted resulting in a local minimum at shift 0 in the function mapping the L^1 norm of the difference between the autocorrelation function and its shifted versions against the shifts.
    # Needs to be inserted separately since find_peaks searches only in the inne of the domain, consequently no boundary values are considered
    peaks = np.insert(peaks, 0, 0)
    peaks = peaks.astype(int)
    return np.array(diffs)[peaks],peaks,stop_calculation

def sum_shifted_function(i, corfunc):
    # Calculates the sum of the L^1 norm of the correlation function and its shifted version for a given shift in the form as they are subtraced in the function shift_diffs.
    # The purpose is to test if already this sum is smaller then
    if i==0:
        return 2*sum(abs(np.array(corfunc))) / np.array(corfunc).size
    else:
        w1 = np.array(corfunc[i:])
        w2 = np.array(corfunc[:-i])
        return sum(abs(w1))/w1.size + sum(abs(w2))/w2.size