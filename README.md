# Period-Detection

If you use the presented period detection method or the provided Python scripts inspired you for further extensions or variations of this framework, we’ll be happy if you cite our paper “On a method for detecting periods and repeating patterns in time series data with autocorrelation and function approximation” (https://doi.org/10.1016/j.patcog.2023.109355) in course of which the Python implementations of this git repository have been worked out.

## Installation
pip install...

## Usage

```Python
from find_period import find_period

find_period(path="time_series_data.csv") # function call
```

### Parameters
- **path (String, required)**: Path to the timeseries data
- **tol_norm_diff (float, default=0.001)**: The tolerance for the norm difference between the the data and the fitted model, see res_criteria = 1.5.
- **number_steps (int, default=1000)**: Number of steps by which the range 0 to 1 of the threshold for zero (sigma) is divided.
- **minimum_number_of_relevant_shifts (int, default=2)**: The minimum number of shifts required for calculation of the period.
- **minimum_number_of_datapoints_for_correlation_test (int, default=300)**: The minimum number of datapoints required for calculation.
- **minimum_ratio_of_datapoints_for_shift_autocorrelation (float, default=0.3)**: The minimum ratio of datapoints for which we calculate the autocorrelation of a shift.
- **consider_only_significant_correlation (bool, default=True)**: A flag declaring the usage only of correlations matching significance.
- **level_of_significance_for_pearson (float, default=0.01)**: The minimum significance level for the correlation criterion.
- **output_flag (bool, default=True)**: Output flag setting plotting to on/off.
- **plot_tolerances (bool, default=True)**: Output flag allowing for tolerances to be plotted.
- **reference_time (pd.Timestamp, default=pd.Timestamp('2017-01-01T12'))**: The reference point for phase calculation.


### Output
These are the possible outputs you get when the output_flag is set during execution.
#### In case a period is found
1. A comparison of the original timeseries and the one predicted by the model.
2. A plot of the norm difference between unshifted and shifted autocorrelation function, optionally with tolerances (horizontal lines).
3. A plot of the autocorrelation function of the data linking the autocorrelation function to the correlation coefficient r and the significance p.
4. A plot of the difference between the original timeseries and the model. In case the model fits well this is only noise.
5. A plot of the autocorrelation function of the original data minus the model. A low autocorrelation function value here indicates a well fit model.

#### In case no period was found
1. A plot of the timeseries data
2. A plot of the norm difference between unshifted and shifted autocorrelation function.
3. A plot of the autocorrelation function of the data linking the autocorrelation function to the correlation coefficient r and the significance p.

### Returns
The function find_period returns a namedtuple Results(period, model, criteria).

#### In case a period is found
- **res_period (float)**: The period duration itself in minutes.
- **res_model (RandomForestRegressor object)**: The best performing model trained to fit the timeseries.
- **res_criteria (float)**: A float acting as performance criterium for the model. Between 0 and 1, set to 1.5 in case of a near perfect fit for the model.

#### In case no period is found
- **res_period (float)**: Set to -1 as no period exists.
- **res_model (RandomForestRegressor object)**: Set to None as no model exists.
- **res_criteria (float)**: Set to 0 as there is no model that fits the timeseries.

#### In case of an error
If an error occurs during execution the error message is printed and further code will still be executed.
- **res_period (float)**: Set to -2 in order to indicate an error.
- **res_model (RandomForestRegressor object)**: Set to None as no model exists.
- **res_criteria (float)**: Set to -2 in order to indicate an error.
