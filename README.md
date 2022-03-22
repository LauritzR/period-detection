# Period-Detection

## Usage

```Python
from find_period_not_equidist import find_period_not_equidist

find_period_not_equidist(path="time_series_data.csv") # function call
```

### Parameters
- **path (String, required)**: Path to the timeseries data
- **tol_norm_diff (float, default=0.001)**: The tolerance for the norm difference between the the data and the fitted model, see res_criteria = 1.5.
- **number_steps (int, default=1000)**: Number of steps by which the range 0 to 1 of the threshold for zero (sigma) is devided.
- **minimum_number_of_relevant_shifts (int, default=2)**: The minimum number of shifts required for calculation.
- **minimum_number_of_datapoints_for_correlation_test (int, default=300)**: The minimum number of datapoints required for calculation.
- **minimum_ratio_of_datapoints_for_shift_autocorrelation (float, default=0.3)**: The minimum ratio of datapoints for which we calculate the autocorrelation of a shift.
- **consider_only_significant_correlation (bool, default=True)**: A flag declaring the usage only of correlations matching our criterion.
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
#### In case a period is found
- **res_period (float)**: The period duration itself in minutes.
- **res_model (RandomForestRegressor object)**: The best performing model trained to fit the timeseries.
- **res_criteria (float)**: A float acting as performance criterium for the model. Between 0 and 1, set to 1.5 in case of a near perfect fit for the model.

#### In case no period is found
- **res_period (float)**: The period duration itself in minutes, here set to -1 as no period exists.
- **res_criteria (float)**: A float acting as performance criterium for the model. Here set to 0 as ther is no model that fits the timeseries.
