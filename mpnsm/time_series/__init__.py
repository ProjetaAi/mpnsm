"""TimeSeries Module."""

FIT_SMOOTH_FREQ = {'D': 30, 'W': 8, 'M': 6, 'Y': 3}

WINDOW_FREQ = {'D': 30, 'W': 8, 'M': 12, 'Y': 3}

CHANGEPOINTS_SECURE_MARGIN_FREQ = {'D': 25, 'W': 12, 'M': 5, 'Y': 1}

OFFSET_ARG_FREQ = {'D': 'days', 'W': 'weeks', 'M': 'months', 'Y': 'years'}

OFFSET_BEGIN_FREQ = {'D': 'D', 'W': 'W', 'M': 'MS', 'Y': 'AS'}

DEFAULT_MODEL = 'ProphetModel'

DEFAULT_CONFIG = {
    'ProphetModel': {
        'growth': 'flat',
        'daily_seasonality': False,
        'weekly_seasonality': False,
        'yearly_seasonality': False,
        'uncertainty_samples': None
    }
}

DEFAULT_OUTLIERS_CONFIG = {'outlier_handle': False}

DEFAULT_FIT_SMOOTH = {'fit_smooth_config': False}

DEFAULT_CV = 'TimeSeriesCV'

DEFAULT_CV_CONFIG = {'n_splits': 1}

DEFAULT_METRIC = 'WMAPEMetric'

DEFAULT_METRICS = [{
    'type': 'WMAPEMetric',
    'pred_col': 'yhat',
    'real_col': 'y_real'
}]

DEFAULT_HYPER = 'HyperparameterTuning'
