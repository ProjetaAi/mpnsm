from typing import Union
from functools import partial
import pandas as pd
import numpy as np


class Metric:
    """
    Abstract Metric Class
    """

    def __init__(self,
                 name: str = 'Metric',
                 higher_better: bool = True,
                 metric_func=None):

        """
        Abstract Metric Class
        """

        self.name = name
        self.higher_better = higher_better
        self.metric_func = metric_func
        self.value = None

    def get_best_value(self, values):

        if self.higher_better:
            return max(values)
        else:
            return min(values)

    def set_metric_function(self, func):

        """
        Function that set metric function in metric class
        Args:
            func: Function that implements metric equation
        """

        self.metric_func = func

    def calculate(self,
                  df: pd.DataFrame,
                  group: Union[list, str] = None,
                  **kwargs) -> pd.DataFrame:
        """
        Function that calculates metric equation and stores it in the object
        Args:
            df (pd.DataFrame): DataFrame with data needed to calculate the metric
            group (Union[List,str]): Group columns to calculate the metric inside group
            **kwargs: Metric equation keyword arguments
        Returns:
            self: Main object
        """

        if group is None:
            self.value = self.metric_func(df, **kwargs)
        else:
            self.value = df.groupby(group).apply(self.metric_func, **kwargs)

        return self


class ConfidenceIntervalMetric(Metric):
    """
    Abstract Confidence Interval Metric Class.
    """

    def __init__(self,
                 pred_col: str,
                 upper_col: str,
                 lower_col: str,
                 name: str = 'CI',
                 higher_better: bool = True,
                 metric_func=None):
        """
        Abstract Confidence Interval Metric Class.
        Args:
            pred_col (str): Prediction Column Name
            upper_col (str): Upper Range Column Name
            lower_col (str): Lower Range Column Name
            name (str): Metric Name. Defaults to 'WMAPE'
            higher_better (bool): If the metric is better when values are higher. Defaults to True.
            metric_func (function): Function that implements metric equation
        """

        super().__init__(name=name,
                         higher_better=higher_better,
                         metric_func=metric_func)

        self.pred_col = pred_col
        self.upper_col = upper_col
        self.lower_col = lower_col


class WeightedCIMetric(ConfidenceIntervalMetric):
    """
    WeightedCI Metric Class.
    """

    def __init__(self,
                 pred_col: str,
                 upper_col: str,
                 lower_col: str,
                 name: str = 'WeightedCI',
                 max_worse: float = 0.05,
                 min_better: float = 0.05):
        """
        WeightedCI Metric Class.
        Equation: sum(abs(upper - lower)) / sum(abs(pred))
        Args:
            pred_col (str): Prediction Column Name
            upper_col (str): Upper Range Column Name
            lower_col (str): Lower Range Column Name
            name (str): Metric Name. Defaults to 'WMAPE'
            max_worse (float): Maximum desired loss in a Hyperparameter Tuning iteration. Defaults to 0.03
            min_better (float): Minimum desired gain in a Hyperparameter Tuning. Defaults to 0.003
        """

        self.max_worse = max_worse
        self.min_better = min_better

        super().__init__(name=name,
                         higher_better=False,
                         pred_col=pred_col,
                         upper_col=upper_col,
                         lower_col=lower_col)

        self.set_metric_function(func=partial(self.weightedcifunc,
                                              upper_col=self.upper_col,
                                              lower_col=self.lower_col,
                                              pred_col=self.pred_col))

    @staticmethod
    def weightedcifunc(df: pd.DataFrame, upper_col: str, lower_col: str,
                       pred_col: str):
        """
        WeightedCI Metric Function.
        Equation: sum(abs(upper - lower)) / sum(abs(pred))
        Args:
            df (pd.DataFrame): DataFrame with prediction values (mean, upper and lower ranges)
            pred_col (str): Prediction Column Name
            upper_col (str): Upper Range Column Name
            lower_col (str): Lower Range Column Name
        """

        return ((np.abs(df[upper_col] - df[lower_col]).sum()) /
                np.abs(df[pred_col]).sum())


class ErrorMetric(Metric):
    """
    Abstract Error Metric Class.
    """

    def __init__(self,
                 pred_col: str,
                 real_col: str,
                 name: str = 'Error',
                 higher_better: bool = True,
                 metric_func=None):
        """
        Abstract Error Metric Class.
        Args:
            pred_col (str): Prediction Column Name
            real_col (str): Real Column Name
            date_col (str): Date Column Name
            name (str): Metric Name. Defaults to 'WMAPE'
            higher_better (bool): If the metric is better when values are higher. Defaults to True.
            metric_func (function): Function that implements metric equation
        """

        super().__init__(name=name,
                         higher_better=higher_better,
                         metric_func=metric_func)

        self.pred_col = pred_col
        self.real_col = real_col


class TSErrorMetric(ErrorMetric):
    """
    Abstract Time Series Error Metric Class.
    """

    def __init__(self,
                 pred_col: str,
                 real_col: str,
                 date_col: str,
                 name: str = 'TSError',
                 higher_better: bool = True):
        """
        Abstract Time Series Error Metric Class.
        Args:
            pred_col (str): Prediction Column Name
            real_col (str): Real Column Name
            date_col (str): Date Column Name
            name (str): Metric Name. Defaults to 'WMAPE'
            higher_better (bool): If the metric is better when values are higher. Defaults to True.
        """

        super().__init__(name=name,
                         pred_col=pred_col,
                         real_col=real_col,
                         higher_better=higher_better)

        self.date_col = date_col


class NonTSErrorMetric(ErrorMetric):
    """Abstract Non Time Series Error Metric"""
    pass


class WMAPEMetric(NonTSErrorMetric):
    """
    WMAPE Metric Class.
    """

    def __init__(self,
                 pred_col: str,
                 real_col: str,
                 name: str = 'WMAPE',
                 negative_values: str = 'handle',
                 max_worse: float = 0.03,
                 min_better: float = 0.003):
        """
        WMAPE Metric Class.
        Equation: sum(real-pred)/sum(real)
        Args:
            pred_col (str): Prediction Column Name
            real_col (str): Real Column Name
            name (str): Metric Name. Defaults to 'WMAPE'
            negative_values (str): How to handle with negative values. Can be 'handle' (makes min-max transform) or 'raise' (returns Error)
            max_worse (float): Maximum desired loss in a Hyperparameter Tuning iteration. Defaults to 0.03
            min_better (float): Minimum desired gain in a Hyperparameter Tuning. Defaults to 0.003
        """

        self.negative_values = negative_values
        self.max_worse = max_worse
        self.min_better = min_better

        super().__init__(name=name,
                         pred_col=pred_col,
                         real_col=real_col,
                         higher_better=False)

        self.set_metric_function(func=partial(self.wmape_func,
                                              real_col=self.real_col,
                                              pred_col=self.pred_col,
                                              negative_values=negative_values))

    @staticmethod
    def wmape_func(df: pd.DataFrame,
                   real_col: str,
                   pred_col: str,
                   negative_values: str = 'handle'):
        """
        Function that implements equation to calculate WMAPE Metric.
        Equation: sum(real_col-pred_col)/sum(real_col)
        Args:
            df (pd.DataFrame): DataFrame with real and predictions values
            pred_col (str): Prediction Column Name
            real_col (str): Real Column Name
            negative_values (str): How to handle with negative values. Can be 'handle' (makes min-max transform) or 'raise' (returns Error)
            
        """

        assert real_col in df, f"Real column ({real_col}) not found in dataframe"
        assert pred_col in df, f"Prediction column ({pred_col}) not found in dataframe"

        if negative_values == 'raise':
            assert (df[real_col] >= 0).all(
            ), 'Your real data has negative values. WMAPE Metric is not recommended'

        if negative_values == 'handle':
            if (df[real_col].min() < 0) & (df[real_col] > 0).any():
                min_real = df[real_col].min()
                df[real_col] -= min_real
                df[pred_col] -= min_real

        return np.abs(df[real_col] - df[pred_col]).sum() / df[real_col].sum()


class CombinedMetric(Metric):
    """
    Abstract Combined Metric Class.
    """

    def __init__(self,
                 name: str = 'CombinedMetric',
                 higher_better: bool = True,
                 metric_func=None):

        super().__init__(name=name,
                         higher_better=higher_better,
                         metric_func=metric_func)


class WMAPECIMetric(CombinedMetric):
    """
    WMAPE-CI Metric Class.
    """

    def __init__(self,
                 pred_col: str,
                 real_col: str,
                 lower_col: str,
                 upper_col: str,
                 name: str = 'WMAPECI',
                 negative_values: str = 'handle',
                 ci_coef: float = 0.5,
                 wmape_coef: float = 0.5,
                 max_worse: float = 0.03,
                 min_better: float = 0.003):
        """
        WMAPE-CI Metric Class.
        Equation: ci_coef * ci + wmape_coef * wmape (See CI equation and WMAPE equation in their respective classes)
        Args:
            pred_col (str): Prediction Column Name
            real_col (str): Real Column Name
            lower_col (str): Lower Range Column Name
            upper_col (str): Upper Range Column Name
            name (str): Metric Name. Defaults to 'WMAPECI'
            negative_values (str): How to handle with negative values. Can be 'handle' (makes min-max transform) or 'raise' (returns Error)
            ci_coef (float): Weight to CI Metric in final equation. Defaults to 0.5
            wmape_coef (float): Weight to WMAPE Metric in final equation. Defaults to 0.5
            max_worse (float): Maximum desired loss in a Hyperparameter Tuning iteration. Defaults to 0.03
            min_better (float): Minimum desired gain in a Hyperparameter Tuning. Defaults to 0.003
        """

        self.negative_values = negative_values
        self.max_worse = max_worse
        self.min_better = min_better
        self.ci_coef = ci_coef
        self.wmape_coef = wmape_coef

        super().__init__(name=name, higher_better=False)

        self.set_metric_function(func=partial(self.wmapecifunc,
                                              real_col=real_col,
                                              pred_col=pred_col,
                                              upper_col=upper_col,
                                              lower_col=lower_col,
                                              ci_coef=ci_coef,
                                              wmape_coef=wmape_coef,
                                              negative_values=negative_values))

    @staticmethod
    def wmapecifunc(df: pd.DataFrame,
                    real_col: str,
                    pred_col: str,
                    upper_col: str,
                    lower_col: str,
                    ci_coef: float = 0.5,
                    wmape_coef: float = 0.5,
                    negative_values: str = 'handle'):
        """
        Function that implements equation to calculate WMAPECI Metric.
        Equation: (ci_coef * ci + wmape_coef * wmape)/(ci_coef+wmape_coef)
        Args:
            df (pd.DataFrame): DataFrame with real and predictions values
            pred_col (str): Prediction Column Name
            real_col (str): Real Column Name
            lower_col (str): Lower Range Column Name
            upper_col (str): Upper Range Column Name
            negative_values (str): How to handle with negative values. Can be 'handle' (makes min-max transform) or 'raise' (returns Error)
            ci_coef (float): Weight to CI Metric in final equation. Defaults to 0.5
            wmape_coef (float): Weight to WMAPE Metric in final equation. Defaults to 0.5
        Returns:
            Number: Metric result
        """

        wmape = WMAPEMetric.wmape_func(df=df,
                                       real_col=real_col,
                                       pred_col=pred_col,
                                       negative_values=negative_values)

        ci = WeightedCIMetric.weightedcifunc(df=df,
                                             upper_col=upper_col,
                                             lower_col=lower_col,
                                             pred_col=pred_col)

        return (ci_coef * ci + wmape_coef * wmape) / (ci_coef + wmape_coef)
