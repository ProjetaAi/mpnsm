from __future__ import annotations

import logging
from abc import abstractmethod
from tokenize import Number
from typing import Any, Dict, List, Union, TYPE_CHECKING
from collections import OrderedDict
import pandas as pd
import numpy as np

from sklearn.model_selection import ParameterGrid
from scipy.stats import mode

from dags.models.time_series.cross_validation import CrossValidation
from dags.models.time_series.metrics import Metric

if TYPE_CHECKING:
    from dags.models.time_series.forecast import TimeSeriesForecast


class HyperparameterTuning:

    """Hyperparameter Tuning abstract class"""

    def __init__(self,
                 hyperparams: List[Dict[str, Union[str, List, Dict]]],
                 forecast: TimeSeriesForecast,
                 cv: CrossValidation,
                 logger=None):

        """
        Hyperparameter Tuning abstract class
        Args:
            hyperparams (List[Dict[str, Union[str, List, Dict]]]): List of hyperparameters configurations. See specifications in each subclass.
            forecast (TimeSeriesForecast): Forecaster instance
            cv (CrossValidation): CrossValidation instance
            logger (logging): Logger instance
        """

        self.hyperparams = hyperparams
        self.forecast = forecast
        self.cv = cv
        self.logger = logger

    @property
    def _logger(self):
        """Logger that will be used in prints"""
        if self.logger is None:
            return logging.getLogger(__name__)
        return self.logger

    @abstractmethod
    def optimize(self,
                 target_name: str,
                 train_df: pd.DataFrame,
                 best_params: Dict[str, Any] = {},
                 verbose: int = 0):

        """
        Optimizer abstract method
        Args:
            target_name (str): Name of target being optimized
            train_df (pd.DataFrame): Dataset to optimize hyperparams for
            best_params (Dict[str, Any]): Dictionary with best hyperparameters
        """

        pass


class HeuristicHyperparameterTuning(HyperparameterTuning):

    def __init__(self,
                 hyperparams: List[Dict[str, Union[str, List, Dict]]],
                 forecast: TimeSeriesForecast,
                 cv: CrossValidation,
                 configs: dict,
                 logger=None):

        """
        Heuristic hyperparameter tuning class
        Args:
            hyperparams (List[Dict[str, Union[str, List, Dict]]]): List of hyperparameters configurations. See specifications in each subclass.
            forecast (TimeSeriesForecast): Forecaster instance
            cv (CrossValidation): CrossValidation instance
            configs (dict): Optimization configs. Ex.: If a certain metric will be chosen by mean or median. 
                            If the model will choose insample or outsample error for optimization
            logger (logging): Logger instance
        """

        super().__init__(hyperparams=hyperparams,
                         forecast=forecast,
                         cv=cv,
                         logger=logger)
        self.configs = configs

    def check_convergence(self,
                          chosen_metric: OrderedDict[int, OrderedDict[Number,
                                                                      Metric]],
                          target_name: str):

        """
        Convergence checking method
        Args:
            chosen_metric (OrderedDict[int, OrderedDict[Number,Metric]]): Dictionary with metric to be used in the optimization
            target_name (str): Name of target being optimized
        """

        # If first test skip check
        if len(chosen_metric) == 1:
            return False

        else:
            # Get last two metrics for comparison
            *_, b_last, last = chosen_metric
            old = chosen_metric.get(b_last).get(target_name)
            new = chosen_metric.get(last).get(target_name)
            metric_scores = []

            for metric in new.keys():

                metric_score = 0
                new_metric_obj = new.get(metric)
                old_metric_obj = old.get(metric)

                # If at least one metric is better than the previous one and no metric is worse, then convergence is not achieved
                if new_metric_obj.higher_better:
                    if new_metric_obj.value >= (old_metric_obj.value +
                                                new_metric_obj.min_better):
                        metric_score = 1
                    elif new_metric_obj.value < (old_metric_obj.value -
                                                 new_metric_obj.max_worse):
                        return True
                else:
                    if new_metric_obj.value <= (old_metric_obj.value -
                                                new_metric_obj.min_better):
                        metric_score = 1
                    elif new_metric_obj.value > (old_metric_obj.value +
                                                 new_metric_obj.max_worse):
                        return True
                metric_scores += [metric_score]

            if not sum(metric_scores):
                return True

        return False

    def choose_best_param(self,
                          chosen_metric: OrderedDict[int, OrderedDict[Number,
                                                                      Metric]],
                          configs: dict):

        """
        Method for choosing best parameters
        Args:
            chosen_metric (OrderedDict[int, OrderedDict[Number,Metric]]): Dictionary with metric to be used in the optimization
            configs (dict): Optimization configs. Ex.: If a certain metric will be chosen by mean or median. 
                            If the model will choose insample or outsample error for optimization
        """

        # Selects second to last values for chosen metric. The last one may be significantly worse due to converge parameters
        chosen = [
            list(reversed(chosen_metric[fold]))[-2] for fold in chosen_metric
        ]

        # Selects best value for parameter based on agg_mode in configs
        agg_mode = configs.get('agg_mode', 'median')

        if agg_mode == 'mode':
            return float(mode(chosen).mode)
        else:
            func = getattr(np, 'median')
            return func(chosen)

    def optimize_hyperparameter(self,
                                name: str,
                                target_name: str,
                                train_df: pd.DataFrame,
                                initial_value: Number,
                                final_value: Number,
                                step: int,
                                mode: str,
                                best_params: Dict[str, Dict[str, Any]],
                                verbose: int = 0):

        """
        Hyperparameter optimization method
        Args:
            name (str): Hyperparam being optimized
            target_name (str): Name of target being optimized
            train_df (pd.DataFrame): Dataset to optimize hyperparams for
            initial_value (Number): Starting hyperparameter value
            final_value (Number): Final hyperparameter value
            step (int): Steps taken between values, can be additive or multiplicative
            mode (str): If hyperparam is additive or multiplicative
            best_params (Dict[str, Dict[str, Any]]): Dictionary with best hyperparameters
        """

        insample = OrderedDict()
        outsample = OrderedDict()

        # Generate list of possible values
        if mode == 'multiplicative':
            values = []
            i = 0
            if initial_value < final_value:
                values.append(initial_value)
                while max(values) < final_value:
                    values.append(initial_value * (step**i))
                    i += 1
            else:
                values.append(final_value)
                while min(values) > initial_value:
                    values.append(final_value / (step**i))
                    i += 1

        if mode == 'additive':
            if initial_value < final_value:
                values = list(
                    np.arange(start=initial_value, stop=final_value,
                              step=step))
            else:
                values = list(
                    np.arange(start=initial_value,
                              stop=final_value,
                              step=-step))

        # Splits dataframe into cross validation sections for optimization
        for train_cv, test_cv in self.cv.split(X=train_df[target_name]):

            if verbose > 0:
                self._logger.info(
                    f'Optimizing hyperparam {name} from {target_name} model - Last Stop: {max(train_cv)}'
                )

            train_cv: pd.RangeIndex
            test_cv: pd.RangeIndex

            in_cv = OrderedDict()
            out_cv = OrderedDict()

            converged = False

            while not converged:

                for ind, value in enumerate(values):

                    if verbose > 0:
                        self._logger.info(
                            f'Optimizing hyperparam {name} from {target_name} model - Fold: {max(train_cv)} - Value: {value}'
                        )

                    best_param_value = best_params.copy()

                    if target_name not in best_param_value:
                        best_param_value[target_name] = {name: value}
                    else:
                        best_param_value[target_name].update({name: value})

                    # Fits and predicts with a certain hyperparam value
                    fcst, _, future = self.forecast.fit_predict(
                        best_params=best_param_value,
                        last_index=max(train_cv),
                        verbose=verbose)

                    # Calculates metrics for hyperparameter value
                    in_cv[value], out_cv[value] = self.forecast.score(
                        full_df=future, fcst=fcst, targets_list=[target_name])

                    # Chooses which metric will be used in optimization based on configs
                    in_or_out = self.configs.get('chosen_calc', 'outsample')

                    if in_or_out == 'outsample':
                        chosen_metric = out_cv.copy()
                    else:
                        chosen_metric = in_cv.copy()

                    # Checks if optimal value has been achieved
                    converged = self.check_convergence(chosen_metric,
                                                       target_name)

                    if ind == (len(values) - 1):
                        converged = True

                    if converged:
                        self._logger.info('Converged')
                        break

            insample[min(test_cv)] = in_cv
            outsample[min(test_cv)] = out_cv

        if in_or_out == 'outsample':
            chosen_metric = outsample.copy()
        else:
            chosen_metric = insample.copy()

        # Selects best parameter and returns
        best_param = self.choose_best_param(chosen_metric, self.configs)

        if target_name not in best_params:
            best_params[target_name] = {name: best_param}
        else:
            best_params[target_name].update({name: best_param})

        return best_params

    def optimize(self,
                 target_name: str,
                 train_df: pd.DataFrame,
                 best_params: Dict[str, Any] = {},
                 verbose: int = 0):

        """
        Optimizer method
        Args:
            target_name (str): Name of target being optimized
            train_df (pd.DataFrame): Dataset to optimize hyperparams for
            best_params (Dict[str, Any]): Dictionary with best hyperparameters
        """

        # Run hyperparam optimization for each hyperparam instance
        for hyperparam in self.hyperparams:
            hyperparam: Dict

            if verbose > 0:
                self._logger.info(
                    f'Optimizing hyperparam {hyperparam["name"]} from {target_name} model'
                )

            best_params = self.optimize_hyperparameter(target_name=target_name,
                                                       train_df=train_df,
                                                       best_params=best_params,
                                                       verbose=verbose,
                                                       **hyperparam)

        return best_params


class ParameterGridHyperparameterTuning(HyperparameterTuning):

    def __init__(self,
                 hyperparams: List[Dict[str, Union[str, List, Dict]]],
                 forecast: TimeSeriesForecast,
                 cv: CrossValidation,
                 configs: dict,
                 logger=None):

        """
        ParameterGrid hyperparameter tuning class
        Args:
            hyperparams (List[Dict[str, Union[str, List, Dict]]]): List of hyperparameters configurations. See specifications in each subclass.
            forecast (TimeSeriesForecast): Forecaster instance
            cv (CrossValidation): CrossValidation instance
            configs (dict): Optimization configs. Ex.: If a certain metric will be chosen by mean or median. 
                            If the model will choose insample or outsample error for optimization
            logger (logging): Logger instance
        """

        self.configs = configs

        super().__init__(hyperparams=hyperparams,
                         forecast=forecast,
                         cv=cv,
                         logger=logger)

    def choose_best_param(
        self,
        insample: OrderedDict[int, OrderedDict[Number, Metric]],
        outsample: OrderedDict[int, OrderedDict[Number, Metric]],
    ):

        """
        Method for choosing best parameters TODO: Rework method to accomodate higher_better metrics
        Args:
            insample (OrderedDict[int, OrderedDict[Number, Metric]]): Dictionary with insample metrics to be used in the optimization
            insample (OrderedDict[int, OrderedDict[Number, Metric]]): Dictionary with outsample metrics to be used in the optimization
        """

        errors_dict = [
            (vs, v,
             next(iter(next(iter(outsample[vs][v].values())).values())).value)
            for vs in iter(outsample) for v in iter(outsample[vs])
        ]

        best_param = min(errors_dict,
                         key=lambda x: x[2])[1], min(errors_dict,
                                                     key=lambda x: x[2])[0]

        return best_param

    def optimize_hyperparameter(self,
                                target_name: str,
                                train_df: pd.DataFrame,
                                best_params: Dict[str, Dict[str, Any]],
                                hyperparams: Dict[str, Any],
                                verbose: int = 0):

        insample = OrderedDict()
        outsample = OrderedDict()

        """
        Hyperparameter optimization method
        Args:
            target_name (str): Name of target being optimized
            train_df (pd.DataFrame): Dataset to optimize hyperparams for
            best_params (Dict[str, Dict[str, Any]]): Dictionary with best hyperparameters
            hyperparams (Dict[str, Any]): Dictionary with hyperparameters
        """

        params_grid = {}
        # Generates all hyperparameter values and outputs in ParameterGrid format
        for hyperparam in hyperparams:

            mode = hyperparam.get('mode')
            initial_value = hyperparam.get('initial_value')
            final_value = hyperparam.get('final_value')
            step = hyperparam.get('step')

            if mode == 'multiplicative':
                values = []
                i = 0
                if initial_value < final_value:
                    values.append(initial_value)
                    while max(values) < final_value:
                        values.append(initial_value * (step**i))
                        i += 1
                else:
                    values.append(final_value)
                    while min(values) > initial_value:
                        values.append(final_value / (step**i))
                        i += 1

            if mode == 'additive':
                if initial_value < final_value:
                    values = list(
                        np.arange(start=initial_value,
                                  stop=final_value,
                                  step=step))
                else:
                    values = list(
                        np.arange(start=initial_value,
                                  stop=final_value,
                                  step=-step))

            params_grid[hyperparam['name']] = values

        # Creates a list with all combinations of hyperparameters
        grid = ParameterGrid(params_grid)

        # Splits dataframe in CV splits
        for train_cv, test_cv in self.cv.split(X=train_df[target_name]):

            train_cv: pd.RangeIndex
            test_cv: pd.RangeIndex

            in_cv = OrderedDict()
            out_cv = OrderedDict()

            # Runs all possibilities for all possible folds
            for ind, p in enumerate(grid):

                if verbose > 0:
                    self._logger.info(
                        f'Optimizing hyperparams for {target_name} model - Fold: {max(train_cv)} - Values: {p}'
                    )

                best_param_value = best_params.copy()

                if target_name not in best_param_value:
                    best_param_value[target_name] = p
                else:
                    best_param_value[target_name].update(p)

                fcst, _, future = self.forecast.fit_predict(
                    best_params=best_param_value,
                    last_index=max(train_cv),
                    verbose=verbose)

                in_cv[ind], out_cv[ind] = self.forecast.score(
                    full_df=future, fcst=fcst, targets_list=[target_name])

                in_or_out = self.configs.get('chosen_calc', 'outsample')

            insample[min(test_cv)] = in_cv
            outsample[min(test_cv)] = out_cv

        if in_or_out == 'outsample':
            chosen_metric = outsample.copy()
        else:
            chosen_metric = insample.copy()
        self._logger.info(chosen_metric)

        # Chooses best parameters and returns
        best_param = self.choose_best_param(chosen_metric, self.configs)

        if target_name not in best_params:
            best_params[target_name] = grid[best_param]
        else:
            best_params[target_name].update(grid[best_param])

        return best_params

    def optimize(self,
                 target_name: str,
                 train_df: pd.DataFrame,
                 best_params: Dict[str, Any] = {},
                 verbose: int = 0):

        """
        Optimizer method
        Args:
            target_name (str): Name of target being optimized
            train_df (pd.DataFrame): Dataset to optimize hyperparams for
            best_params (Dict[str, Any]): Dictionary with best hyperparameters
        """

        if verbose > 0:
            self._logger.info(
                f'Optimizing hyperparams for {target_name} model')

        best_params = self.optimize_hyperparameter(
            target_name=target_name,
            train_df=train_df,
            best_params=best_params,
            verbose=verbose,
            hyperparams=self.hyperparams)

        return best_params