from __future__ import annotations

import logging
from copy import deepcopy
from importlib import import_module
from typing import List, Dict, Any, Tuple, Union, TYPE_CHECKING
from collections import OrderedDict
import pandas as pd
import numpy as np
from scipy.stats import mode

from dags.models.time_series import (DEFAULT_MODEL, DEFAULT_CONFIG, DEFAULT_CV,
                                     OFFSET_ARG_FREQ, DEFAULT_HYPER,
                                     OFFSET_BEGIN_FREQ)

from dags.models.time_series.config import get_columns_needed
from dags.models.time_series.utils import identify_outliers

from dags.models.time_series.model import SupervisedTSModel
from dags.models.time_series.metrics import Metric
from dags.models.time_series.target import Target
from dags.models.time_series.regressor import Regressor, SeasonalityRegressor

if TYPE_CHECKING:
    from dags.models.time_series.hyperparameter_tuning import HyperparameterTuning
    from dags.models.time_series.cross_validation import CrossValidation


class TimeSeriesForecast():
    """Class for making Time Series Forecasts."""

    def __init__(self,
                 data: pd.DataFrame,
                 forecast_unit: str,
                 date_col: str,
                 freq: str,
                 predict_mode: str,
                 targets: List[Dict[str, Union[str, Dict, List]]],
                 cv: Dict[str, Any],
                 units: list = []):
        """Class for making Time Series Forecasts.
        Args:
            data (pd.DataFrame): DataFrame with TimeSeries data and regressors data
            forecast_unit (str): Name that identifies the forecast
            date_col (str): Column that represents date in data
            freq (str): Frequency alias of the forecast time freequency. Use pandas alias.
            predict_mode (str): Can be 'full' or 'stepwise'. 
                                'full' makes each prediction target at once. If you have two targets,
                                the second target will only be predicted after the end of the first target full prediction (at the whole horizon). 
                                'stepwise' makes each prediction step at once. If you have two targets,
                                the second target first step will be predicted after the end of the first target first step prediction. 
            targets (List[Dict[str, Union[str, Dict, List]]]): List of target configurations as Dict. 
                                                              To each target config, see Target doc for more details
            cv (Dict[str, str]): CV can be specified as a Dict in the following way:
                                 Mandatory arguments:
                                    - type: CV Class name
                                    **cv_params: Arguments of the CV Class.
                                        See the corresponding CV class doc for more details.
            units (list): List with strings that define the time series unit. Will be used in loggings. Defaults to [].
        """

        self.data: pd.DataFrame = data
        self.date_col: str = date_col
        self.forecast_unit: str = forecast_unit
        self.predict_mode: str = predict_mode
        self.freq: str = freq
        self.targets: List[Target] = [Target(**target) for target in targets]

        assert len(
            self.targets) > 0, "Forecaster must have at least one target"

        self.regressors: Dict[str, List[str]] = {}
        self.cv: Dict[str, str] = cv
        self.units: list = units

        self.best_params: Dict[str, Dict[str, float]] = {}

    @property
    def _logger(self):
        logger = logging.getLogger('_'.join(self.units))
        logger.setLevel(logging.INFO)
        return logger

    def make_inputs(
        self,
        full_df: pd.DataFrame,
        model: Dict[str, SupervisedTSModel],
        last_index: int,
        verbose: int = 0
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[
            str, SupervisedTSModel]]:
        """
        Method to refine inputs for the forecaster.
        Args:
            full_df (pd.DataFrame): Full DataFrame used for modelling
            model (Dict[str, SupervisedTSModel]): Dict with models per target
            last_index (int): Last index of training sample in full dataframe
            verbose (int): Level of verbosity
        Returns:
            Dict[str, pd.DataFrame]: Full DataFrame 
            Dict[str, pd.DataFrame]: Train DataFrame
            Dict[str, SupervisedTSModel]: SupervisedTSModel Dict
        """

        full: Dict[str, pd.DataFrame] = {}
        train_df: Dict[str, pd.DataFrame] = {}

        for target in self.targets:  #testando targets[0]

            if verbose > 0:
                self._logger.info(f'Initializing {target.target_col} inputs')

            fut_unit = full_df.copy()

            fut_unit['y_real'] = fut_unit[
                target.target_col].copy()  # cria y_real com target real

            if last_index is not None:  # não entendi o uso do last_index
                fut_unit[target.target_col] = np.where(
                    fut_unit.index <= last_index, fut_unit[target.target_col],
                    np.nan)  # tudo dps do last_index vira nan para ser previsto
                fut_unit['is_train'] = np.where(fut_unit.index <= last_index,
                                                1, 0)
            else:
                fut_unit['is_train'] = np.where(
                    fut_unit.index <=
                    fut_unit.dropna(subset=[target.target_col]).index.max(), 1,
                    0)

            target_copy: Target = deepcopy(target)

            for reg in target_copy.regressors:

                reg_dict: Dict = reg.copy()

                reg_module = import_module('dags.models.time_series.regressor')

                # pega apenas o tipo do regressor (se não existe default é normal)
                regressor_type = reg_dict.pop('type', 'NormalRegressor')

                regressor = getattr(reg_module, regressor_type)

                if 'Seasonality' in regressor_type:
                    reg_dict.update({'freq': self.freq})

                regressor: Regressor = regressor(date_col=self.date_col,
                                                 **reg_dict)

                model[target.target_col], fut_unit = regressor.add_regressor(
                    df=fut_unit, model=model[target.target_col])

            full[target.
                 target_col] = fut_unit  #adiciona df da target no dic full
            train_df[target.target_col] = fut_unit.loc[fut_unit[
                'is_train'] == 1]  # adiciona apenas df de treino na train_df

        return full, train_df, model  # retorna full -> dataframes completos (target + regressores) | train_df -> full apenas com dados de treino | model -> dicionário com parâmetros da rodagem + regressores

    def initialize_model(
            self,
            target_col: str,
            model: Dict[str, Union[str, Dict, list]],
            hyperparams: Dict[str, Any] = {}) -> SupervisedTSModel:
        """
        Method to initialize model instances
        Args:
            target_col (str): Column that defines target
            model (Dict[str, Union[str, Dict, list]]): Dict with model configuration. See Model class for more details
            hyperparams (Dict[str, Any]): Dict with hyperparameter desired for the model.
        Returns:
            SupervisedTSModel: Model instance
        """

        model_copy = model.copy()

        # Criação de instancia do Prophet
        model_type = model_copy.get('type', DEFAULT_MODEL)

        if 'type' in model_copy:
            _ = model_copy.pop('type')

        if 'hyperparams_kwargs' in model_copy:
            _ = model_copy.pop(
                'hyperparams_kwargs'
            )  #remove keys desnecessárias (tipo já foi separado e hyperparams é variável do método)

        model_unit = import_module('dags.models.time_series.model')
        model_unit = getattr(model_unit,
                             model_type)  # cria o objeto prophetmodel

        params = DEFAULT_CONFIG.get(
            model_type,
            {}).copy()  # chama o default, se não tenho args ele deixa o padrão
        params.update(model_copy)
        params.update(hyperparams)  # insere parâmetros da rodagem no objeto

        model_unit = model_unit(
            freq=self.freq,
            date_col=self.date_col,
            target_col=target_col,
            **params)  # inicializa o objeto model com os parâmetros da rodagem

        model_unit: SupervisedTSModel

        return model_unit  # retorna o objeto model com os parâmetros

    def initialize(
        self,
        full_df: pd.DataFrame = None,
        model: Dict[str, SupervisedTSModel] = None,
        last_index: int = None,
        hyperparams: Dict[str, Any] = {},
        verbose: int = 0
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[
            str, SupervisedTSModel]]:
        """
        Method to initialize model instance and refine the basic inputs
        Args:
            full_df (pd.DataFrame): Full DataFrame used for modelling
            model (Dict[str, SupervisedTSModel]): Dict with models per target
            last_index (int): Last index of training sample in full dataframe
            hyperparams (Dict[str, Any]): Dict with hyperparameter desired for the model.
            verbose (int): Level of verbosity
        Returns:
            Dict[str, pd.DataFrame]: Full DataFrame 
            Dict[str, pd.DataFrame]: Train DataFrame
            Dict[str, SupervisedTSModel]: SupervisedTSModel Dict
        """

        hyperparams_copy = hyperparams.copy()

        if model is None:

            model = {}

            for target in self.targets:

                if verbose > 0:
                    self._logger.info(
                        f'Initializing {target.target_col} model')

                model[target.target_col] = self.initialize_model(
                    model=target.model,
                    target_col=target.target_col,
                    hyperparams=hyperparams_copy.pop(target.target_col, {}))

        if full_df is None:
            full_df = self.data.copy()

        full_df, train_df, model = self.make_inputs(full_df=full_df,
                                                    model=model,
                                                    last_index=last_index,
                                                    verbose=verbose)

        return full_df, train_df, model

    def _fit(self, model: SupervisedTSModel, target: Target,
             train_df: pd.DataFrame) -> Tuple[SupervisedTSModel, pd.DataFrame]:
        """
        Method that fits model to a specific target
        Args:
            model (SupervisedTSModel): Model instance
            target (Target): Target instance
            train_df (pd.DataFrame): Train DataFrame 
        Returns:
            model (SupervisedTSModel): Model fitted
            train_df (pd.DataFrame): Enriched Train DataFrame
        """

        target_copy: Target = deepcopy(target)

        outliers_config = target_copy.outliers_config
        outlier_handle = outliers_config.pop('outlier_handle', 'fit')

        if outlier_handle is not False:
            outliers = identify_outliers(df2=train_df,
                                         target_col=target_copy.target_col,
                                         **outliers_config)
        else:
            outliers = []

        # Se a estrategia de lidar com outliers for remover, ele remove as datas e esvazia a lista de outliers para nao serem fittados no fit
        if outlier_handle == 'drop':
            train_df = train_df.loc[~train_df[self.date_col].isin(outliers)]
            outliers = []

        # removendo intervalos indesejados do período de treinamento
        drop_intervals = target_copy.drop_intervals
        for interval in drop_intervals:
            train_df = train_df.loc[~train_df[self.date_col].
                                    between(interval[0], interval[1])]

        #Guardando outliers no objeto model
        outliers = [
            outlier for outlier in outliers if outlier in train_df[
                self.date_col].dt.strftime('%Y-%m-%d').unique()
        ]
        model.save_outliers(outliers)

        #Adicionando fit outliers
        if outlier_handle == 'fit':
            model.add_fit_outliers()  #adiciona os outliers como regressores???
            train_df = model.set_outliers(train_df)

        # retirando todos os valores NaN depois dos cálculos
        train_df = train_df.dropna(how='all', axis=1).dropna()

        # Treino do modelo do prophet. Utilizamos a suppress_stdout_stderr para evitar verbose desnecessario
        model.fit(train_df, **target_copy.model.get('fit_kwargs', {}))

        return model, train_df

    def fit(
        self,
        model: Dict[str, SupervisedTSModel],
        train_df: Dict[str, pd.DataFrame],
        verbose: int = 0
    ) -> Tuple[Dict[str, SupervisedTSModel], Dict[str, pd.DataFrame]]:
        """
        Method that fits model to a specific target
        Args:
            model (Dict[str,SupervisedTSModel]): Dict of Model instances per target
            train_df (Dict[str,pd.DataFrame]): Dict of Train DataFrames per target
            verbose (int): Level of verbosity 
        Returns:
            model (Dict[str,SupervisedTSModel]): Dict of Models fitted per target
            train_df (Dict[str,pd.DataFrame]): Dict of Enriched Train DataFrames per target
        """

        for target in self.targets:

            if verbose > 0:
                self._logger.info(f"Fitting {target.target_col} model")

            (model[target.target_col],
             train_df[target.target_col]) = self._fit(
                 model=model[target.target_col],
                 target=target,
                 train_df=train_df[target.target_col])

        return model, train_df

    def update_data(self, future: Dict[str, pd.DataFrame],
                    fcst: Dict[str, pd.DataFrame], start: int,
                    step: int) -> pd.DataFrame:
        """
        Method to update data in future dataframe while predictions are being made
        Args:
            future (Dict[str, pd.DataFrame]): Dict of predictions per target
            fcst (Dict[str, pd.DataFrame]): Dict of Model results dataframe per target
            start (int): Index in df to start updating data
            step (int): Index relative to start to update data
        Returns:
            future (Dict[str, pd.DataFrame]): Dict of Future dataframes updated per target
        """

        for target in self.targets:

            # Updating targets
            for target_col, _ in future.items():
                if target_col in future[target.target_col].columns:
                    future[target.target_col].loc[
                        start + step,
                        target_col] = fcst[target_col].loc[start + step,
                                                           'yhat'].values

        for target in self.targets:

            # Updating features
            for regressor in target.regressors:
                regressor: Dict
                reg_type = regressor.get('type', 'NormalRegressor')
                if reg_type == 'CalculatedRegressor':
                    if len(
                            set(regressor['calc_cols']).intersection(
                                set(list(future.keys())))) > 0:
                        future[
                            target.
                            target_col].loc[:, regressor['name']] = regressor[
                                'calc_func'](future[target.target_col]).values

        return future

    def predict(
        self,
        full_df: Dict[str, pd.DataFrame],
        model: Dict[str, SupervisedTSModel],
        verbose=0
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, SupervisedTSModel], Dict[
            str, pd.DataFrame]]:
        """
        Method that make predictions given a trained model
        Args:
            full_df (pd.DataFrame): Full DataFrame used for modelling
            model (Dict[str, SupervisedTSModel]): Model Dict per target
            verbose (int): Level of verbosity
        Returns:
            fcst (Dict[str, pd.DataFrame]):  Dict of DataFrame with model results detailed per target
            model (Dict[str, SupervisedTSModel]): SupervisedTSModel Dict
            future (Dict[str, pd.DataFrame]): Dict of DataFrame with past and future predictions per target
        """

        fcst: Dict[str, pd.DataFrame] = {}

        if verbose > 0:
            self._logger.info(
                f'Running predict in predict_mode = {self.predict_mode}')

        # Executa a previsão de todo o horizonte em um step só
        if self.predict_mode == 'full':

            for target in self.targets:
                fcst[target.target_col] = model[
                    target.target_col].predict(
                        full_df[target.target_col])

        # Executa a previsão passo a passo do horizonte.
        if self.predict_mode == 'stepwise':

            for target in self.targets:

                last_date = full_df[target.target_col].dropna(
                    subset=[target.target_col])[
                        self.date_col].max()  # ultima data real
                horizon = len(full_df[target.target_col].loc[
                    full_df[target.target_col][self.date_col] > last_date]
                              )  # n de meses/dias previstos
                start = full_df[target.target_col].loc[
                    full_df[target.target_col][self.date_col] ==
                    last_date].index + 1  # indice do primeiro ponto de dado previsto

                fcst[target.target_col] = model[target.target_col].predict(
                    full_df[target.target_col].loc[full_df[target.target_col]
                                                   ['is_train'] == 1])
                future_dates = pd.DataFrame(
                    pd.date_range(
                        start=fcst[target.target_col][self.date_col].min(),
                        end=fcst[target.target_col][self.date_col].max() +
                        pd.DateOffset(**{OFFSET_ARG_FREQ[self.freq]: horizon}),
                        freq=OFFSET_BEGIN_FREQ[self.freq]),
                    columns=[self.date_col])  # dataframe com todas as datas
                fcst[target.target_col] = future_dates.merge(
                    fcst[target.target_col], on=self.date_col,
                    how='left')  # adiciona datas futuras no dataframe

            for step in range(horizon):

                date = pd.to_datetime(last_date) + pd.DateOffset(
                    **{OFFSET_ARG_FREQ[self.freq]: 1 + step})

                if verbose > 0:
                    self._logger.info(f'Predicting step {1+step}:{date}')

                for target in self.targets:

                    if verbose > 0:
                        self._logger.info(
                            f'Predicting step {1+step}:{date} for {target.target_col} model'
                        )

                    fcst[target.target_col].loc[start + step, :] = model[
                        target.target_col].predict(
                            full_df[target.target_col].loc[start +
                                                           step, :]).values

                    full_df = self.update_data(future=full_df,
                                               fcst=fcst,
                                               start=start,
                                               step=step)

        return fcst, model, full_df

    def fit_predict(self,
                    full_df=None,
                    last_index=None,
                    verbose=0,
                    best_params=None):
        """
        Method that trains models and make predictions
        Args:
            full_df (pd.DataFrame): Full DataFrame used for modelling
            last_index (int): Last index of training sample in full dataframe
            verbose (int): Level of verbosity
            best_params (dict): Dict with best_params per target
        Returns:
            fcst (Dict[str, pd.DataFrame]):  Dict of DataFrame with model results detailed per target
            model (Dict[str, SupervisedTSModel]): SupervisedTSModel Dict
            future (Dict[str, pd.DataFrame]): Dict of DataFrame with past and future predictions per target
        """

        if best_params is None:
            best_params = self.best_params.copy()

        full_df, train_df, model = self.initialize(full_df=full_df,
                                                   last_index=last_index,
                                                   hyperparams=best_params,
                                                   verbose=verbose)
        model, train_df = self.fit(model=model,
                                   train_df=train_df,
                                   verbose=verbose)

        fcst, model, future = self.predict(full_df=full_df,
                                           model=model,
                                           verbose=verbose)

        return fcst, model, future

    def initialize_metric(
            self, metric: Dict[str, Union[str, int, Metric]]) -> Metric:
        """
        Method to initialize metric instances
        Args:
            metric (Dict[str, Union[str, Metric, int]]): Dict with metric configuration. 
            See Metric class for more details
        Returns:
            Metric: Metric instance
        """

        metric_copy = metric.copy()

        # Criação de instancia do Prophet
        metric_type = metric_copy.get('type')

        if 'type' in metric_copy:
            _ = metric_copy.pop('type')

        metric_unit = import_module('dags.models.time_series.metrics')
        metric_unit = getattr(metric_unit, metric_type)

        params = metric_copy.copy()

        metric_unit = metric_unit(**params)

        metric_unit: Metric

        return metric_unit

    def score(self,
              full_df: Dict[str, pd.DataFrame],
              fcst: Dict[str, pd.DataFrame],
              targets_list: List = None):
        """
        Method to score models and get in-sample and out-sample metrics
        Args:
            full_df (Dict[str, pd.DataFrame]): Dict of full DataFrame used for modelling per target
            fcst (pd.DataFrame): Dict of Model results per target
            targets_list (List): List of targets desired to be scored
        Returns:
            insample (dict): Dict of insample results
            outsample (dict): Dict of outsample results
        """

        insample = {}
        outsample = {}

        if targets_list is None:
            targets_list = list(full_df.keys())

        for target in self.targets:

            if target.target_col not in targets_list:
                continue

            insample[target.target_col] = {}
            outsample[target.target_col] = {}

            # Cria dataframe de erro in-sample (treino)
            score_df = fcst[target.target_col][[
                'ds', 'yhat', 'yhat_upper', 'yhat_lower'
            ]].merge(full_df[target.target_col], on=self.date_col, how='left')

            insample_unit = score_df.loc[score_df['is_train'] == 1]
            outsample_unit = score_df.loc[score_df['is_train'] == 0]

            metric_unit = target.metrics

            for metric in metric_unit:
                metric_obj: Metric = self.initialize_metric(metric)
                insample_obj = deepcopy(metric_obj)
                outsample_obj = deepcopy(metric_obj)
                insample[target.target_col][
                    insample_obj.name] = insample_obj.calculate(
                        df=insample_unit)
                outsample[target.target_col][
                    outsample_obj.name] = outsample_obj.calculate(
                        df=outsample_unit)

        return insample, outsample

    def initialize_cv(self, cv: Dict[str, Union[str, CrossValidation, List]]):
        """
        Method to initialize cross validation instances
        Args:
            hyper (Dict[str, Union[str, CrossValidation, list]]): Dict with cross validation configuration. 
            See CrossValidation class for more details
        Returns:
            CrossValidation: CrossValidation instance
        """

        cv_copy = cv.copy()

        # Criação de instancia do Prophet
        cv_type = cv_copy.get('type', DEFAULT_CV)

        if 'type' in cv_copy:
            _ = cv_copy.pop('type')

        cv_unit = import_module('dags.models.time_series.cross_validation')
        cv_unit = getattr(cv_unit, cv_type)

        params = cv_copy.copy()

        cv_unit = cv_unit(**params)

        cv_unit: CrossValidation

        return cv_unit

    def initialize_hyper(self, hyper: Dict[str,
                                           Union[str, HyperparameterTuning,
                                                 List]]):
        """
        Method to initialize hyperparameter instances
        Args:
            hyper (Dict[str, Union[str, HyperparameterTuning, list]]): Dict with hyperparameter tuning configuration. 
            See HyperparameterTuning class for more details
        Returns:
            HyperparameterTuning: HyperparameterTuning instance
        """

        hyper_copy = hyper.copy()

        # Criação de instancia do Prophet
        hyper_type = hyper_copy.get('type', DEFAULT_HYPER)
        hyper_configs = hyper_copy.get('configs')

        if 'type' in hyper_copy:
            _ = hyper_copy.pop('type')

        hyper_unit = import_module(
            'dags.models.time_series.hyperparameter_tuning')
        hyper_unit = getattr(hyper_unit, hyper_type)

        hyper_unit = hyper_unit(hyperparams=hyper_copy['hyperparams'],
                                forecast=self,
                                cv=hyper_copy['cv'],
                                configs=hyper_configs,
                                logger=self._logger)

        hyper_unit: HyperparameterTuning

        return hyper_unit

    def optimize(self,
                 train_df: Dict[str, pd.DataFrame],
                 verbose: int = 0) -> Dict[str, SupervisedTSModel]:
        """
        Method that makes hyperparameter optimization
        Args:
            train_df (dict): Dict with train_df per target
            verbose (int): Level of verbosity
        Returns:
            best_params (dict): Dict with best_params per target
        """

        targets = self.targets.copy()
        best_params = {}

        for target in targets:  #ACRESCENTAR CONFIG DE SALVAR NO LAKE SE QUISER

            self._logger.info(f'Optimizing {target.target_col} model')

            hyper = target.model.get('hyperparams_kwargs', {})

            if len(hyper) > 0:
                cv = self.initialize_cv(self.cv.copy())
                hyper['cv'] = cv
                hyper = self.initialize_hyper(hyper)
                best_params = hyper.optimize(target_name=target.target_col,
                                             train_df=train_df,
                                             verbose=verbose,
                                             best_params=best_params)
        return best_params

    def optimize_predict(self, save_params=True, verbose=0):
        """
        Method that makes forecast optimization
        Args:
            best_params (dict): Dict with best params per target
            verbose (int): Level of verbosity
        Returns:
            fcst (pd.DataFrame): DataFrame with model results detailed
            model (pd.DataFrame): Dict of models per target
            future (pd.DataFrame): DataFrame with past and future predictions
        """

        _, train_df, _ = self.initialize(verbose=verbose)
        best_params = self.optimize(train_df=train_df, verbose=verbose)
        fcst, model, future = self.fit_predict(verbose=verbose,
                                               best_params=best_params)

        if save_params:
            self.best_params = best_params.copy()

        return fcst, model, future

    def fit_predictCV(self, best_params=None, verbose=0):
        """
        Method that makes cross validation inside a model
        Args:
            best_params (dict): Dict with best params per target
            verbose (int): Level of verbosity
        Returns:
            fcst_df (pd.DataFrame): DataFrame with model results detailed
            future_df (pd.DataFrame): DataFrame with past and future predictions
            insample (Dict): Dict with insample errors
            outsample (Dict): Dict with outsample errors
        """

        if best_params is None:
            best_params = self.best_params.copy()

        _, train_df, _ = self.initialize(verbose=verbose)

        insample = OrderedDict()
        outsample = OrderedDict()

        fcst_df = pd.DataFrame()
        future_df = pd.DataFrame()

        cv = self.initialize_cv(self.cv.copy())

        tgt = self.targets[0].target_col
        for train_cv, _ in cv.split(X=train_df[tgt]):

            train_cv: pd.RangeIndex

            best_param_value = best_params.copy()

            fcst, _, future = self.fit_predict(best_params=best_param_value,
                                               last_index=max(train_cv),
                                               verbose=verbose)

            insample[max(train_cv)], outsample[max(train_cv)] = self.score(
                full_df=future, fcst=fcst)

            fcst = fcst[tgt].copy()
            future = future[tgt].copy()

            fcst['fold'] = max(train_cv)
            future['fold'] = max(train_cv)

            fcst_df = pd.concat([fcst_df, fcst], sort=True, ignore_index=True)
            future_df = pd.concat([future_df, future],
                                  sort=True,
                                  ignore_index=True)

        return fcst_df, future_df, insample, outsample

    def get_best_fold(self, metric):
        """
        Method to get best fold of a cross validation
        Args:
            metric (Dict[str,Metric]): Dict with metrics per target
        Returns:
            chosen_fold
        """

        best_folds = {}
        for target_name in [target.target_col for target in self.targets]:

            first_metric = next(
                iter(metric[next(iter(metric))].get(target_name)))
            higher_better = metric[next(
                iter(metric))].get(target_name).get(first_metric).higher_better

            tuple_metrics = [
                (fold, metric[fold].get(target_name).get(first_metric).value)
                for fold in iter(metric)
            ]

            if higher_better:
                best_fold = max(tuple_metrics, key=lambda t: t[1])[0]
            else:
                best_fold = min(tuple_metrics, key=lambda t: t[1])[0]

            best_folds[target_name] = best_fold

        chosen_fold = int(mode(list(best_folds.values())).mode)

        return chosen_fold

    def load_params(self, params):
        """
        Method that load params inside self scope
        Args:
            params (Dict[str,Dict[str,float]]): Dict with targets and their respective parameters
        """

        self.best_params = params
