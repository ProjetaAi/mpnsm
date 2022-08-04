from abc import abstractmethod
from typing import Any, Dict
from copy import deepcopy
import pandas as pd
import numpy as np

from dags.models.time_series import (FIT_SMOOTH_FREQ, DEFAULT_FIT_SMOOTH,
                                     OFFSET_ARG_FREQ, WINDOW_FREQ)
from dags.models.time_series.utils import set_changepoints
from dags.models.utils import suppress_stdout_stderr


class Model:

    def __init__(self, model: Any):
        self.model = model
        self.outliers = []
        self.outlier_handle = False

    def save_outliers(self, outliers):
        self.outliers = outliers

    def add_fit_outliers(self):
        self.outlier_handle = 'fit'
        for outlier in self.outliers:
            self.add_regressor(name=f'is_outlier_{outlier}')

    @abstractmethod
    def add_regressor(self, **kwargs):
        pass

    @abstractmethod
    def regressor_coefficients(self, fcst: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, **fit_kwargs):
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def save_model(self, location: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_model(self, location: str) -> pd.DataFrame:
        pass


class SupervisedModel(Model):

    def __init__(self, model: Any, target_col: str):

        super().__init__(model=model)
        self.target_col = target_col

    @abstractmethod
    def add_regressor(self, **kwargs):
        pass

    @abstractmethod
    def regressor_coefficients(self, fcst: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, **fit_kwargs):
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class SupervisedTSModel(SupervisedModel):

    def __init__(self, date_col: str, target_col: str, freq: str, model: Any):

        super().__init__(model=model, target_col=target_col)
        self.freq = freq
        self.date_col = date_col

    def set_outliers(self, df: pd.DataFrame):

        # se o outlier_handle for fit, ele fittara as datas outliers como um dummy
        for outlier in self.outliers:
            df[f'is_outlier_{outlier}'] = np.where(
                df[self.date_col] == outlier, 1, 0)
        return df

    @abstractmethod
    def add_regressor(self, **kwargs):
        pass

    @abstractmethod
    def regressor_coefficients(self, fcst: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, **fit_kwargs):
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class ProphetModel(SupervisedTSModel):

    def __init__(self,
                 date_col: str,
                 target_col: str,
                 freq: str,
                 fit_smooth_config: dict = DEFAULT_FIT_SMOOTH,
                 **model_kwargs: Dict[str, Any]):
        """
        Prophet Model Class
        Args:
            date_col (str): Column that represents date
            target_col (str): Column that represents target
            freq (str): Time frequency alias. See Pandas time-frequency aliases
            fit_smooth_config (dict): Fit smooth configuration. Defaults to {'fit_smooth_config':False}
                                      Arguments:
                                         fit_smooth_config (bool): Boolean to define if smooth error fitting will be used.
                                         window (int): Optional if fit_smooth_config is True. 
                                                       Window of rolling mean that will be applied in the result.
                                                       Defaults to FIT_SMOOTH_FREQ arguments (in __init__.py)
                                         future (str): Can be 'zero','ffill','mean' or 'forecast' (default). 
                                                       Method of predicting future smoothing error.
            **model_kwargs (Dict[str, Any]): Arguments of Prophet object. See in Prophet documentation.
                                      Optional Arguments:
                                         trend_kwargs (dict): Dict to configure trend component of the model. 
                                                              Mainly used to 'logistic' trends.
                                            Arguments:
                                                cap (dict): Cap configuration. It has two arguments:
                                                    mode (str): Can be 'window' or 'constant' (default). 
                                                                If 'window', can pass the 'window' arg, 
                                                                that corresponds to window that will be passed 
                                                                in the rolling_max calculation
                                                floor (dict): Floor configuration. It has two arguments:
                                                    mode (str): Can be 'window' or 'constant' (default). 
                                                                If 'window', can pass the 'window' arg, 
                                                                that corresponds to window that will be passed 
                                                                in the rolling_min calculation
                                         changepoints_kwargs (dict): Dict to configure changepoints of the model. 
                                                                     Not used with trend = 'flat'
                                            Arguments:
                                                cap (dict): Cap configuration. It has two arguments:
                                                    mode (str): Can be 'window' or 'constant' (default). 
                                                                If 'window', can pass the 'window' arg, 
                                                                that corresponds to window that will be passed 
                                                                in the rolling_max calculation
                                                floor (dict): Floor configuration. It has two arguments:
                                                    mode (str): Can be 'window' or 'constant' (default). 
                                                                If 'window', can pass the 'window' arg, 
                                                                that corresponds to window that will be passed 
                                                                in the rolling_min calculation
                                      
        """

        try:
            from prophet import Prophet
        except ImportError:
            raise Exception("Please install Prophet to use this Model Class")

        super().__init__(model=Prophet,
                         freq=freq,
                         date_col=date_col,
                         target_col=target_col)
        self.model: Prophet
        self.kwargs = model_kwargs
        self.changepoints_kwargs = self.kwargs.pop('changepoints_kwargs', {})
        self.trend_kwargs = self.kwargs.pop('trend_kwargs', {})
        self.model = self.model(**self.kwargs)

        self.cap = None
        self.floor = None

        self.fit_smooth_config = fit_smooth_config

        if 'fit_smooth_config' not in self.fit_smooth_config:
            self.fit_smooth_config['fit_smooth_config'] = True

        self.fit_error = pd.DataFrame()
        self.model_raw = None
        self.smooth_model = None

    def add_regressor(self, **kwargs):
        """Method to add regressor to Prophet object. See Prophet.add_regressor documentation for more info"""
        self.model.add_regressor(**kwargs)

    def regressor_coefficients(self, fcst: pd.DataFrame):
        """
        Method to extract regressor coefficients of a Prophet Model.
        Args:
            fcst (pd.DataFrame): Output of forecaster predict.
        Returns:
            new_fcst (pd.DataFrame): Forecaster predict output enriched
        """

        try:
            from prophet import utilities
        except ImportError:
            raise Exception("Please install Prophet to use this method")

        if len(self.model.extra_regressors) > 0:
            coefs = utilities.regressor_coefficients(self.model)
            coefs['col'] = coefs['regressor'] + '_' + coefs['regressor_mode']
            coefs = coefs.rename(columns={
                'center': 'value_center'
            }).drop(columns=['regressor', 'regressor_mode'])
            coefs = coefs.set_index(['col']).unstack(
                ['col']).to_frame().T.reorder_levels([1, 0], axis=1)
            coefs.columns = ['_'.join(col) for col in coefs.columns]
            coefs = coefs.assign(**{'key': 1})
            new_fcst = fcst.assign(**{
                'key': 1
            }).merge(coefs, on='key').drop(columns='key')

        return new_fcst

    def fit_logistic_terms(self, train_df: pd.DataFrame):
        """
        Method to fit logistic terms (cap and floor) in trend.
        Args:
            train_df (pd.DataFrame): Train dataframe.
        Returns:
            train_df (pd.DataFrame): Train dataframe with logistic terms.
        """

        cap = self.trend_kwargs.get('cap', {'mode': 'constant'})
        cap_mode = cap.get('mode', 'constant')

        # Se tivermos saturadores superiores, inserir no dataframe de futuro
        if cap_mode == 'window':
            cap_window = cap.get('window', 0.2)
            window = int(cap_window * len(train_df.dropna(subset=['y'])))
            train_df['cap'] = train_df['y'].rolling(window,
                                                    center=True,
                                                    min_periods=1).max()

        if cap_mode == 'constant':
            train_df['cap'] = train_df['y'].quantile(0.98)

        cap_pct_limit = cap.get('pct_limit', 0.2)

        train_df['cap'] *= (1 + cap_pct_limit)
        train_df['cap'] = train_df['cap'].fillna(method='backfill').ffill()

        floor = self.trend_kwargs.get('floor', {'mode': 'constant'})
        floor_mode = floor.get('mode', 'constant')

        # Se tivermos saturadores inferiores, inserir no dataframe de futuro
        if floor_mode == 'window':
            floor_window = floor.get('window', 0.2)
            window = int(floor_window * len(train_df.dropna(subset=['y'])))
            train_df['floor'] = train_df['y'].rolling(window,
                                                      center=True,
                                                      min_periods=1).min()

        if floor_mode == 'constant':
            train_df['floor'] = train_df['y'].quantile(0.02)

        floor_pct_limit = floor.get('pct_limit', 0.2)

        train_df['floor'] *= (1 - floor_pct_limit)
        train_df['floor'] = train_df['floor'].fillna(method='backfill').ffill()

        train_df['cap'] = np.where(train_df['cap'] <= train_df['floor'],
                                   train_df['floor'] * 1.01, train_df['cap'])
        train_df['floor'] = np.where(train_df['cap'] <= train_df['floor'],
                                     train_df['cap'] * 0.99, train_df['floor'])
        train_df['cap'] = np.where(
            (train_df['cap'] == train_df['floor']) & (train_df['cap'] == 0),
            1e-6, train_df['cap'])

        if self.cap is None:
            self.cap = train_df[['ds', 'cap']].copy()
        if self.floor is None:
            self.floor = train_df[['ds', 'floor']].copy()

        return train_df

    def set_changepoints(self, train_df: pd.DataFrame):
        """
        Custom algorithm to set prophet changepoints.
        Args:
            train_df (pd.DataFrame): Train DataFrame
        """

        # Pega os parametros de cpr e cps
        cpr = int(len(train_df) * self.model.changepoint_range)

        offset_start = pd.DateOffset(
            **{OFFSET_ARG_FREQ[self.freq]: WINDOW_FREQ[self.freq]})
        offset_end = pd.DateOffset(**{OFFSET_ARG_FREQ[self.freq]: cpr})
        window = WINDOW_FREQ[self.freq]

        # Parametro de minima distancia entre changepoints desejada
        window = self.changepoints_kwargs.get('min_dist_changepoints', 1)

        if 'max_changepoints' not in self.changepoints_kwargs:
            self.changepoints_kwargs['max_changepoints'] = int(
                np.ceil(len(train_df) / (window)))

        # Range de datas aonde se pode encontrar um changepoint
        first_date_possible = pd.to_datetime(
            train_df['ds'].min()) + offset_start
        last_date_possible = pd.to_datetime(train_df['ds'].max()) - offset_end

        changepoints = set_changepoints(
            train_df,
            freq=self.freq,
            range_dates_possible=[first_date_possible, last_date_possible],
            **self.changepoints_kwargs)

        # Removendo changepoints pos training data e resettando ao objeto
        init_date_cv = train_df['ds'].max()
        init_date_cv_start = train_df['ds'].min()
        changepoints = [
            cp for cp in changepoints
            if (pd.to_datetime(cp) < pd.to_datetime(init_date_cv))
            & (pd.to_datetime(cp) > pd.to_datetime(init_date_cv_start))
        ]
        self.model.changepoints = pd.Series(pd.to_datetime(changepoints),
                                            name='ds')
        self.model.n_changepoints = len(self.model.changepoints)

    def _fit_smooth(self, train_df: pd.DataFrame):
        """
        Custom algorithm to add residual of the first fit as a feature to a final model
        Args:
            train_df (pd.DataFrame): Train DataFrame
        """

        fcst: pd.DataFrame = self.model.predict(train_df)

        fit_smooth_params: Dict = self.fit_smooth_config.copy()
        fit_smooth_params.pop('fit_smooth_config',True)
        future = fit_smooth_params.pop('future', 'forecast')
        window = fit_smooth_params.pop('window', FIT_SMOOTH_FREQ[self.freq])

        # Pega valor do forecast, junta com valor real para calcular erro
        fit_error = fcst[['ds', 'yhat']].copy()
        fit_error = fit_error.merge(train_df[['ds',
                                              'y']].copy(),
                                    on='ds',
                                    how='left')
        fit_error['smooth_error'] = np.where(
            fit_error['y'].isna(), np.nan,
            fit_error['yhat'] - fit_error['y'])

        # Metodos de previsao do valor futuro do smooth error
        if future == 'ffill':
            fit_error['smooth_error'] = fit_error['smooth_error'].ffill()
            fit_error['smooth_error'] = fit_error['smooth_error'].fillna(
                method='backfill')

        if future == 'zero':
            fit_error['smooth_error'] = fit_error['smooth_error'].fillna(0)

        if future == 'mean':
            fit_error['smooth_error'] = fit_error['smooth_error'].fillna(
                fit_error['smooth_error'].mean())

        # Se forecast, pega os parametros necessarios e realiza o forecast
        if future == 'forecast':

            from prophet import Prophet

            logistic_kwargs = {}
            logistic_cols = []

            if fit_smooth_params.get('growth','linear') == 'logistic':

                logistic_kwargs = fit_smooth_params.pop('logistic_kwargs',{})

                if 'cap' not in logistic_kwargs:
                    logistic_kwargs.update({'cap': fit_error['smooth_error'].max()})

                if 'floor' not in logistic_kwargs:
                    logistic_kwargs.update({'floor': fit_error['smooth_error'].min()})

                fit_error['cap'] = logistic_kwargs['cap']
                fit_error['floor'] = logistic_kwargs['floor']
                logistic_cols = ['cap','floor']
                self.logistic_kwargs = logistic_kwargs

            self.smooth_model = Prophet(**fit_smooth_params)

            with suppress_stdout_stderr():
                self.smooth_model.fit(fit_error[['ds',
                                                'smooth_error']+logistic_cols].dropna().rename(columns={'smooth_error':'y'}))

            fit_error['smooth_error'] = fit_error['smooth_error'].fillna(
                self.smooth_model.predict(fit_error[['ds']+logistic_cols])['yhat'])

        # Suaviza o erro para entrar como input no modelo final.
        fit_error['smooth_error'] = fit_error['smooth_error'].rolling(
            int(window), center=True, min_periods=1).mean()

        self.fit_error = fit_error[['ds', 'smooth_error']]

        train_df = train_df.merge(self.fit_error, on='ds', how='left')

        # Retreina o modelo com a nova feature de erro suavizado
        self.model_raw.add_regressor(name='smooth_error')

        with suppress_stdout_stderr():
            self.model_raw.fit(train_df)

        self.model = deepcopy(self.model_raw)
        del self.model_raw

    def fit(self, train_df, **fit_kwargs):
        """
        Method that fits Prophet Model
        Args:
            train_df (pd.DataFrame): Train DataFrame
            **fit_kwargs: Keyword arguments of fit method of Prophet object. See Prophet.fit() for more info
        """

        train_df = train_df.rename(columns={
            self.date_col: 'ds',
            self.target_col: 'y'
        })

        self.set_changepoints(train_df)

        if self.model.growth == 'logistic':
            train_df = self.fit_logistic_terms(train_df=train_df)

        if self.fit_smooth_config['fit_smooth_config']:
            self.model_raw = deepcopy(self.model)

        with suppress_stdout_stderr():
            self.model.fit(train_df, **fit_kwargs)

        if self.fit_smooth_config['fit_smooth_config']:
            self._fit_smooth(train_df=train_df)

    def predict_logistic_terms(self, df: pd.DataFrame):
        """
        Method that predict logistic terms to future
        Args:
            df (pd.DataFrame): Future dataframe
        Returns:
            df (pd.DataFrame): Future dataframe with logistic terms
        """

        if self.cap is not None:
            new_df = self.cap.merge(df, on=['ds'],
                                    how='outer').sort_values(['ds'])
            new_df['cap'] = new_df['cap'].ffill()
            df = df.merge(new_df[['ds', 'cap']], how='left')

        if self.floor is not None:
            new_df = self.floor.merge(new_df, on=['ds'],
                                      how='outer').sort_values(['ds'])
            new_df['floor'] = new_df['floor'].ffill()
            df = df.merge(new_df[['ds', 'floor']], how='left')

        return df

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method that make Prophet predictions to future
        Args:
            df (pd.DataFrame): Future dataframe
        Returns:
            fcst (pd.DataFrame): Model results dataframe
        """

        df = df.rename(columns={self.date_col: 'ds', self.target_col: 'y'})

        if self.model.growth == 'logistic':
            df = self.predict_logistic_terms(df)

        if self.outlier_handle == 'fit':
            df = self.set_outliers(df)

        if self.fit_smooth_config['fit_smooth_config']:
            
            logistic_cols = []

            if self.smooth_model.growth == 'logistic':
                df['smooth_cap'] = self.logistic_kwargs['cap']
                df['smooth_floor'] = self.logistic_kwargs['floor']
                logistic_cols = ['smooth_cap','smooth_floor']

            df['smooth_error'] = self.smooth_model.predict(
                df[['ds']+logistic_cols].rename(columns={'smooth_cap':'cap','smooth_floor':'floor'}))['yhat'].values

        fcst = self.model.predict(df)

        if self.model.uncertainty_samples == 0:
            fcst = self.add_uncertainty(fcst)

        return fcst

    def add_uncertainty(self, fcst):
        """
        Method that add uncertainty to Prophet results
        Args:
            fcst (pd.DataFrame): Model results dataframe
        Returns:
            fcst (pd.DataFrame): Model results dataframe with uncertainty columns
        """

        from dags.models.time_series.prophet.ts_uncertainty import add_prophet_uncertainty
        # Adicionando incerteza de forma customizada
        fcst = add_prophet_uncertainty(prophet_obj=self.model,
                                       forecast_df=fcst)
        return fcst

    def save_model(self):
        """
        Method that serialize prophet model to be properly saved as pickle
        """

        try:
            from prophet.serialize import model_to_json
        except ImportError:
            raise Exception("Please install Prophet>=1.1 to use this method")

        self.model = model_to_json(self.model)

        if self.fit_smooth_config['fit_smooth_config']:
            self.smooth_model = model_to_json(self.smooth_model)

        return self

    def load_model(self, saved_model):
        """
        Method that de-serialize json prophet model to be well loaded and used.
        """

        try:
            from prophet.serialize import model_from_json
            from prophet import Prophet
        except ImportError:
            raise Exception("Please install Prophet>=1.1 to use this method")

        self = saved_model
        self.model: Prophet = model_from_json(self.model)

        if self.fit_smooth_config['fit_smooth_config']:
            self.smooth_model: Prophet = model_from_json(self.smooth_model)

        return self


class SimpleTSModel(SupervisedTSModel):

    def __init__(self,
                 date_col: str,
                 target_col: str,
                 freq: str,
                 window: int,
                 center: bool = True,
                 multiple_window: int = 1,
                 treat_zeros: bool = True,
                 agg_method: str = 'mean'):

        super().__init__(model=None,
                         date_col=date_col,
                         freq=freq,
                         target_col=target_col)
        self.freq = freq
        self.date_col = date_col
        self.treat_zeros = treat_zeros
        self.window = window
        self.multiple_window = multiple_window
        self.center = center
        self.agg_method = agg_method

        self.train_df = None

    def add_regressor(self, **kwargs):
        return None

    def fit(self, train_df: pd.DataFrame, **fit_kwargs):

        if self.treat_zeros:
            train_df['yhat_continuous'] = train_df[self.target_col].replace(
                {0: np.nan})  # remove zeros
            train_df['yhat_continuous'] = train_df['yhat_continuous'].ffill(
            )  # aplica ffill para assumir continuidade na serie
            train_df['is_not_zero'] = np.where(train_df[self.target_col] == 0,
                                               0, 1)  # compras zero

        else:
            train_df['yhat_continuous'] = train_df[self.target_col].copy()

            train_df['is_not_zero'] = 1  # constante

        train_df['yhat_continuous'] = train_df['yhat_continuous'].shift(
            1).rolling(
                window=self.window,
                min_periods=int(self.window / 2),
                center=self.center).agg(
                    self.agg_method)  #aplica media movel do valor ffillzado
        train_df['yhat_continuous'] = train_df['yhat_continuous'].ffill()

        train_df['time_wo_buy_mean'] = train_df['is_not_zero'].transform(
            lambda x: round((self.window * self.multiple_window) / (x.rolling(
                (self.window * self.multiple_window),
                min_periods=(self.window * self.multiple_window),
                center=False).sum()), 0))
        train_df['time_wo_buy_mean'] = train_df['time_wo_buy_mean'].ffill()

        self.train_df = train_df.copy()

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:

        assert self.train_df is not None, "Your model has not been fitted"

        df = df.merge(self.train_df, on=[self.date_col], how='left')
        df['yhat_continuous'] = df['yhat_continuous'].ffill()
        df['time_wo_buy_mean'] = df['time_wo_buy_mean'].ffill()

        df['y_zero'] = df[self.target_col].fillna(
            0)  # assume zero pra nans e pro futuro
        df['is_not_zero_2'] = np.where(df['y_zero'] == 0, 0, 1)  # compras zero
        df['zero_cycles'] = df['is_not_zero_2'].cumsum()  # ciclos entre zeros

        df['time_wo_buy'] = 1
        df['time_wo_buy'] = df.groupby(
            ['zero_cycles'])['time_wo_buy'].cumsum()  # tempo entre compras
        df['will_buy'] = np.where(
            df['time_wo_buy'] == 1, 1,
            np.where(df['time_wo_buy'] % df['time_wo_buy_mean'] == 0, 1, 0)
        )  # resto da divisao entre zero cycles e time wo buy mean igual a zero me da exatamente os pontos aonde terei o tempo medio sem compra respeitado.

        df['yhat'] = df['yhat_continuous'] * df['will_buy']
        df = df[[
            self.date_col, 'yhat', 'yhat_continuous', 'will_buy',
            'time_wo_buy_mean'
        ]]

        return df


class SklearnTSModel(SupervisedTSModel):
    pass