from abc import abstractmethod
import pandas as pd
import numpy as np
from typing import List, Any

from dags.models.time_series.model import Model
from dags.models.time_series.utils import add_seasonality


class Regressor:
    """Regressor abstract class"""

    def __init__(self, name: str, date_col: str = None, **kwargs):
        self.name = name
        self.kwargs = kwargs
        self.date_col = date_col

    """Regressor abstract class"""

    @abstractmethod
    def add_regressor(self, df: pd.DataFrame, model: Model):
        return model, df


class NormalRegressor(Regressor):
    """Normal Regressor Class."""

    def add_regressor(self, df: pd.DataFrame, model: Model):
        """Method for Adding Normal Regressors. Inputs Directly into ProphetModel's 'add_regressor'"""

        kwargs = self.kwargs.copy()
        kwargs.update({'name': self.name})
        model.add_regressor(**kwargs)

        return model, df


class CalculatedRegressor(Regressor):
    """Calculated Regressor Class."""

    def __init__(self,
                 name: str,
                 calc_cols: List,
                 calc_func: Any,
                 date_col: str = None,
                 **kwargs):
        
        """
        Calculated Regressor Class.
        Args:
        name (str): Regressor's column name
        calc_cols (list): Columns in the calculations
        calc_func (Any): Function to be used in calculation
        date_col (str): Column with date
        """

        super().__init__(name=name, date_col=date_col, **kwargs)
        self.calc_cols = calc_cols
        self.calc_func = calc_func

    def add_regressor(self, df: pd.DataFrame, model: Model):
        """Method for Adding Calculated Regressors"""

        df[self.name] = self.calc_func(df)

        
        kwargs = self.kwargs.copy()
        kwargs.update({'name': self.name})
        model.add_regressor(**kwargs)

        return model, df


class SpecialEventRegressor(Regressor):
    """Special Event Regressor Class."""

    def __init__(self, name: str, dates: List, date_col: str = None, **kwargs):

        """
        Special Event Regressor Class.
        Args:
        name (str): Regressor's column name
        dates (list): List with dicts containing date and value Ex.: [{'date':'2018-05-01','value':10}]
        date_col (str): Column with date
        """

        super().__init__(name=name, date_col=date_col, **kwargs)
        self.dates = dates
        self.feature_name = f'is_{self.name}'

    def add_regressor(self, df: pd.DataFrame, model: Model):
        """Method for Adding Special Event Regressors"""

        df[self.feature_name] = 0

        for entry in self.dates:

            entry_dates = entry["date"]
            value = entry["value"] if "value" in entry else 1

            if not isinstance(entry_dates, list):
                entry_dates = [entry_dates]

            for date in entry_dates:
                if isinstance(date, str):
                    df[self.feature_name] = np.where(df[self.date_col] == date,
                                                     value,
                                                     df[self.feature_name])
                elif isinstance(date, tuple):
                    df[self.feature_name] = np.where(
                        df[self.date_col].between(*date), value,
                        df[self.feature_name])

        kwargs = self.kwargs.copy()
        kwargs.update({'name': self.feature_name})
        model.add_regressor(**kwargs)

        return model, df


class BigChangerRegressor(Regressor):
    """Big Changer Regressor Class."""

    def __init__(self, name: str, dates: List, date_col: str = None, **kwargs):

        """
        Big Changer Regressor Class.
        Args:
        name (str): Regressor's column name
        dates (list): List with all big changer dates, can be str or tuple for 
                        interval big changers. Ex.: ['2021-01-01',('2022-02-01','2022-05-01')]
        date_col (str): Column with date
        """

        super().__init__(name=name, date_col=date_col, **kwargs)
        self.dates = dates

    def add_regressor(self, df: pd.DataFrame, model: Model):
        """Method for Adding Big Changer Regressors"""

        for big_changer in self.dates:

            kwargs = self.kwargs.copy()

            if isinstance(big_changer, str):
                df[f'after_{big_changer}'] = np.where(
                    df[self.date_col] >= big_changer, 1, 0)
                kwargs.update({'name': f'after_{big_changer}'})

            if isinstance(big_changer, tuple):
                df[f'between_{big_changer[0]}_{big_changer[1]}'] = np.where(
                    df[self.date_col].between(big_changer[0], big_changer[1]),
                    1, 0)
                kwargs.update({'name': f'between_{big_changer[0]}_{big_changer[1]}'})
            
            model.add_regressor(**kwargs)

        return model, df


class SeasonalityRegressor(Regressor):
    """Abstract Seasonality Regressor Class."""

    def __init__(self,
                 name: str,
                 period: int,
                 freq: str,
                 date_col: str = None,
                 create_binaries: bool = False,
                 remove_unvariant_cycles: bool = False,
                 **kwargs):

        """
        Abstract Seasonality Regressor Class.
        Args:
        name (str): Regressor's column name
        period (int): Period of the seasonality (12 for yearly seasonality in monthly data)
        freq (int): Frequency's name Ex.: 'M' for monthly and 'D' for daily
        date_col (str): Column with date
        create_binaries (bool): Create binary data for every period seasonality
        remove_unvariant_cycles (bool): Remove cycles with unvariant data
        """

        super().__init__(name=name, date_col=date_col, **kwargs)
        self.period = period
        self.freq = freq
        self.create_binaries = create_binaries
        self.remove_unvariant_cycles = remove_unvariant_cycles

    @abstractmethod
    def add_regressor(self, df: pd.DataFrame, model: Model):
        return model, df


class MedianSeasonalityRegressor(SeasonalityRegressor):
    """Median Seasonality Regressor Class."""

    def __init__(self,
                 name: str,
                 period: int,
                 num_cycles: int,
                 freq: str,
                 date_col: str = None,
                 create_binaries: bool = False,
                 remove_unvariant_cycles: bool = False,
                 **kwargs):

        """
        Abstract Seasonality Regressor Class.
        Args:
        name (str): Regressor's column name
        period (int): Period of the seasonality (12 for yearly seasonality in monthly data)
        num_cycles (int): Set number of seasonality cycles
        freq (int): Frequency's name Ex.: 'M' for monthly and 'D' for daily
        date_col (str): Column with date
        create_binaries (bool): Create binary data for every period seasonality
        remove_unvariant_cycles (bool): Remove cycles with unvariant data
        """

        super().__init__(name=name, date_col=date_col, period=period, 
                         freq=freq, create_binaries=create_binaries, 
                         remove_unvariant_cycles=remove_unvariant_cycles,
                         **kwargs)
        self.num_cycles = num_cycles

    def add_regressor(self, df: pd.DataFrame, model: Model):
        """Method for Adding Median Seasonality Regressors"""

        # adiciona colunas de season no df e nomeia elas no seas_cols
        df, seas_cols = add_seasonality(df=df,
                                        freq=self.freq,
                                        name=self.name,
                                        period=self.period,
                                        tipo='median',
                                        num_cycles=self.num_cycles,
                                        remove_unvariant_cycles=self.remove_unvariant_cycles,
                                        create_binaries=self.create_binaries)

        # adiciona season como regressor
        for col in seas_cols:
            kwargs = self.kwargs.copy()
            kwargs.update({'name':col})
            model.add_regressor(**kwargs)

        return model, df

class FourierSeasonalityRegressor(SeasonalityRegressor):
    """Fourier Seasonality Regressor Class."""

    def __init__(self,
                 name: str,
                 period: int,
                 fourier_order: int,
                 freq: str,
                 date_col: str = None,
                 create_binaries: bool = False,
                 remove_unvariant_cycles: bool = False,
                 **kwargs):

        """
        Fourier Seasonality Regressor Class.
        Args:
        name (str): Regressor's column name
        period (int): Period of the seasonality (12 for yearly seasonality in monthly data)
        fourier_order (int): Number of sines and cosines that will be fitted into the model
        freq (int): Frequency's name Ex.: 'M' for monthly and 'D' for daily
        date_col (str): Column with date
        create_binaries (bool): Create binary data for every period seasonality
        remove_unvariant_cycles (bool): Remove cycles with unvariant data
        """

        super().__init__(name=name, date_col=date_col, period=period, 
                         freq=freq, create_binaries=create_binaries, 
                         remove_unvariant_cycles=remove_unvariant_cycles,
                         **kwargs)
        self.fourier_order = fourier_order

    def add_regressor(self, df: pd.DataFrame, model: Model):
        """Method for Adding Fourier Seasonality Regressors"""

        # adiciona colunas de season no df e nomeia elas no seas_cols
        df, seas_cols = add_seasonality(df=df,
                                        freq=self.freq,
                                        name=self.name,
                                        period=self.period,
                                        tipo='fourier',
                                        fourier_order=self.fourier_order,
                                        remove_unvariant_cycles=self.remove_unvariant_cycles,
                                        create_binaries=self.create_binaries)

        # adiciona season como regressor
        for col in seas_cols:
            kwargs = self.kwargs.copy()
            kwargs.update({'name':col})
            model.add_regressor(**kwargs)

        return model, df


#TODO: Implement Splitted Regressor
class SplittedRegressor(Regressor):
    pass