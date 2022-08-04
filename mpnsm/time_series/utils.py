import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ipp_ds.data_cleaning.feature_engineering import new_date_encoder
from dags.models.time_series import *


class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


def add_seasonality(df: pd.DataFrame,
                    freq: str,
                    name: str,
                    period: int,
                    tipo: str = 'fourier',
                    fourier_order: int = 3,
                    num_cycles: int = None,
                    create_binaries: bool = False,
                    remove_unvariant_cycles: bool = False,
                    **kwargs):
    """
    Function that adds seasonality as an external regressor.
    This was done so that seasonal features could be created according to the desired time frequency.

    Args:
        df (pd.DataFrame): DataFrame with features
        freq (str): Pandas Frequency alias. Pode ser 'Y', 'M' ou 'd'
        name (str): Seasonality feature name
        fourier_order (int): Seasonality Fourier Order. Number of sines and cosines that will be fitted in model.
        period (int): Seasonality period in freq units.
        num_cycles (int): Set number of seasonality cycles
        create_binaries (bool): Create binary data for every period seasonality
        remove_unvariant_cycles (bool): Remove cycles with unvariant data
    Returns:
        df2 (pd.DataFrame): DataFrame with seasonal features
        fou_terms (list): List with column names of seasonal features
    """

    if (tipo == 'fourier') & (fourier_order == 0):
        raise "You're trying to fit seasonality with tipo=fourier and fourier_order=0"

    df2 = df.copy()
    df2['t'] = new_date_encoder(pd.to_datetime(df2['ds']),
                                freq=freq,
                                date_start='1989-12-31')
    df2['unit_cycle'] = (df2['t'] / int(period)).astype('int')
    df2[f'{name}_cycle'] = (df2['t'] % int(period)).astype('int')

    num_cycles = int(num_cycles) if num_cycles is not None else max(
        5, int((len(df2) / int(period)) * 0.25))

    seas_cols = []

    if tipo == 'fourier':
        #cada termo corresponde a sen(2*pi*f*t) / cos(2*pi*f*t), sendo f frequencia
        for i in range(fourier_order):
            for fun_name, fun in {'sin': np.sin, 'cos': np.cos}.items():
                df2[f'{name}_{fun_name}_{i}'] = fun(
                    2.0 * (i + 1) * np.pi * df2['t'].values / int(period))
                seas_cols += [f'{name}_{fun_name}_{i}']

    else:
        df2['full_cycle'] = df2.groupby(['unit_cycle'])['y'].transform('count')
        df2['y_mod'] = np.where(df2['full_cycle'] == int(period), df2['y'],
                                np.nan)
        df2['y_mod'] = df2['y_mod'] - df2['y_mod'].rolling(
            num_cycles * 2, min_periods=1, center=True).median()
        df2['y_mod'] -= df2.groupby(['unit_cycle'])['y_mod'].transform('min')
        df2['y_mod'] += 1
        df2[f'seas_{name}'] = df2.groupby(['unit_cycle'
                                           ])['y_mod'].transform('sum')
        df2[f'seas_{name}'] = df2['y_mod'] / df2[f'seas_{name}']

        if tipo == 'constant':
            df2[f'seas_{name}'] = df2.groupby(
                [f'{name}_cycle'])[f'seas_{name}'].transform('median')

        if tipo == 'rolling':
            df2[f'seas_{name}'] = df2.groupby(
                [f'{name}_cycle'])[f'seas_{name}'].transform(
                    lambda x: x.rolling(num_cycles, min_periods=1).median())
            df2[f'seas_{name}'] = df2.groupby([f'{name}_cycle'
                                               ])[f'seas_{name}'].shift(1)

        df2[f'seas_{name}'] = df2.groupby([f'{name}_cycle'
                                           ])[f'seas_{name}'].ffill()
        df2[f'seas_{name}'] = df2.groupby([f'{name}_cycle'
                                           ])[f'seas_{name}'].backfill()
        df2[f'seas_{name}'] /= df2.groupby(['unit_cycle'
                                            ])[f'seas_{name}'].transform('sum')

        df2[f'seas_{name}'] = np.where(df2['full_cycle'] == int(period),
                                       df2[f'seas_{name}'], np.nan)
        df2[f'seas_{name}'] = df2.groupby([f'{name}_cycle'
                                           ])[f'seas_{name}'].ffill()
        df2[f'seas_{name}'] = df2.groupby([f'{name}_cycle'
                                           ])[f'seas_{name}'].backfill()

        df2[f'seas_{name}'] = df2[f'seas_{name}'].fillna(1 / int(period))

        if create_binaries:
            for i in range(int(period)):
                df2[f'seas_{name}_{i}'] = np.where(df2[f'{name}_cycle'] == i,
                                                   df2[f'seas_{name}'], 0)
                seas_cols += [f'seas_{name}_{i}']
            df2 = df2.drop(columns=f'seas_{name}')

        else:
            seas_cols += [f'seas_{name}']

    if remove_unvariant_cycles:

        df2[f'{name}_median_cycle'] = df2.groupby([f'{name}_cycle'
                                                   ])['y'].transform('median')
        df2[f'{name}_min_cycle'] = df2.groupby([f'{name}_cycle'
                                                ])['y'].transform('min')
        df2['is_inv_cycle'] = np.where(
            df2[f'{name}_median_cycle'] == df2[f'{name}_min_cycle'], 1, 0)

        if len(df2) <= int(period) * 5:
            df2['is_inv_cycle'] = 0

        var_df = df2.loc[df2['is_inv_cycle'] == 0, seas_cols]
        seas_cols = var_df.loc[:, var_df.min() != var_df.max()].columns

        del var_df

    df2 = df2.drop(
        columns=['unit_cycle', 't', 'y_mod', 'full_cycle', f'{name}_cycle'],
        errors='ignore')

    return df2, seas_cols


def identify_outliers(df2,
                      target_col='y',
                      method='quantile',
                      limit=0.02,
                      max_outliers=0.06,
                      remove_first=0.06,
                      remove_last=0.06,
                      minimum_to_abort=0.3,
                      freq='M',
                      special_events=[],
                      remove_dummy_variables_effect=True,
                      window=None):
    """
    Função para identificação de outliers de uma série historica

    Args:
        df2 (pd.DataFrame): DataFrame com coluna de data (ds) e variavel (y)
        method (str): Método de identificação de outlier. pode ser quantile ou iqr
        limit (float): Se metodo = quantile, quantil da cauda a ser cortado.
                       Se método = iqr, numero de iqrs pra definir os limites superior e inferior.
        max_outliers (number): Se < 1, percentual da base total maximo de outliers possiveis. Se >=1, numero maximo de outliers
        remove_first (int): Remove primeiros % de registros da base pra nao encontrar outliers nele
        remove_last (int): Remove ultimo % de registros da base pra nao encontrar outliers nele
        minimum_to_abort (float): Mínimo percentual da base considerada outlier para que ignore a existência de outliers
        freq (str): frequência de data do pandas
        special_events(list): Lista de indices de eventos especiais
        remove_dummy_variables_effect: Remove dummies da serie
        window: Define periodo considerado na sazonalidade

    Returns:
        pd.DataFrame: DataFrame com outliers removidos
    """

    df = df2.copy().rename(columns={target_col: 'y'})

    window = window if window is not None else WINDOW_FREQ[freq]

    # Extrai-se coeficientes sazonais para poder dessazonalizar a serie
    df['date_index'] = new_date_encoder(df['ds'], freq=freq)
    df['unit_cycle'] = df['date_index'] % window
    df['cycle'] = np.ceil(df['date_index'] / window)
    df['cycle'] = df['cycle'].astype(int)
    df['y_trend'] = df['y'].rolling(window, center=True,
                                    min_periods=1).median()
    df['y_detrend'] = df['y'] / df['y_trend']
    df['sazon'] = df.groupby('unit_cycle')['y_detrend'].transform(
        lambda x: x.median())

    # Dessazonaliza a serie e depois compara com a media movel para pegar maiores distorçoes
    df['y_desazon'] = df['y'] / df['sazon']
    df['y_des_ma'] = df['y_desazon'].rolling(window,
                                             min_periods=1,
                                             center=True).median()
    df['noise'] = df['y_desazon'] - df['y_des_ma']

    # Removendo datas com variaveis dummy da analise de outliers
    if remove_dummy_variables_effect:
        dummy_variables = df.columns[df.isin([0, 1]).all()]
        if len(dummy_variables) > 0:
            df['is_dummy_date'] = df[dummy_variables].sum(axis=1)
            df = df.loc[df['is_dummy_date'] == 0].copy()

    # Removendo eventos especiais da analise de outliers
    if len(special_events) > 0:
        special_events = pd.json_normalize(special_events,
                                           record_path=['dates'
                                                        ])['date'].tolist()
        for event in special_events:
            if not isinstance(event, list):
                event = [event]
            for date in event:
                if isinstance(date, tuple):
                    df = df.loc[~(df['ds'].between(*date))].copy()
                elif isinstance(date, str):
                    df = df.loc[~(df['ds'] == date)].copy()

    #Evita que se encontre outliers no inicio e final da série
    arg_name = OFFSET_ARG_FREQ[freq]

    remove_first_arg = {arg_name: int(np.ceil(len(df) * remove_first))}
    remove_last_arg = {arg_name: int(np.ceil(len(df) * remove_last))}

    first_full_date = df['ds'].min() + pd.DateOffset(**remove_first_arg)
    last_full_date = df['ds'].max() - pd.DateOffset(**remove_last_arg)

    #Somente permite identificar outliers nesse intervalo
    df = df.loc[df['ds'].between(first_full_date, last_full_date)]

    #Se o filtro acima faz o dataframe ficar vazio, retorne zero outliers
    if len(df) == 0:
        return []

    #Se o método é quantile, ele usa limit sendo o quantil da cauda a ser cortado.
    #Ex: Se limit = 0.02, ele corta entre 0.02 e 0.98

    if method == 'quantile':
        outliers = df[(df['noise'] <= df['noise'].quantile(limit)) |
                      (df['noise'] >= df['noise'].quantile(1 - limit))]

    #Se o método é iqr, ele usa limit como o número de IQRs a serem removidos a partir dos quantis de 0.25 e 0.75.

    elif method == 'iqr':
        iqr_lower = df['noise'].quantile(0.25) - limit * (
            df['noise'].quantile(0.75) - df['noise'].quantile(0.25))
        iqr_upper = df['noise'].quantile(0.75) + limit * (
            df['noise'].quantile(0.75) - df['noise'].quantile(0.25))
        outliers = df[(df['noise'] <= iqr_lower) | (df['noise'] >= iqr_upper)]

    # Após o filtro advindo do método, o identificador corta iterativamente uma data de cima e uma data de baixo
    # até chegar ao critério de max_outliers

    signal = 1

    outliers = outliers.sort_values('noise')

    if (len(outliers) / len(df) > minimum_to_abort):
        return []

    if max_outliers < 1:
        cond = len(outliers) / len(df) > max_outliers

    else:
        cond = len(outliers) > max_outliers

    while cond:

        if signal == 1:
            outliers = outliers.sort_values('noise', ascending=False).tail(-1)
        else:
            outliers = outliers.sort_values('noise', ascending=True).tail(-1)

        cond = len(outliers) > max_outliers if max_outliers >= 1 else len(
            outliers) / len(df) > max_outliers

        signal *= -1

    list_outliers = list(outliers['ds'].dt.strftime('%Y-%m-%d').unique())

    return list_outliers


def set_changepoints(df: pd.DataFrame,
                     variable: str = 'y',
                     date_col: str = 'ds',
                     freq: str = 'M',
                     range_dates_possible: list = [],
                     max_changepoints: int = 6,
                     min_dist_changepoints: int = 10,
                     manual_changepoints: list = [],
                     plot=False):
    """
    Algorithm to set candidate changepoints for Prophet Model.
    It takes the largest second derivatives of the smoothed curve.

    Args:
        df (pd.DataFrame): DataFrame with date column and desired target variable
        variable (str): Variable column name
        date_col (str): Date column name
        freq (str): Pandas frequency alias.
        range_dates_possible (list): Range of possible dates to find changepoints.
        max_changepoints_per_derivative (int): Maximum number of changepoints to be fitted.
        min_dist_changepoints (int): Minimum distance between changepoints.
        manual_changepoints (list): List of dates with changepoints to be manually inserted in final list.

    Returns:
        list: List of candidate changepoints dates
    """

    # Initial handle
    df = df.rename(columns={variable: 'y', date_col: 'ds'})
    a = df.set_index('ds')['y'].dropna().to_frame('y')
    a['modified'] = a['y'].copy()

    range_dates_possible = range_dates_possible if len(
        range_dates_possible) > 0 else [a.index.min(),
                                        a.index.max()]

    # Smoothing curve with moving average
    window = WINDOW_FREQ[freq]

    for _ in range(3):
        a['modified'] = a['modified'].rolling(window,
                                              min_periods=1,
                                              center=True).mean().shift(-1)

    # Extracting second derivatives
    b = {}
    a['diff0'] = a['modified'].copy()
    a[f'diff2'] = a[f'diff0'].diff().diff()
    a[f'diff_abs2'] = a[f'diff2'].abs()

    secure_margin = {'M': 5, 'D': 25, 'h': 50}

    c = a.loc[range_dates_possible[0]:range_dates_possible[1], :].sort_values(
        f'diff_abs2', ascending=False)
    pot_cp = c[f'diff_abs2'].head(max_changepoints * secure_margin[freq]).index
    passed = []

    # Applying filter to guarantee min_dist_changepoints and maximum number of changepoints
    for date in pot_cp:
        if date in c.index:
            c['distance'] = np.abs(
                (pd.to_datetime(date) - pd.to_datetime(c.index)) //
                np.timedelta64(1, freq))
            c = c[~(c['distance'].between(0, min_dist_changepoints))
                  & ~(c.index.isin(passed))]
            passed += [date]

    b[1] = a.loc[passed].sort_values(f'diff_abs2', ascending=False).head(
        int(max_changepoints)).index

    list_cp = list(b[1]) + manual_changepoints

    # To plot if desired
    if plot:

        color = {1: 'orange'}

        fig, ax = plt.subplots()

        sns.lineplot(data=a, x=a.index, y='y', ax=ax)

        a['modified'].plot(ax=ax, ls='dashed', color='red')

        for diff, cps in b.items():
            for index in cps:
                ax.axvline(index, ls='--', color=color[diff])

        ax.set_title(variable)

        fig.show()

    return list_cp
