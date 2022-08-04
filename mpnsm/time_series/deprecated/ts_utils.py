from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from ipp_ds.data_cleaning.feature_engineering import new_date_encoder

from demanda_vendas.models.time_series import *

def identify_outliers(df2,
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

    df = df2.copy()

    window = window if window is not None else WINDOW_FREQ[freq]

    # Extrai-se coeficientes sazonais para poder dessazonalizar a serie
    df['date_index'] = new_date_encoder(df['ds'],freq=freq)
    df['unit_cycle'] = df['date_index']%window
    df['cycle'] = np.ceil(df['date_index']/window)
    df['cycle'] = df['cycle'].astype(int)
    df['y_trend'] = df['y'].rolling(window, center=True, min_periods=1).median()
    df['y_detrend'] = df['y'] / df['y_trend']
    df['sazon'] = df.groupby('unit_cycle')['y_detrend'].transform(lambda x: x.median())

    # Dessazonaliza a serie e depois compara com a media movel para pegar maiores distorçoes
    df['y_desazon'] = df['y'] / df['sazon']
    df['y_des_ma'] = df['y_desazon'].rolling(window, min_periods=1, center=True).median()
    df['noise'] = df['y_desazon'] - df['y_des_ma']

    # Removendo datas com variaveis dummy da analise de outliers
    if remove_dummy_variables_effect:
        dummy_variables = df.columns[df.isin([0,1]).all()]
        if len(dummy_variables) > 0:
            df['is_dummy_date'] = df[dummy_variables].sum(axis=1)
            df = df.loc[df['is_dummy_date']==0].copy()

    # Removendo eventos especiais da analise de outliers
    if len(special_events) > 0:
        special_events = pd.json_normalize(special_events, record_path=['dates'])['date'].tolist()
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

    remove_first_arg = {arg_name: int(np.ceil(len(df)*remove_first))}
    remove_last_arg = {arg_name: int(np.ceil(len(df)*remove_last))}

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
        outliers = df[(df['noise'] <= df['noise'].quantile(limit)) | (df['noise'] >= df['noise'].quantile(1-limit))]

    #Se o método é iqr, ele usa limit como o número de IQRs a serem removidos a partir dos quantis de 0.25 e 0.75.

    elif method == 'iqr':
        iqr_lower = df['noise'].quantile(0.25) - limit*(df['noise'].quantile(0.75)-df['noise'].quantile(0.25))
        iqr_upper = df['noise'].quantile(0.75) + limit*(df['noise'].quantile(0.75)-df['noise'].quantile(0.25))
        outliers = df[(df['noise'] <= iqr_lower) | (df['noise'] >= iqr_upper)]

    # Após o filtro advindo do método, o identificador corta iterativamente uma data de cima e uma data de baixo
    # até chegar ao critério de max_outliers

    signal = 1

    outliers = outliers.sort_values('noise')

    if (len(outliers)/len(df) > minimum_to_abort):
        return []

    if max_outliers < 1:
        cond = len(outliers)/len(df) > max_outliers

    else:
        cond = len(outliers) > max_outliers

    while cond:

        if signal == 1:
            outliers = outliers.sort_values('noise',ascending=False).tail(-1)
        else:
            outliers = outliers.sort_values('noise',ascending=True).tail(-1)

        cond = len(outliers) > max_outliers if max_outliers >= 1 else len(outliers)/len(df) > max_outliers

        signal *= -1

    list_outliers = list(outliers['ds'].dt.strftime('%Y-%m-%d').unique())

    return list_outliers

def set_changepoints(df,
                     variable='y',
                     date_col='ds',
                     freq = 'M',
                     range_dates_possible=[],
                     max_changepoints=6,
                     min_dist_changepoints = 10,
                     manual_changepoints = [],
                     plot=False):

    """
    Algoritmo para setar changepoints candidatos para o prophet.
    Ele pega as maiores segundas derivadas da curva suavizada.

    Args:
        df (pd.DataFrame): DataFrame com data e variável desejada
        variable (str): Nome da coluna da variável
        date_col (str): Nome da coluna de data
        freq (str): String de frequencia de series datetime padrão do pandas.
        range_dates_possible (list): Range de datas possiveis de se encontrar changepoints
        max_changepoints_per_derivative (int): Numero maximo de changepoints a serem encontrados
        min_dist_changepoints (int): Distancia minima entre changepoints
        manual_changepoints (list): Lista de datas com changepoints a serem colocados manualmente pelo usuário

    Returns:
        list: Lista de datas com changepoints candidatos
    """

    # Preparações iniciais necessarias para o algoritmo
    df = df.rename(columns={variable:'y', date_col:'ds'})
    a = df.set_index('ds')['y'].dropna().to_frame('y')
    a['modified'] = a['y'].copy()

    range_dates_possible = range_dates_possible if len(range_dates_possible)>0 else [a.index.min(), a.index.max()]

    # Suavização da curva extraindo média movel
    window = WINDOW_FREQ[freq]

    for _ in range(3):
        a['modified'] = a['modified'].rolling(window, min_periods=1, center=True).mean().shift(-1)

    # Calcula-se as segundas derivadas e extrai as max_changepoints*5 maiores
    b = {}
    a['diff0'] = a['modified'].copy()
    a[f'diff2'] = a[f'diff0'].diff().diff()
    a[f'diff_abs2'] = a[f'diff2'].abs()

    secure_margin = {'M':5,'D':25,'h':50}

    c = a.loc[range_dates_possible[0]:range_dates_possible[1],:].sort_values(f'diff_abs2', ascending = False)
    pot_cp = c[f'diff_abs2'].head(max_changepoints*secure_margin[freq]).index
    passed = []

    # Aplica-se o filtro para se garantir distancia minima (min_dist_changepoints) e extrai as max_changepoints maiores
    for date in pot_cp:
        if date in c.index:
            c['distance'] = np.abs((pd.to_datetime(date) - pd.to_datetime(c.index)) // np.timedelta64(1, freq))
            c = c[~(c['distance'].between(0, min_dist_changepoints))&~(c.index.isin(passed))]
            passed += [date]

    b[1] = a.loc[passed].sort_values(f'diff_abs2', ascending = False).head(int(max_changepoints)).index

    list_cp = list(b[1]) + manual_changepoints

    # Caso interesse ao analista plotar a série com os changepoints e ganhar intuição no método
    if plot:

        color = {1:'orange'}

        fig,ax = plt.subplots()

        sns.lineplot(data=a,x=a.index,y='y',ax=ax)

        a['modified'].plot(ax=ax,ls='dashed',color='red')

        for diff, cps in b.items():
            for index in cps:
                ax.axvline(index,ls='--',color=color[diff])

        ax.set_title(variable)

        fig.show()

    return list_cp

def error(df, type_error):

    """
    Função que calcula erro num dataframe que possua uma coluna "erro" e outra "y" com valores reais.

    Args:
        df (pd.DataFrame): DataFrame com valor de erro e valor real
        type_error (str): Método de cálculo do erro. Pode ser 'weighted','mean_abs_pct' ou 'median_abs_pct'

    Returns:
        Valor: Erro da série
    """

    assert type_error in ['weighted','mean_abs_pct','median_abs_pct'], "Esse método de erro não foi implementado"

    if type_error == 'weighted':
        return np.abs(df['erro']).sum()/df['y'].sum()

    if type_error == 'mean_abs_pct':
        return (np.abs(df['erro'])/np.abs(df['y'])).mean()

    if type_error == 'median_abs_pct':
        return (np.abs(df['erro'])/np.abs(df['y'])).median()


def eval_results(train_df,
                 valid_df,
                 fcst,
                 init_date,
                 type_error='weighted',
                 freq='M',
                 period_eval=0):

    """
    Função para avaliar resultados
    Args:
        train_df (pd.DataFrame): DataFrame com dados de treino
        valid_df (pd.DataFrame): DataFrame com dados de validação
        fcst (pd.DataFrame): DataFrame de previsões
        init_date (str): Data de corte
        type_error (str): Métrica de erro. Pode ser 'weighted', 'mean_abs_pct', 'median_abs_pct'
        freq (str): Frequencia da série. 'M'/'D'
        period_eval (int): Periodo para se avaliar o erro da série (relativo ao ultimo ponto). Se 0, ve série toda.
    Returns:
        erro (int): Métrica de erro agregada
        erro_df (pd.DataFrame): DataFrame com valores reais e previstos
        ci (int): Métrica de range
    """

    # Cria dataframe de erro in-sample (treino)
    erro_in_df = train_df.merge(fcst, on='ds')
    erro_in_df['erro'] = erro_in_df['y']-erro_in_df['yhat']

    # Cria dataframe de erro out-sample (test)
    if len(valid_df) > 0:
        valid_df = valid_df.loc[(valid_df['ds'] > train_df['ds'].max())]
        erro_out_df = valid_df.merge(fcst, on='ds')
        erro_out_df['erro'] = erro_out_df['y']-erro_out_df['yhat']

    # Calcula o offset que define o periodo de analise do erro
    offset = pd.DateOffset(**{OFFSET_ARG_FREQ[freq]:period_eval})

    if period_eval > 0:
        erro_in_filt = erro_in_df.loc[erro_in_df['ds'] >= erro_in_df['ds'].max()-offset]

    erro_in = error(erro_in_filt, type_error=type_error)

    if len(valid_df) > 0:
        erro_out = error(erro_out_df, type_error=type_error)

    fut_ci = fcst.loc[fcst.ds > init_date]
    ci = (1-((np.abs(fut_ci['yhat_upper']-fut_ci['yhat_lower']).sum())/np.abs(fut_ci['yhat']).sum()))
    ci = 0 if ci < 0 else ci

    dict_out = {'ci': ci,
                'in_sample': {'erro':erro_in, 'erro_df':erro_in_df},
                'out_sample': {'erro':np.nan, 'erro_df':pd.DataFrame()}}

    if len(valid_df) > 0:
        dict_out.update({'out_sample':{'erro':erro_out, 'erro_out_df':erro_out_df}})

    return dict_out.values()

def add_seasonality(df,
                    freq,
                    name,
                    period,
                    tipo = 'fourier',
                    fourier_order = 3,
                    num_cycles = None,
                    create_binaries = False,
                    remove_unvariant_cycles = False,
                    **kwargs):

    """
    Função que adiciona sazonalidade como regressor externo.
    Isso foi realizado para que se pudesse criar as features sazonais de acordo a frequencia de tempo desejada.

    Args:
        df (pd.DataFrame): DataFrame com features
        freq (str): Frequency alias do pandas. Pode ser 'Y', 'M' ou 'd'
        name (str): Nome da sazonalidade
        fourier_order (int): Ordem de fourier da sazonalidade. Numero de componentes de cossenos e senos a serem fittados.
        period (int): Periodo da sazonalidade em unidades de tempo de freq.
    Returns:
        df2 (pd.DataFrame): DataFrame com features sazonais
        fou_terms (list): Lista com nome das colunas das features sazonais
    """

    if (tipo == 'fourier') & (fourier_order == 0):
        raise "Você está tentando fittar uma sazonalidade usando tipo=fourier e fourier_order=0"

    df2 = df.copy()
    df2['t'] = new_date_encoder(pd.to_datetime(df2['ds']), freq=freq, date_start = '1989-12-31')
    df2['UNIT_CYCLE'] = (df2['t'] / int(period)).astype('int')
    df2[f'{name}_CYCLE'] = (df2['t'] % int(period)).astype('int')

    num_cycles = int(num_cycles) if num_cycles is not None else max(5, int((len(df2) / int(period)) * 0.25))

    seas_cols = []

    if tipo == 'fourier':
        #cada termo corresponde a sen(2*pi*f*t) / cos(2*pi*f*t), sendo f frequencia
        for i in range(fourier_order):
            for fun_name, fun in {'SIN':np.sin, 'COS':np.cos}.items():
                df2[f'{name}_{fun_name}_{i}'] = fun(2.0 * (i + 1) * np.pi * df2['t'].values / int(period))
                seas_cols += [f'{name}_{fun_name}_{i}']

    else:
        df2['FULL_CYCLE'] = df2.groupby(['UNIT_CYCLE'])['y'].transform('count')
        df2['Y_MOD'] = np.where(df2['FULL_CYCLE'] == int(period), df2['y'], np.nan)
        df2['Y_MOD'] = df2['Y_MOD'] - df2['Y_MOD'].rolling(num_cycles*2, min_periods=1, center=True).median()
        df2['Y_MOD'] -= df2.groupby(['UNIT_CYCLE'])['Y_MOD'].transform('min')
        df2['Y_MOD'] += 1
        df2[f'SEAS_{name}'] = df2.groupby(['UNIT_CYCLE'])['Y_MOD'].transform('sum')
        df2[f'SEAS_{name}'] = df2['Y_MOD'] / df2[f'SEAS_{name}']

        if tipo == 'constant':
            df2[f'SEAS_{name}'] = df2.groupby([f'{name}_CYCLE'])[f'SEAS_{name}'].transform('median')

        if tipo == 'rolling':
            df2[f'SEAS_{name}'] = df2.groupby([f'{name}_CYCLE'])[f'SEAS_{name}'].transform(lambda x: x.rolling(num_cycles, min_periods=1).median())
            df2[f'SEAS_{name}'] = df2.groupby([f'{name}_CYCLE'])[f'SEAS_{name}'].shift(1)

        df2[f'SEAS_{name}'] = df2.groupby([f'{name}_CYCLE'])[f'SEAS_{name}'].ffill()
        df2[f'SEAS_{name}'] = df2.groupby([f'{name}_CYCLE'])[f'SEAS_{name}'].backfill()
        df2[f'SEAS_{name}'] /= df2.groupby(['UNIT_CYCLE'])[f'SEAS_{name}'].transform('sum')

        df2[f'SEAS_{name}'] = np.where(df2['FULL_CYCLE'] == int(period), df2[f'SEAS_{name}'], np.nan)
        df2[f'SEAS_{name}'] = df2.groupby([f'{name}_CYCLE'])[f'SEAS_{name}'].ffill()
        df2[f'SEAS_{name}'] = df2.groupby([f'{name}_CYCLE'])[f'SEAS_{name}'].backfill()

        df2[f'SEAS_{name}'] = df2[f'SEAS_{name}'].fillna(1 / int(period))

        if create_binaries:
            for i in range(int(period)):
                df2[f'SEAS_{name}_{i}'] = np.where(df2[f'{name}_CYCLE']==i, df2[f'SEAS_{name}'], 0)
                seas_cols += [f'SEAS_{name}_{i}']
            df2 = df2.drop(columns=f'SEAS_{name}')

        else:
            seas_cols += [f'SEAS_{name}']

    unv_df = pd.DataFrame()

    if remove_unvariant_cycles:

        df2[f'{name}_MEDIAN_CYCLE'] = df2.groupby([f'{name}_CYCLE'])['y'].transform('median')
        df2[f'{name}_MIN_CYCLE'] = df2.groupby([f'{name}_CYCLE'])['y'].transform('min')

        unv_df = df2.loc[df2[f'{name}_MEDIAN_CYCLE'] == df2[f'{name}_MIN_CYCLE']].copy().rename(columns={f'{name}_MEDIAN_CYCLE':'yhat'}).drop(columns=[f'{name}_MIN_CYCLE'])
        df2 = df2.loc[df2[f'{name}_MEDIAN_CYCLE'] != df2[f'{name}_MIN_CYCLE']].drop(columns=[f'{name}_MIN_CYCLE',f'{name}_MEDIAN_CYCLE'])

        if len(df2) <= int(period)*5:
            df2 = unv_df.copy().drop(columns=['yhat'])
            unv_df = pd.DataFrame()

        const_cols = []
        for col in seas_cols:
            if df2[col].min() == df2[col].max():
                df2 = df2.drop(columns=col)
                const_cols += [col]

        seas_cols = [col for col in seas_cols if col not in const_cols]

    df2 = df2.drop(columns = ['UNIT_CYCLE','t','Y_MOD','FULL_CYCLE', f'{name}_CYCLE'], errors='ignore')
    unv_df = unv_df.drop(columns = ['UNIT_CYCLE','t','Y_MOD','FULL_CYCLE', f'{name}_CYCLE','y'], errors='ignore')

    return df2, seas_cols, unv_df

def add_calculated_regressors(df, regressors):

    """
    Função que permite a adição de regressores calculados a modelagem
    Args:
        df (pd.DataFrame): DataFrame com features
        regressors (pd.DataFrame): DataFrame de configuração
    Returns:
        pd.DataFrame: DataFrame com features calculadas
    """

    is_calc_cols = regressors.loc[regressors['regressor_is_calc']].copy()

    calc_cols = []

    for index, group in is_calc_cols.groupby(['regressor_name']):
        group = group.reset_index(drop=True)
        func = group['regressor_calc_func'].values[0]
        calc_cols += group['regressor_calc_cols'].values[0]
        df[index] = func(df)

    calc_cols = list(set(calc_cols))

    return df, calc_cols