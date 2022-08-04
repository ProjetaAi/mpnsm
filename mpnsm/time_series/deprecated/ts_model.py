import gc
from copy import deepcopy
import pandas as pd
import numpy as np

from prophet import Prophet, utilities
import dill

import warnings
warnings.filterwarnings('ignore')

from demanda_vendas.models.utils import suppress_stdout_stderr
from demanda_vendas.models.time_series import *
from demanda_vendas.models.time_series.ts_uncertainty import add_prophet_uncertainty
from demanda_vendas.models.time_series.ts_utils import identify_outliers, set_changepoints, add_seasonality, eval_results, add_calculated_regressors

def make_future_dataframe(df,
                          train_dict,
                          regressors_list,
                          calc_cols,
                          freq):
    
    """
    Função que constroi o dataframe de features a ser inputado no predict

    Args:
        df (pd.DataFrame): DataFrame com features e target
        train_dict (dict): Dicionário com variaveis necessarias para construir o future
        regressors_list (list): Lista de regressores
        calc_cols (list): Lista de colunas que auxiliam no calculo dos regressores calculados
        freq (str): Frequencia da serie
    """

    reg_list_full = regressors_list.copy()

    fut_cols = list(set(['ds','y']+regressors_list+calc_cols))

    # Cria o dataframe future, que sera colocado na ação de predict
    future = df.loc[:, fut_cols].copy()

    # Quando nao tiver valor futuro, preenche futuro com ffill
    future[regressors_list] = future[regressors_list].ffill()

    # Preenche valores nans historicos com backfill
    future[regressors_list] = future[regressors_list].backfill()

    # Cria features de eventos especiais
    if 'special_events' in list(train_dict.keys()):
        for event in train_dict['special_events']:
            feature_name = f'IS_{event["name"]}'
            future[feature_name] = 0
            for entry in event["dates"]:
                dates = entry["date"]
                value = entry["value"] if "value" in entry else 1
                if not isinstance(dates, list):
                    dates = [dates]
                for date in dates:
                    if isinstance(date, str):
                        future[feature_name] = np.where(future['ds'] == date, value, future[feature_name])
                    elif isinstance(date, tuple):
                        future[feature_name] = np.where(future['ds'].between(*date), value, future[feature_name])
            reg_list_full += [feature_name]

    # Adiciona os dummy outliers, se outliers_handle = fit
    if 'outliers' in list(train_dict.keys()):
        if train_dict['outlier_handle'] == 'fit':
            for outlier in train_dict['outliers']:
                future[f'IS_OUTLIER_{outlier}'] = np.where(future['ds']==outlier, 1, 0)
                reg_list_full += [f'IS_OUTLIER_{outlier}']

    # Adiciona os sinais de degrau 'big_changers'
    if 'big_changers' in list(train_dict.keys()):

        for big_changer in train_dict['big_changers']:

            if isinstance(big_changer, str):
                future[f'AFTER_{big_changer}'] = np.where(future['ds']>=big_changer, 1, 0)
                reg_list_full += [f'AFTER_{big_changer}']

            if isinstance(big_changer, tuple):
                future[f'BETWEEN_{big_changer[0]}_{big_changer[1]}'] = np.where(future['ds'].between(big_changer[0], big_changer[1]), 1, 0)
                reg_list_full += [f'BETWEEN_{big_changer[0]}_{big_changer[1]}']

    # Adiciona as componentes sazonais
    if 'seasonalities' in list(train_dict.keys()):
        unv_dfs = pd.DataFrame()
        for params in train_dict['seasonalities'].to_dict(orient='records'):
            future, seas_cols, unv_df = add_seasonality(future, freq=freq, **params)
            reg_list_full += seas_cols
            if len(unv_df) > 0:
                unv_dfs = pd.concat([unv_dfs, unv_df], sort=True, ignore_index=True)

    # Se tivermos saturadores superiores, inserir no dataframe de futuro
    if train_dict['trend'] == 'logistic':

        if train_dict['cap']['mode'] == 'window':
            window = int(train_dict['cap']['window']*len(future.dropna(subset=['y'])))
            future['cap'] = future['y'].rolling(window, center=True, min_periods=1).max()

        if train_dict['cap']['mode'] == 'constant':
            future['cap'] = future.loc[~future['ds'].isin(train_dict['outliers']), 'y'].max()

        future['cap'] *= (1+train_dict['cap']['pct_limit'])
        future['cap'] = future['cap'].backfill().ffill()

        # Se tivermos saturadores inferiores, inserir no dataframe de futuro
        if train_dict['floor']['mode'] == 'window':
            window = int(train_dict['floor']['window']*len(future.dropna(subset=['y'])))
            future['floor'] = future['y'].rolling(window, center=True, min_periods=1).min()

        if train_dict['floor']['mode'] == 'constant':
            future['floor'] = future.loc[~future['ds'].isin(train_dict['outliers']), 'y'].min()

        future['floor'] *= (1-train_dict['floor']['pct_limit'])
        future['floor'] = future['floor'].backfill().ffill()

        future['cap'] = np.where(future['cap'] <= future['floor'], future['floor']*1.01, future['cap'])
        future['floor'] = np.where(future['cap'] <= future['floor'], future['cap']*0.99, future['floor'])
        future['cap'] = np.where((future['cap']==future['floor']) & (future['cap']==0), 1e-6, future['cap'])

    return future, reg_list_full, unv_dfs

def fit(df_prop, rec_control = 0, **kwargs):

    """
    Função para ajustar o estimador do prophet
    Args:
        df_prop (pd.DataFrame): DataFrame com dados necessários
        prophet_mode (str): Modo do Prophet. Pode ser additive ou multiplicative.
        changepoints (list): Lista de changepoints candidatos
        seasonalities (dict): Dicionario com sazonalidades e parametros relacionados
        regressors (dict): Dicionario com regressores e parametros relacionados
        cps (float): Parametro changepoint_prior_scale do prophet
        cpr (float): Parametro changepoint_range do prophet
        int_wid (float): Indice de confiança do modelo
        mcmc_samples (int): Numero de amostras a serem usadas nas cadeias MCMC para geração da incerteza

    Returns:
        model (Prophet): Instancia do modelo Prophet
        df_prop (pd.DataFrame): DataFrame de treino
    """

    #Definição de valores padrão caso nao se encontrem os valores
    changepoints = kwargs.get('changepoints',[])
    outliers = kwargs.get('outliers',[])
    outlier_handle = kwargs.get('outlier_handle','fit')
    drop_intervals = kwargs.get('drop_intervals', [])
    big_changers = kwargs.get('big_changers',[])
    special_events = kwargs.get('special_events',[])
    seasonalities = kwargs.get('seasonalities', pd.DataFrame())
    regressors = kwargs.get('regressors', pd.DataFrame())
    mcmc_samples = kwargs.get('mcmc_samples', 0)
    prophet_mode = kwargs.get('prophet_mode', 'multiplicative')
    trend = kwargs.get('trend','linear')
    cap = kwargs.get('cap', {})
    floor = kwargs.get('floor', {})

    cps = kwargs.get('cps', 0.0625)
    cpr = kwargs.get('cpr', 0.8)
    int_wid = kwargs.get('int_wid',0.95)
    freq = kwargs.get('freq','M')

    # Criação de instancia do Prophet
    model = Prophet(growth=trend, #tipo de sinal de tendencia. pode ser linear, flat ou logistic. caso logistic, exige parametro cap a ser adicionado no df_prop e no future
                    seasonality_mode=prophet_mode, #modo da equação a ser fittada. pode ser multiplicative ou additive.
                    daily_seasonality=False, #nós desligamos as sazonalidades padrao para poder adicionar as customizadas pelo add_seasonality
                    weekly_seasonality=False,
                    yearly_seasonality=False,
                    changepoints=changepoints, #nós passamos os changepoints customizados conforme o algoritmo desenvolvido pela casa, que permite espaçamento e encontra pontos melhores que o padrao do Prophet
                    changepoint_prior_scale=cps, #parametro de regularização de changepoint
                    changepoint_range = cpr, #parametro que define limite maximo de changepoint. ele é inutilizado quando se passa o parametro changepoints.
                    interval_width = int_wid, #índice de confiança do range de incerteza
                    uncertainty_samples=None, #passamos None pois estamos overridando o modo de se calcular incertezas do Prophet por uma função custom
                    mcmc_samples = mcmc_samples) #numero de amostras a serem fittadas nas cadeias markovianas que ajudam na estimação de incerteza dos coeficientes. se 0, o prophet usa estimação por média a posteriori.

    # Adição das sazonalidades, se existentes. Elas serão adicionadas como regressores externos e cada componente de fourier sera adicionada separadamente, que nem o Prophet faz por trás.
    if len(seasonalities) > 0:

        for params in seasonalities.to_dict(orient='records'):

            df_prop, cols, _ = add_seasonality(df_prop, freq=freq, **params)

            for col in cols:

                mode = prophet_mode

                if 'mode' in params:
                    mode = params['mode']

                standardize = 'auto'
                if 'standardize' in params:
                    standardize = params['standardize']

                prior_scale = model.seasonality_prior_scale
                if 'prior_scale' in params:
                    prior_scale = params['prior_scale']

                model.add_regressor(col, mode=mode, standardize=standardize, prior_scale=prior_scale)

    # Adição dos regressores externos, se existentes.
    # cada regressor pode ter quatro parametros passados:
    # name: nome do regressor. é mandatorio passar esse parametro.
    # mode: additive ou multiplicative. sera adicionado a equação de fit como um termo aditivo (+ coef*regressor) ou multiplicativo (+ trend*coef*regressor)
    # prior_scale: escala a priori que indica pro otimizador um conhecimento inicial do peso do coeficiente.
    # standardize: se voce quer normalizar ou não a variável

    if len(regressors) > 0:

        reg_df = regressors.copy()
        reg_df.columns = reg_df.columns.str.replace('regressor_','')

        if 'mode' in reg_df.columns:
            reg_df['mode'] = reg_df['mode'].fillna(prophet_mode)
        else:
            reg_df['mode'] = prophet_mode

        if 'prior_scale' in reg_df.columns:
            reg_df['prior_scale'] = reg_df['prior_scale'].fillna(model.holidays_prior_scale)
        else:
            reg_df['prior_scale'] = model.holidays_prior_scale

        if 'prior_mean' in reg_df.columns:
            reg_df['prior_mean'] = reg_df['prior_mean'].fillna(None)
        else:
            reg_df['prior_mean'] = None

        if 'standardize' in reg_df.columns:
            reg_df['standardize'] = reg_df['standardize'].fillna('auto')
        else:
            reg_df['standardize'] = 'auto'

        if 'constraint' in reg_df.columns:
            reg_df['constraint'] = reg_df['constraint'].fillna('unconstrained')
        else:
            reg_df['constraint'] = 'unconstrained'

        for params in reg_df.to_dict(orient='records'):
            model.add_regressor(name = params['name'],
                                prior_scale = params['prior_scale'],
                                standardize = params['standardize'],
                                #constraint = params['constraint'], #quando a migração pro git com a mod for efetivada, descomentar essa linha de codigo
                                mode = params['mode'])

    # Cria features de eventos especiais
    for event in special_events:
        feature_name = f'IS_{event["name"]}'
        df_prop[feature_name] = 0
        for entry in event["dates"]:
            dates = entry["date"]
            value = entry["value"] if "value" in entry else 1
            if not isinstance(dates, list):
                dates = [dates]
            for date in dates:
                if isinstance(date, str):
                    df_prop[feature_name] = np.where(df_prop['ds'] == date, value, df_prop[feature_name])
                elif isinstance(date, tuple):
                    df_prop[feature_name] = np.where(df_prop['ds'].between(*date), value, df_prop[feature_name])

        ev_copy = event.copy()
        ev_copy.pop('name')
        ev_copy.pop('dates')

        model.add_regressor(feature_name, **ev_copy)

    fit_outliers = []
    # Se a estrategia de lidar com outliers for remover, ele remove as datas e esvazia a lista de outliers para nao serem fittados no fit
    if outlier_handle == 'drop':
        df_prop = df_prop.loc[~df_prop['ds'].isin(outliers)]
    elif outlier_handle is False:
        fit_outliers = []
    else:
        fit_outliers = outliers.copy()

    # removendo intervalos indesejados do período de treinamento
    for interval in drop_intervals:
        df_prop = df_prop.loc[~df_prop['ds'].between(interval[0], interval[1])]

    # se o outlier_handle for fit, ele fittara as datas outliers como um dummy
    for outlier in fit_outliers:
        df_prop[f'IS_OUTLIER_{outlier}'] = np.where(df_prop['ds']==outlier, 1, 0)
        model.add_regressor(f'IS_OUTLIER_{outlier}')

    # se algum big changer for passado, ele adicionara um sinal degrau, onde o antes da data é zero e o ela+futuro = 1.
    for big_changer in big_changers:

        if isinstance(big_changer, str):
            df_prop[f'AFTER_{big_changer}'] = np.where(df_prop['ds']>=big_changer, 1, 0)
            model.add_regressor(f'AFTER_{big_changer}')

        if isinstance(big_changer, tuple):
            df_prop[f'BETWEEN_{big_changer[0]}_{big_changer[1]}'] = np.where(df_prop['ds'].between(big_changer[0], big_changer[1]), 1, 0)
            model.add_regressor(f'BETWEEN_{big_changer[0]}_{big_changer[1]}')

    # Se tivermos saturadores superiores, inserir no dataframe de futuro
    if trend == 'logistic':

        if cap['mode'] == 'window':
            window = int(cap['window']*len(df_prop))
            df_prop['cap'] = df_prop['y'].rolling(window, center=True, min_periods=1).max()

        if cap['mode'] == 'constant':
            df_prop['cap'] = df_prop['y'].max()

        df_prop['cap'] *= (1+cap['pct_limit'])
        df_prop['cap'] = df_prop['cap'].backfill().ffill()

        # Se tivermos saturadores inferiores, inserir no dataframe de futuro
        if floor['mode'] == 'window':
            window = int(floor['window']*len(df_prop))
            df_prop['floor'] = df_prop['y'].rolling(window, center=True, min_periods=1).min()

        if floor['mode'] == 'constant':
            df_prop['floor'] = df_prop['y'].min()

        df_prop['floor'] *= (1-floor['pct_limit'])
        df_prop['floor'] = df_prop['floor'].backfill().ffill()

        df_prop['cap'] = np.where(df_prop['cap'] <= df_prop['floor'], df_prop['floor']*1.01, df_prop['cap'])
        df_prop['floor'] = np.where(df_prop['cap'] <= df_prop['floor'], df_prop['cap']*0.99, df_prop['floor'])
        df_prop['cap'] = np.where((df_prop['cap']==df_prop['floor']) & (df_prop['cap']==0), 1e-6, df_prop['cap'])

    # retirando todos os valores NaN depois dos cálculos
    df_prop = df_prop.dropna(how='all', axis=1).dropna()

    # Removendo changepoints pos training data e resettando ao objeto
    init_date_cv = df_prop['ds'].max()
    init_date_cv_start = df_prop['ds'].min()
    changepoints = [cp for cp in changepoints if (pd.to_datetime(cp) < pd.to_datetime(init_date_cv)) & (pd.to_datetime(cp) > pd.to_datetime(init_date_cv_start))]
    model.changepoints = pd.Series(pd.to_datetime(changepoints), name='ds')
    model.n_changepoints = len(model.changepoints)

    # Fazendo copia crua do modelo
    model_raw = deepcopy(model)

    # Treino do modelo do prophet. Utilizamos a suppress_stdout_stderr para evitar verbose desnecessario
    try:
        with suppress_stdout_stderr():
            model.fit(df_prop, iter = 1000)
            return model, df_prop, model_raw

    except Exception as e:
        try:
            print(e)
            print(f'Algoritmo falhou. Tentando trocar otimizador para Newton')
            with suppress_stdout_stderr():
                model.history = None
                model.fit(df_prop, algorithm='Newton',iter = 1000)
                return model, df_prop, model_raw
        except:
            if rec_control == 1:
                raise f'O modelo não conseguiu convergir com as opções dadas.'
            kwargs.pop('trend')
            rec_control = 1
            print(f'Algoritmo falhou para convergir com trend=logistic. Trocando para trend=linear')
            return fit(df_prop, rec_control = rec_control, **kwargs)

def predict_step(future,
                 df_prop,
                 model_obj,
                 train_dict,
                 reg_list_full):

    # Executa a previsão de todo o horizonte em um step só
    if train_dict['predict_mode'] == 'full':
        fcst = model_obj.predict(future)

    # Executa a previsão passo a passo do horizonte.
    if train_dict['predict_mode'] == 'stepwise':

        last_date = df_prop['ds'].max()
        horizon = len(future.loc[future['ds']>last_date])
        start = future.loc[future['ds']==last_date].index+1

        fcst = model_obj.predict(future.dropna(subset=['y']))
        fcst_future = pd.DataFrame(data=None, index=range(start[0], start[0]+horizon), columns=fcst.columns)
        fcst = pd.concat([fcst, fcst_future])

        for step in range(horizon):  

            fcst.loc[start+step, :] = model_obj.predict(future.loc[start+step, :]).values

            if 'y' in future.columns:
                future.loc[start+step, 'y'] = fcst.loc[start+step, 'yhat'].values

            future, _ = add_calculated_regressors(future, regressors=train_dict['regressors'])

            # Preenche com ffill os valores futuros nans a cada iteração
            future[reg_list_full] = future[reg_list_full].ffill()

            # Preenche com backfill os valores iniciais a cada iteração
            future[reg_list_full] = future[reg_list_full].backfill()
    
    # Adicionando incerteza de forma customizada
    fcst = add_prophet_uncertainty(prophet_obj=model_obj, 
                                   forecast_df=fcst)

    return fcst, future

def predict(df: pd.DataFrame, 
            train_dict: dict, 
            regressors_list: list, 
            freq: str, 
            model,
            model_raw):
    
    """
    Função que realiza a previsão dos resultados do modelo.
    Args:
        df (pd.DataFrame): DataFrame com features e targets
        train_dict (dict): Dicionario com variaveis necessarias para realizar a previsao
        regressors_list (list): Lista de regressores
        freq (str): Frequencia da serie temporal
        model: Objeto de modelo treinado.
        model_raw: Objeto de modelo não-treinado
    Returns:
        fcst (pd.DataFrame): DataFrame de forecast final
        reg_list_full (list): Lista de regressores incrementada
        model: Objeto de modelo treinado final
        future (pd.DataFrame): DataFrame future usado na previsão
    """

    # Calcula-se dataframe futuro com features finais e incrementa lista de regressores
    future, reg_list_full, unv_dfs = make_future_dataframe(df=df, 
                                                           train_dict=train_dict, 
                                                           regressors_list=regressors_list, 
                                                           calc_cols=train_dict['calc_cols'], 
                                                           freq=freq)

    # Dropa-se valores com y=nan para ter o dataset de treino
    df_prop = future.dropna(subset=['y'])
    
    # Executa-se a previsão pura
    fcst, future = predict_step(future = future, 
                                df_prop = df_prop,
                                model_obj = model,
                                train_dict = train_dict,
                                reg_list_full = reg_list_full)

    #Fitta residuo suavizado
    if train_dict['fit_smooth_config']['fit_smooth_error']:

        # Pega valor do forecast, junta com valor real para calcular erro
        fit_error = fcst[['ds','yhat']].copy()
        fit_error = fit_error.merge(df_prop[['ds','y']].copy(), on='ds', how='left')
        fit_error['SMOOTH_ERROR'] = np.where(fit_error['y'].isna(), np.nan, fit_error['yhat'] - fit_error['y'])

        # Metodos de previsao do valor futuro do smooth error
        if train_dict['fit_smooth_config']['future'] == 'ffill':
            fit_error['SMOOTH_ERROR'] = fit_error['SMOOTH_ERROR'].ffill().backfill()
        
        if train_dict['fit_smooth_config']['future'] == 'zero':
            fit_error['SMOOTH_ERROR'] = fit_error['SMOOTH_ERROR'].fillna(0)
        
        if train_dict['fit_smooth_config']['future'] == 'mean':
            fit_error['SMOOTH_ERROR'] = fit_error['SMOOTH_ERROR'].fillna(fit_error['SMOOTH_ERROR'].mean())

        # Se forecast, pega os parametros necessarios e realiza o forecast
        if train_dict['fit_smooth_config']['future'] == 'forecast':

            fit_smooth_params = train_dict['fit_smooth_config'].copy()

            limit_vals = {}

            for param in ['future','window','fit_smooth_error','cap_limit','floor_limit']:
                if param in fit_smooth_params:
                    if param in ['cap_limit','floor_limit']:
                        limit_vals.update({param: fit_smooth_params[param]})
                    fit_smooth_params.pop(param)
            
            fit_smooth_params['uncertainty_samples'] = None

            limit_cols = []

            fit_error_growth = 'linear'
            if 'growth' in fit_smooth_params:
                fit_error_growth = fit_smooth_params['growth']

            if fit_error_growth == 'logistic':

                if 'cap_limit' in limit_vals:
                    cap_limit = limit_vals['cap_limit']
                else:
                    cap_limit = 0.2

                if 'floor_limit' in limit_vals:
                    floor_limit = limit_vals['floor_limit']
                else:
                    floor_limit = 0.2

                fit_error['cap'] = fit_error['SMOOTH_ERROR'].max() * (1+cap_limit)
                fit_error['floor'] = fit_error['SMOOTH_ERROR'].min() * (1-floor_limit)

                limit_cols = ['cap','floor']

            try:
                with suppress_stdout_stderr():
                    fit_error_model = Prophet(**fit_smooth_params)
                    fit_error_model.fit(fit_error[['ds','SMOOTH_ERROR']+limit_cols].dropna().rename(columns={'SMOOTH_ERROR':'y'}), iter = 1000)
            
            except:
                fit_smooth_params['growth'] = 'flat'
                with suppress_stdout_stderr():
                    fit_error_model = Prophet(**fit_smooth_params)
                    fit_error_model.fit(fit_error[['ds','SMOOTH_ERROR']+limit_cols].dropna().rename(columns={'SMOOTH_ERROR':'y'}), iter = 1000)
                
            fit_error['SMOOTH_ERROR'] = fit_error['SMOOTH_ERROR'].fillna(fit_error_model.predict(fit_error[['ds']+limit_cols])['yhat'])
        
        # Suaviza o erro para entrar como input no modelo final.
        fit_error['SMOOTH_ERROR'] = fit_error['SMOOTH_ERROR'].rolling(int(train_dict['fit_smooth_config']['window']), center=True, min_periods=1).mean()

        fit_error = fit_error[['ds','SMOOTH_ERROR']]

        df_prop = df_prop.merge(fit_error, on='ds', how='left')
        future = future.merge(fit_error, on='ds', how='left')

        # Retreina o modelo com a nova feature de erro suavizado
        with suppress_stdout_stderr():
            model_raw.add_regressor(name='SMOOTH_ERROR', mode='additive')
            model_raw.fit(df_prop)
        
        # Guardo minha previsão com valor puro e reseto valores futuros de y
        future['Y_PURE'] = future['y'].copy()
        future['y'] = np.where(future['ds'] > df_prop['ds'].max(), np.nan, future['y'])
        
        # Realiza a previsão final
        fcst, future = predict_step(future = future, 
                                    df_prop = df_prop,
                                    model_obj = model_raw,
                                    train_dict = train_dict,
                                    reg_list_full = reg_list_full)

        # Trazendo o objeto model pro novo model fittado com o residuos
        model = model_raw

        # Adicionando a lista de regresssores finais
        reg_list_full += ['SMOOTH_ERROR']

    # Concatena os resultados dos ciclos invariantes
    if len(unv_dfs) > 0:
        unv_dfs['yhat_upper'] = unv_dfs['yhat'] * 1.05
        unv_dfs['yhat_lower'] = unv_dfs['yhat'] * 0.95
        fcst = pd.concat([fcst, unv_dfs], sort=True, ignore_index=True)
        fcst = fcst.sort_values('ds')
        
    return fcst, reg_list_full, model, future

def TimeSeriesOptimize(train_df, complete_df, train_dict, init_date, unit, variable, prop_params):

    """
    Função que realiza a otimização dos hiperparâmetros do Prophet.
    Pode-se otimizar os hiperparametros changepoint prior scale e changepoint range (afetando limite final de onde um changepoint pode ser encontrado).
    Pode se otimizar usando as seguintes estrategias:
        - joint, que faz uma otimização conjunta de erro e range de confiança,
        - erro, que faz uma otimização de erro
        - ci, que faz uma otimização de range
    A otimização é feita fazendo uma validação cruzada em x meses a serem definidos pelo parametro validation_months do prop_params.

    Args:
        train_df (pd.DataFrame):
        complete_df: (pd.DataFrame): DataFrame com todos os dados
        train_dict (dict): Dicionário com parametros auxiliares de treino do prophet
        init_date (str): Data limite de treino
        unit (str): Identificadores da granularidade
        variable (str): Nome da variável
        prop_params (dict): Dicionário de hiperparâmetros do Prophet
    Returns:
        parameter_dict (dict): Dicionário com parametros otimizados
    """

    freq = train_dict['freq']

    verbose = 1
    if 'verbose' in list(prop_params.keys()):
        verbose = prop_params['verbose']

    #Dessa forma, eu ja trago ele dentro do parameter_dict e removo do parameters

    standard_parameter = {'cpr':('additive','erro'), 'cps':('multiplicative','joint')}
    parameter_dict = {key: prop_params[f'{key}_start'] for key in ['cpr','cps'] if prop_params[f'optimize_{key}']==False}
    parameters = {key: standard_parameter[key] for key in ['cpr','cps'] if prop_params[f'optimize_{key}']}

    counter_params = 0

    # Para cada parametro a ser otimizado, eu realizo validação cruzada e escolho melhor parametro dado criterio definido: 
    for parameter, (step_mode, parameter_strategy) in parameters.items():
        
        # Definição da lista de valores a serem testados (start, end, step e step_mode)
        parameter_start = prop_params['cps_start'] if parameter=='cps' else prop_params['cpr_start']
        parameter_end = prop_params['cps_end'] if parameter=='cps' else prop_params['cpr_end']
        step = prop_params['cps_step'] if parameter=='cps' else prop_params['cpr_step']

        mult = lambda x, y: x*y
        add = lambda x, y: x+y
        div = lambda x, y: x/y
        sub = lambda x, y: x-y

        if step_mode == 'multiplicative':
            func_step = mult if parameter_start>parameter_end else div
            inv_step = div if parameter_start>parameter_end else mult

        if step_mode == 'additive':
            func_step = add if parameter_start>parameter_end else sub
            inv_step = sub if parameter_start>parameter_end else add

        #Parametros de otimização

        # Minimo ganho de intervalo de confiança entre iterações
        min_ci_gain = prop_params['min_ci_gain']

        # Minimo aceitavel de intervalo de confiança
        min_ci_acc = prop_params['min_ci_acc']

        # Indice de confiança do modelo
        int_width = prop_params['int_width']

        # Erro maximo perdido entre iterações
        max_erro_step = prop_params['max_erro_step'] if parameter == 'cps' else 0.003

        # Perda máxima de confiança entre iterações
        max_ci_loss = prop_params['max_ci_loss']

        # Periodo de validação cruzada
        validation_period = prop_params['validation_period']

        # Passo da validação cruzada
        validation_step = 1

        if 'validation_step' in prop_params.keys():
            validation_step = prop_params['validation_step']

        dict_results = {}

        counter = 0

        # Calcula-se horizonte de futuro oficial a ser usado nas validações cruzadas.
        last_real_date = complete_df.dropna(subset=['y'])['ds'].max()
        default_horizon = len(complete_df.loc[complete_df['ds']>last_real_date])

        # Para cada step da validação cruzada, ele otimiza o hiperparametro desejado
        for i in range(validation_period, -1, -validation_step):
            
            # Se na metade dos valores eu tiver tido valores iguais pro hiperparametro, ele vira meu valor final
            max_trials = int(validation_period/validation_step)

            if counter >= max_trials/2:
                df_results_partial = pd.DataFrame.from_dict(dict_results, orient='index')
                if df_results_partial[parameter].min() == df_results_partial[parameter].max():
                    if verbose >= 2:
                        print('Optimization converged')
                    break

            param = parameter_start
            ci_last = 0
            erro_last = 0
            run_next = True

            offset = pd.DateOffset(**{OFFSET_ARG_FREQ[freq]:i})

            init_date_cv = pd.to_datetime(init_date) - offset

            dict_results[init_date_cv] = {}

            while run_next:
                
                # Limita minha base de treino até o limite da validação cruzada
                df_prop = train_df.reset_index()
                df_prop = df_prop.loc[df_prop['ds'] <= init_date_cv].copy()

                # Limita meu horizonte da complete_df até default_horizon
                comp_df = complete_df.loc[complete_df['ds'] <= (pd.to_datetime(init_date_cv) + pd.DateOffset(**{OFFSET_ARG_FREQ[freq]:default_horizon}))].copy()
                
                # Evita que meu y do complete_df da iteração da validação cruzada tenha valor a frente de init_date_cv
                comp_df['y'] = np.where(comp_df['ds'] <= init_date_cv, comp_df['y'], np.nan)

                # Calcula o changepoint range
                if 'cpr' in parameter_dict:
                    units_cpr = int(parameter_dict['cpr'])

                else:
                    units_cpr = int(prop_params['cpr_start']) if parameter != 'cpr' else int(param)

                cpr = (len(df_prop)-units_cpr)/len(df_prop)

                # Valor default caso cpr caia fora do intervalo de 0 e 1
                if (cpr <= 0) | (cpr >= 1):
                    cpr = 0.8

                if 'cps' in parameter_dict:
                    cps = parameter_dict['cps']

                else:
                    cps = prop_params['cps_start'] if parameter != 'cps' else param
                
                # Parametros para o set_changepoints
                
                offset_start = pd.DateOffset(**{OFFSET_ARG_FREQ[freq]:WINDOW_FREQ[freq]})
                offset_end = pd.DateOffset(**{OFFSET_ARG_FREQ[freq]:units_cpr})
                window = WINDOW_FREQ[freq]

                # Minima distancia entre changepoints desejada
                if 'min_dist_cp' in list(prop_params.keys()):
                    window = prop_params['min_dist_cp']

                num_max_cp = int(np.ceil(len(df_prop)/(window)))

                # Numero maximo de changepoints desejado
                if 'num_max_cp' in list(prop_params.keys()):
                    num_max_cp = prop_params['num_max_cp']

                # Definição dos limites ate onde pode se encontrar changepoints
                first_date_possible = pd.to_datetime(df_prop['ds'].min()) + offset_start
                last_date_possible = pd.to_datetime(df_prop['ds'].max()) - offset_end

                # Algoritmo de escolha dos changepoints
                changepoints = set_changepoints(df_prop,
                                                range_dates_possible=[first_date_possible, last_date_possible],
                                                freq = freq,
                                                max_changepoints=num_max_cp,
                                                min_dist_changepoints=window,
                                                manual_changepoints=train_dict['man_cp'])

                #Fixing manual changepoints bug in cross validation
                changepoints = [cp for cp in changepoints if pd.to_datetime(cp) < pd.to_datetime(init_date_cv)]

                # Treina o modelo
                model, df_prop, model_raw = fit(df_prop=df_prop,
                                        changepoints=changepoints,
                                        cps=cps,
                                        cpr=cpr,
                                        int_wid=int_width,
                                        **train_dict)

                regressors_list = []

                if 'regressor_name' in train_dict['regressors'].columns:
                    regressors_list = list(train_dict['regressors']['regressor_name'].unique())

                # Executa a previsão desejada.
                fcst, regressors_list, model, _ = predict(df=comp_df, 
                                                          train_dict=train_dict, 
                                                          regressors_list=regressors_list, 
                                                          freq=freq, 
                                                          model=model,
                                                          model_raw=model_raw)

                # Avalia os resultados
                ci, in_sample, out_sample = eval_results(train_df=df_prop,
                                                        valid_df=train_df.reset_index(),
                                                        fcst=fcst,
                                                        init_date=init_date_cv,
                                                        type_error=prop_params['type_error'],
                                                        freq=freq,
                                                        period_eval=prop_params['period_eval'])

                del df_prop
                del fcst
                gc.collect()

                if verbose >= 3:
                    print(f'Variable: {variable} - Unit: {unit} - InitialMonth: {init_date_cv} - {parameter}: {param} - CI: {round(ci,4)*100}% - Erro In-Sample: {round(in_sample["erro"],4)*100}% - Erro Out-Sample: {round(out_sample["erro"],4)*100}% - 4' )

                # Duas estrategias de otimização de hiperparametros:
                # joint: busca otimizar erro e intervalo de confiança
                # ci/erro: busca otimizar a metrica desejada.

                if parameter_strategy == 'joint':

                    if (ci < (ci_last-max_ci_loss)) | ((in_sample["erro"] > (erro_last + max_erro_step)) & (erro_last != 0)):
                        dict_results[init_date_cv]['ci'] = ci_last
                        dict_results[init_date_cv]['erro_in'] = erro_last
                        dict_results[init_date_cv]['erro_out'] = erro_out_last
                        dict_results[init_date_cv][parameter] = func_step(param, step)
                        if verbose >= 2:
                            print(f'{parameter}: {func_step(param, step)} chosen to {unit}: Variable: {variable} - InitialMonth: {init_date_cv} - CI: {round(ci_last,4)*100}% - Erro In-Sample: {round(in_sample["erro"],4)*100}% - Erro Out-Sample: {round(out_sample["erro"],4)*100}% - 1' )
                        break

                    if (ci >= min_ci_acc)&(abs(ci-ci_last) < min_ci_gain):

                        if (in_sample["erro"] <= (erro_last-max_erro_step)):
                            dict_results[init_date_cv]['ci'] = ci
                            dict_results[init_date_cv]['erro_in'] = in_sample["erro"]
                            dict_results[init_date_cv]['erro_out'] = out_sample["erro"]
                            dict_results[init_date_cv][parameter] = param
                            if verbose >=2:
                                print(f' {parameter}: {param} chosen to {unit}: Variable: {variable} - InitialMonth: {init_date_cv} - CI: {round(ci,4)*100}% - Erro In-Sample: {round(in_sample["erro"],4)*100}% - Erro Out-Sample: {round(out_sample["erro"],4)*100}% - 2' )
                            break

                        else:
                            dict_results[init_date_cv]['ci'] = ci_last
                            dict_results[init_date_cv]['erro_in'] = erro_last
                            dict_results[init_date_cv]['erro_out'] = erro_out_last
                            dict_results[init_date_cv][parameter] = func_step(param, step)
                            if verbose >= 2:
                                print(f'{parameter}: {func_step(param, step)} chosen to {unit}: Variable: {variable} - InitialMonth: {init_date_cv} - CI: {round(ci_last,4)*100}% - Erro In-Sample: {round(in_sample["erro"],4)*100}% - Erro Out-Sample: {round(out_sample["erro"],4)*100}% - 1' )
                            break

                    ci_last = ci
                    erro_last = in_sample["erro"]
                    erro_out_last = out_sample["erro"]
                    param = inv_step(param, step)

                    cond = param < parameter_end if parameter_start > parameter_end else param > parameter_end

                    if cond:
                        dict_results[init_date_cv]['ci'] = ci
                        dict_results[init_date_cv]['erro_in'] = in_sample["erro"]
                        dict_results[init_date_cv]['erro_out'] = out_sample["erro"]
                        dict_results[init_date_cv][parameter] = func_step(param, step)
                        if verbose >= 2:
                            print(f'{parameter}: {func_step(param, step)} chosen to {unit}: Variable: {variable} - InitialMonth: {init_date_cv} - CI: {round(ci,4)*100}% - Erro In-Sample: {round(in_sample["erro"],4)*100}% - Erro Out-Sample: {round(out_sample["erro"],4)*100}% - 3')
                        run_next = False

                else:

                    if ((parameter_strategy == 'ci') & (ci > ci_last)) | ((parameter_strategy == 'erro') & (in_sample["erro"] < erro_last)) | (erro_last == 0):

                        dict_results[init_date_cv]['ci'] = ci
                        dict_results[init_date_cv]['erro_in'] = in_sample["erro"]
                        dict_results[init_date_cv]['erro_out'] = out_sample["erro"]
                        dict_results[init_date_cv][parameter] = param

                    ci_last = ci
                    erro_last = in_sample["erro"]
                    erro_out_last = out_sample["erro"]
                    param = inv_step(param, step)

                    cond = param < parameter_end if parameter_start > parameter_end else param > parameter_end

                    if cond:

                        if verbose >= 2:
                            print(f'{parameter}: {dict_results[init_date_cv][parameter]} chosen to {unit}: Variable: {variable} - InitialMonth: {init_date_cv} - CI: {round(dict_results[init_date_cv]["ci"],4)*100}% - Erro In-Sample: {round(in_sample["erro"],4)*100}% - Erro Out-Sample: {round(out_sample["erro"],4)*100}% - 3' )

                        run_next = False

            counter += 1

        df_results = pd.DataFrame.from_dict(dict_results, orient='index')

        # Pega parametro por mediana dos resultados
        parameter_dict[parameter] = df_results[parameter].median()

        if parameter == 'cpr':
            parameter_dict[parameter] = int(parameter_dict[parameter])

        counter_params += 1

        if counter_params == len(parameters):
            parameter_dict[f'erro_out'] = df_results['erro_out'].median()

    return parameter_dict

def TimeSeriesForecast(df, variable, regressors, prop_params) -> pd.DataFrame:

    """
    Função que realiza a previsão utilizando-se da ferramenta Prophet
    Args:
        df (pd.DataFrame): DataFrame com variáveis necessárias
        variable (str): Nome da variável a ser projetada
        regressors (pd.DataFrame): DataFrame com informações dos regressores
        prop_params (dict): Dicionário com parametros do Prophet
    Returns:
        pd.DataFrame: DataFrame com previsões realizadas
    """

    try:
        # Definindo dataset de treino
        train_df_or = df.dropna(subset=['y'])
        init_date = train_df_or['INITIAL_DATE'].max()
        train_df_or = train_df_or.drop(columns=['INITIAL_DATE'])

        # Filtrando nos regressors somente os registros relativos a unidade sendo fittada
        possible_grans = regressors['data_gran'].values[0]

        for col in possible_grans:
            if col in regressors.columns:
                regressors = regressors.loc[regressors[col].isin([train_df_or[col].unique()[0]])]

        cols_dup = [col for col in regressors.columns if col in ['regressor_name','seasonality_name']]

        if len(cols_dup) > 0:
            assert len(regressors) == len(regressors.drop_duplicates(subset=cols_dup)), "Foi encontrada duplicidade na lista de regressores passadas ao algoritmo"

        # Remove-se parametros dos regressors totalmente vazios
        regressors = regressors.dropna(how='all', axis=1)

        # Define parametros de fit caso existentes, senão, colocam valores standard.
        try:
            freq = regressors['freq'].values[0]
        except:
            freq = 'M'

        # Changepoints manuais. Changepoints desejados no modelo independente do algoritmo
        try:
            man_cp = regressors['manual_changepoints'].values[0]
        except:
            man_cp = []

        # Grandes mudanças. Para modelar grandes quedas/subidas de patamar.
        try:
            big_changers = regressors['big_changers'].values[0]
        except:
            big_changers = []

        # Modo do prophet. Pode ser additive ou multiplicative.
        try:
            prophet_mode = regressors['prophet_mode'].values[0]
        except:
            prophet_mode = 'multiplicative'
        
        # Modo de previsão do modelo. Pode ser full (de uma vez só) ou stepwise (passo a passo)
        try:
            predict_mode = regressors['predict_mode'].values[0]
        except:
            predict_mode = 'full'

        # Data inicial de treino do modelo
        try:
            initial_date = regressors['initial_date'].values[0]
        except:
            initial_date = train_df_or['ds'].min().strftime('%Y-%m-%d')
        
        # Tipo de tendencia do modelo
        try:
            trend = regressors['trend'].values[0]
        except:
            trend = 'linear'

        # Calculo de cap e floor necessarios para definir os limites da regressao logistica do trend.
        cap = {}

        if 'cap.window' not in regressors.columns:
            cap['window'] = 0.20
        elif regressors['cap.window'].isna().all():
            cap['window'] = 0.20
        else:
            cap['window'] = regressors['cap.window'].values[0]

        if 'cap.pct_limit' not in regressors.columns:
            cap['pct_limit'] = 0.20
        elif regressors['cap.pct_limit'].isna().all():
            cap['pct_limit'] = 0.20
        else:
            cap['pct_limit'] = regressors['cap.pct_limit'].values[0]

        if 'cap.mode' not in regressors.columns:
            cap['mode'] = 'constant'
        elif regressors['cap.mode'].isna().all():
            cap['mode'] = 'constant'
        else:
            cap['mode'] = regressors['cap.mode'].values[0]

        floor = {}

        if 'floor.window' not in regressors.columns:
            floor['window'] = 0.20
        elif regressors['floor.window'].isna().all():
            floor['window'] = 0.20
        else:
            floor['window'] = regressors['floor.window'].values[0]

        if 'floor.pct_limit' not in regressors.columns:
            floor['pct_limit'] = 0.20
        elif regressors['floor.pct_limit'].isna().all():
            floor['pct_limit'] = 0.20
        else:
            floor['pct_limit'] = regressors['floor.pct_limit'].values[0]

        if 'floor.mode' not in regressors.columns:
            floor['mode'] = 'constant'
        elif regressors['floor.mode'].isna().all():
            floor['mode'] = 'constant'
        else:
            floor['mode'] = regressors['floor.mode'].values[0]

        # Separa variaveis sazonais
        seasonalities = pd.DataFrame()

        try:
            seasonalities = regressors.loc[:, regressors.columns.str.contains('seasonality')].copy().drop_duplicates(subset=['seasonality_name'])
            seasonalities = seasonalities.dropna(how='all',axis=1)
            seasonalities.columns = seasonalities.columns.str.replace(f'seasonality_','')

            if 'fourier_order' in seasonalities.columns:
                seasonalities['fourier_order'] = seasonalities['fourier_order'].fillna(3).astype('int')

            if 'num_cycles' in seasonalities.columns:
                seasonalities['num_cycles'] = seasonalities['num_cycles'].fillna(0).astype('int')

            if 'tipo' in seasonalities.columns:
                seasonalities['tipo'] = seasonalities['tipo'].fillna('fourier')

            if 'standardize' in seasonalities.columns:
                seasonalities['standardize'] = seasonalities['standardize'].fillna('auto')

            if 'create_binaries' in seasonalities.columns:
                seasonalities['create_binaries'] = seasonalities['create_binaries'].fillna(False)

            if 'remove_unvariant_cycles' in seasonalities.columns:
                seasonalities['remove_unvariant_cycles'] = seasonalities['remove_unvariant_cycles'].fillna(False)

        except:
            seasonalities = pd.DataFrame()

        # Eventos especiais a serem modelados
        try:
            special_events = regressors['special_events'].values[0]
        except:
            special_events = []
        
        #Outlier configuration
        outliers_config = regressors.loc[:,regressors.columns.str.contains('outliers_config')].reset_index(drop=True)
        outliers_config.columns = outliers_config.columns.str.replace('outliers_config.','')
        outliers_config = outliers_config.to_dict(orient='index')

        if len(outliers_config) > 0:
            outliers_config = outliers_config[0]

        # Metodo de lidar com outliers. Pode ser 'fit', 'drop' ou False.
        try:
            outlier_handle = regressors['outlier_handle'].values[0]
        except:
            outlier_handle = 'fit'
        
        # Se a tendencia for logistica, o default de lidar com outliers será dropar e o prophet_mode default será additive.
        if trend == 'logistic':
            outlier_handle = 'drop'
            prophet_mode = 'additive'
        
        #Fit smooth configuration
        fit_smooth_config = regressors.loc[:,regressors.columns.str.contains('fit_smooth_config')].reset_index(drop=True)
        fit_smooth_config.columns = fit_smooth_config.columns.str.replace('fit_smooth_config.','')
        fit_smooth_config = fit_smooth_config.to_dict(orient='index')

        if len(fit_smooth_config) > 0:
            fit_smooth_config = fit_smooth_config[0]
        
        # Parametros default do fit smooth error
        default_fsc = {'fit_smooth_error':False, 
                       'window': FIT_SMOOTH_FREQ[freq], 
                       'future':'forecast',
                       'growth':'flat'}

        default_fsc.update(fit_smooth_config)
        fit_smooth_config = default_fsc

        # Setta os parametros customizados para a serie do prop_params
        prop_params_list = [col for col in regressors.columns if 'prop_params' in col]
        prop_params_reg = prop_params.copy()

        for col in prop_params_list:
            prop_params_reg.update({col.replace('prop_params.',''): regressors[col].values[0]})

        # Se trend==flat, default é nao otimizar.
        if trend == 'flat':
            prop_params_reg.update({'cps_start':1, 'cps_end': 1, 'cpr_start':6, 'cpr_end':6})

        #Dropando períodos desejados:
        drop_intervals = []
        if 'drop_interval' in regressors.columns:
            for interval in regressors['drop_interval'].values[0]:
                drop_intervals += [interval]
        
        # Pega lista de regressores final

        if 'regressor_name' in regressors.columns:
            regressors_df = regressors.loc[:, regressors.columns.str.contains('regressor')].drop_duplicates(subset=['regressor_name'])
            regressors_list = list(regressors_df['regressor_name'].unique())

        else:
            regressors_df = pd.DataFrame()
            regressors_list = []

        #Reconvertendo função de calculos
        if 'regressor_calc_func' in regressors_df.columns:
            regressors_df['regressor_calc_func'] = regressors_df['regressor_calc_func'].apply(lambda x: x if x is None else dill.loads(x))

        #Adicionando regressores calculados
        calc_cols = []
        if ('regressor_is_calc' in regressors_df.columns) and ('regressor_name' in regressors_df.columns):
            df, calc_cols = add_calculated_regressors(df, regressors_df)

        # Removendo valores antes da data inicial desejada da série
        if 'initial_date' in regressors.columns:
            df = df.loc[df['ds'] >= initial_date]
        
        #Recalcula train_df_or
        train_df_or = df.dropna(subset=['y'])

        # Extrai o nome da unidade que representa a serie historica
        unit = list(train_df_or[col].unique()[0] for col in train_df_or.columns if col in possible_grans)

        # Quando não tem valor do regressor para a série a ser prevista
        train_df = train_df_or.dropna(how='all', axis=1)
        regressors_list = [col for col in regressors_list if col in train_df.columns]
        train_df = train_df.loc[:, train_df.columns.isin(['ds','y']+regressors_list)]

        # Realiza a idenficação dos registros que sao considerados outliers
        outliers = identify_outliers(train_df, freq=freq,
                                     special_events=special_events, **outliers_config)

        train_df = train_df.set_index('ds')

        # Cria a lista de regressores final a ser fittada no modelo.
        regressors_list = list(train_df.columns.drop(['y']))

        if 'regressor_name' in regressors_df.columns:
            regressors_df = regressors_df.loc[regressors_df['regressor_name'].isin(regressors_list)]

        # Constroi o dicionario de treino, a ser passado na fit
        train_dict = {'regressors':regressors_df, 'seasonalities':seasonalities, 'freq':freq,
                      'man_cp':man_cp, 'outliers':outliers, 'big_changers':big_changers,
                      'prophet_mode': prophet_mode, 'trend':trend, 'cap':cap, 'floor':floor,
                      'special_events': special_events, 'outlier_handle': outlier_handle,
                      'drop_intervals': drop_intervals, 'fit_smooth_config': fit_smooth_config,
                      'predict_mode':predict_mode, 'calc_cols':calc_cols}

        # Realiza a otimização dos hiperparametros com validação cruzada do Prophet
        parameter_dict = TimeSeriesOptimize(train_df=train_df,
                                            complete_df=df,
                                            train_dict=train_dict,
                                            init_date=init_date,
                                            unit=unit,
                                            variable=variable,
                                            prop_params=prop_params_reg)

        df_prop = train_df.reset_index()

        del train_df
        gc.collect()

        # Define parametros finais para o treino do objeto final

        # Filtra para pegar só data com valores reais para a base de treino final
        df_prop = df_prop.loc[df_prop['ds'] <= init_date].copy()

        # Pega os parametros de cpr e cps
        cpr = (len(df_prop)-parameter_dict['cpr'])/len(df_prop)

        if (cpr <= 0) | (cpr >= 1):
            cpr = 0.8
        
        offset_start = pd.DateOffset(**{OFFSET_ARG_FREQ[freq]:WINDOW_FREQ[freq]})
        offset_end = pd.DateOffset(**{OFFSET_ARG_FREQ[freq]:parameter_dict['cpr']})
        window = WINDOW_FREQ[freq]

        # Parametro de minima distancia entre changepoints desejada
        if 'min_dist_cp' in list(prop_params_reg.keys()):
            window = prop_params_reg['min_dist_cp']

        # Parametro de numero maximo de changepoints desejado
        num_max_cp = int(np.ceil(len(df_prop)/(window)))

        if 'num_max_cp' in list(prop_params_reg.keys()):
            num_max_cp = prop_params_reg['num_max_cp']

        # Range de datas aonde se pode encontrar um changepoint
        first_date_possible = pd.to_datetime(df_prop['ds'].min()) + offset_start
        last_date_possible = pd.to_datetime(df_prop['ds'].max()) - offset_end

        # Fitta os changepoints conforme algoritmo desenhado na casa, que permite espaçamento minimo entre changepoints.
        changepoints = set_changepoints(df_prop,
                                        range_dates_possible=[first_date_possible, last_date_possible],
                                        max_changepoints=num_max_cp,
                                        freq=freq,
                                        min_dist_changepoints=window,
                                        manual_changepoints = man_cp)

        # Treina o objeto final
        model, df_prop, model_raw = fit(df_prop=df_prop,
                                        changepoints=changepoints,
                                        cps=parameter_dict['cps'],
                                        cpr=cpr,
                                        int_wid=prop_params_reg['int_width'],
                                        mcmc_samples=prop_params_reg['mcmc_samples'],
                                        **train_dict)

        # Preve o modelo com o objeto final
        fcst, reg_list_full, model, future = predict(df=df, 
                                                    train_dict=train_dict, 
                                                    regressors_list=regressors_list, 
                                                    freq=freq, 
                                                    model=model,
                                                    model_raw=model_raw)

        # Avalia os resultados com o modelo final
        ci, in_sample, _ = eval_results(train_df=df_prop, valid_df=pd.DataFrame(),
                                        fcst=fcst, init_date=init_date,
                                        type_error=prop_params_reg['type_error'], freq=freq,
                                        period_eval=prop_params_reg['period_eval'])

        del df_prop
        gc.collect()

        # Gera o dataframe final para debug e analise dos resultados
        verbose = 1

        if 'verbose' in list(prop_params_reg.keys()):
            verbose = prop_params_reg['verbose']

        if verbose >= 1:
            print(f'Hyperparameters chosen to {unit}: Variable: {variable} - CPS: {parameter_dict["cps"]} - CPR - {parameter_dict["cpr"]} - CI: {round(ci,4)*100}% - Erro In-Sample: {round(in_sample["erro"],4)*100}% - Erro Out-Sample: {round(parameter_dict["erro_out"],4)*100}% - 3')

        # Traz para o dataframe final os valores de erro e dos hiperparametros selecionados
        fcst['ERRO_FINAL'] = in_sample['erro']
        fcst['ERRO_OUT'] = parameter_dict['erro_out']
        fcst['CI'] = ci
        fcst['CPR'] = cpr
        fcst['CPS'] = parameter_dict['cps']

        # Adiciona valor real (y) ao dataframe final para analise de erros
        fcst = fcst.merge(train_df_or[['ds','y']], on='ds', how='left')

        del train_df_or
        gc.collect()

        # Adiciona valor dos regressores para analise de impacto das variaveis
        future = future[['ds']+reg_list_full].set_index('ds')
        future.columns = future.columns+'_REAL'
        fcst = fcst.merge(future.reset_index(), on='ds', how='left')

        # Adiciona valor da previsão antes do smooth
        if 'Y_PURE' in future.columns:
            fcst = fcst.merge(future[['ds','Y_PURE']], on='ds', how='left')

        del future
        gc.collect()

        # Adiciona changepoints fittados para debug
        cp_df = pd.DataFrame(model.changepoints, columns=['ds']).assign(**{'IS_CHANGEPOINT':1})
        fcst = fcst.merge(cp_df, on='ds', how='left')
        fcst['IS_CHANGEPOINT'] = fcst['IS_CHANGEPOINT'].fillna(0)

        del cp_df
        gc.collect()

        # Adiciona coeficientes do modelo para debug e simulação dos resultados.
        if len(model.extra_regressors) > 0:
            coefs = utilities.regressor_coefficients(model)
            coefs['col'] = coefs['regressor'] + '_' + coefs['regressor_mode']
            coefs = coefs.rename(columns={'center':'value_center'}).drop(columns=['regressor','regressor_mode'])
            coefs = coefs.set_index(['col']).unstack(['col']).to_frame().T.reorder_levels([1, 0],axis=1)
            coefs.columns = ['_'.join(col) for col in coefs.columns]
            coefs = coefs.assign(**{'key':1})
            fcst = fcst.assign(**{'key':1}).merge(coefs, on='key').drop(columns='key')

        #removendo colunas de outlier
        fcst['IS_OUTLIER_AGG'] = fcst[[col for col in fcst.columns if (col.replace('IS_OUTLIER_','') in outliers) & ('_REAL' not in col)]].sum(axis=1)
        fcst['IS_OUTLIER'] = np.where(fcst['ds'].isin(outliers), 1, 0)

        for out in outliers:
            fcst = fcst.drop(columns=[col for col in fcst.columns if out in col])
        
        fcst.columns = fcst.columns.str.upper()

        return fcst

    except Exception as e:
        print(f'Error in {unit} - {e}')
        return pd.DataFrame()