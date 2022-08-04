import ast
import pandas as pd
import numpy as np
import dill
from demanda_vendas.models.time_series.deprecated.ts_model import TimeSeriesForecast
from ipp_ds.data_cleaning.parallelism import applyParallel

import warnings
warnings.filterwarnings('ignore')

import logging
logger = logging.getLogger(__name__)

from ipp_ds.data_cleaning.path import path_join

def fill_nan_units(df, regressors, reg_group_cols):

    """
    Adiciona unidades não explicitamente passadas na regressors
    Args:
        df (pd.DataFrame): DataFrame com séries temporais
        regressors (pd.DataFrame): DataFrame com configuradores
        reg_group_cols (list): Lista de colunas que definem granularidade da tabela
    """

    reg_copy = regressors.copy()
    df_propeth = df.copy()

    for col in reg_group_cols:
        if col not in reg_copy.columns:
            reg_copy[col] = np.nan
        reg_copy.loc[reg_copy[col].isna(), col] = f'[]'
        reg_copy[col] = reg_copy[col].apply(lambda x: x if isinstance(x, list) else ast.literal_eval(x))
        reg_copy = reg_copy.explode(column=col, ignore_index=True)

    nan_rows = False

    cols_full_filled = []

    for col in reg_group_cols:
        nan_rows = nan_rows | (reg_copy[col].isna())
        if reg_copy[col].isna().any() == False:
            cols_full_filled += [col]

    reg_copy['IS_DEFINED'] = np.where(nan_rows, np.nan, 1)

    #Pegando somente as unidades desejadas
    units = df_propeth[reg_group_cols].drop_duplicates().reset_index(drop=True)

    if len(cols_full_filled)>0:
        cols_full_distinct = reg_copy[cols_full_filled].drop_duplicates()
        units = units.merge(cols_full_distinct, on=cols_full_filled)

    reg_full = units.merge(reg_copy, on=reg_group_cols, how='outer').sort_values(['IS_DEFINED']+reg_group_cols, na_position='first')
    reg_full[reg_group_cols] = reg_full[reg_group_cols].fillna('TODOS')

    todos_rows = False
    for col in reg_group_cols:
        todos_rows = todos_rows | (reg_full[col]=='TODOS')

    nunique = reg_full.loc[todos_rows, reg_group_cols].replace({'TODOS':np.nan}).nunique().sort_values(ascending=True)

    if nunique.sum() == 0:

        for col in reg_full.columns:
            reg_full[col] = np.where(reg_full['IS_DEFINED']==1, reg_full[col], reg_full[col].ffill())

    else:

        for i in range(len(nunique)):
            keys = list(nunique[i:].index)
            for col in reg_full.columns:
                if col not in keys:
                    reg_full[col] = np.where(reg_full['IS_DEFINED']==1,reg_full[col],reg_full.groupby(keys)[col].ffill())

    reg_full = reg_full.loc[~todos_rows].drop(columns=['IS_DEFINED'])

    return reg_full

def TimeSeriesManager(df: pd.DataFrame,
                      regressors: pd.DataFrame,
                      prop_params: dict,
                      io_method = pd,
                      group_columns: list = [],
                      gran: str = '',
                      target_col: str = 'y',
                      date_col: str = 'ds',
                      write: bool = False,
                      folder: list = ['',''],
                      backend: str = 'loky',
                      n_jobs: int = 1,
                      baseline_version: str = None,
                      output_version: str = None) -> pd.DataFrame:
    """
    Funcao que ajusta um modelo de prophet para série histórica e realiza previsões com esses modelos ajustados.

    Args:
        df (pd.DataFrame): DataFrame com dados desejados.
                           ATENÇÃO! Ele precisa já conter as datas futuras com regressores com seus valores preenchidos.
        regressors (pd.DataFrame): DataFrame com informação dos regressores,
                                   correspondente ao output da função build_regressors.
        prop_params (dict): Dicionário com parametros do prophet
        io_method: Qual lib voce deseja para fazer operação de io. Default é pd (pandas).
                   A lib precisa ter os métodos read_parquet e to_parquet implementados.
        group_columns (list): Lista de colunas que contem granularidade da tabela
        gran (str): String que define granularidade
        target_col (str): Coluna que contem dados do target a ser projetado pelo prophet
        date_col (str): Nome da coluna que representa data
        write (bool): Se deseja persistir as previsões em uma pasta
        folder (list): Lista com dois elementos. Primeiro é a pasta base do arquivo a ser persistido e o
                       segundo é o sufixo a ser adicionado ao nome do arquivo.
        backend (str): Backend de paralelismo. Para rodagem normal, sugerimos uso do loky. Para debug em notebooks, usar threading.
        n_jobs (int): Numero de jobs a serem usados na distribuição das séries
        baseline_version (str): String que identifica a versão do modelo a ser usada como baseline (que o processo realizará a leitura)
        output_version (str): String que idenfica a versão do modelo a ser gerada como output

    Returns:
        pd.DataFrame: DataFrame com resultados das projeções
    """
    baseline_version = '' if baseline_version is None else f'_{baseline_version}'
    output_version = '' if output_version is None else f'_{output_version}'

    if 'cache_feat' in regressors.columns:
        regressors['cache_feat'] = regressors['cache_feat'].fillna(False)
    else:
        regressors['cache_feat'] = False

    if regressors['cache_feat'].any():
        try:
            old_predictions = io_method.read_parquet(f'{folder[0]}/{target_col}_{folder[1]}{baseline_version}.parquet')
            logger.info(f'Loaded {target_col} projections from cache')
        except:
            old_predictions = pd.DataFrame()
            regressors['cache_feat'] = False
            logger.info(f'Did not find any {target_col} projections from cache')

    else:
        old_predictions = pd.DataFrame()
        regressors['cache_feat'] = False
        logger.info(f'Did not find any {target_col} projections from cache')

    if (regressors['cache_feat']==False).any():

        df_propeth = df[[date_col]+group_columns].copy().rename(columns={date_col: "ds"})
        df_propeth["y"] = df[target_col].copy()

        # Filtrando para so pegar linhas correspondentes ao target
        regressors = regressors.loc[regressors['target_col'] == target_col]

        if 'gran' in regressors.columns:
            regressors = regressors.loc[regressors['gran'].isin([gran])]

        # Definindo group_columns do regressor
        reg_group_cols = list(regressors.columns[regressors.columns.isin(group_columns)])

        # Preenche valores de unidades vazias
        if len(reg_group_cols)>0:
            regressors = fill_nan_units(df=df_propeth,
                                        regressors=regressors,
                                        reg_group_cols=reg_group_cols)

        #Explodindo colunas de regressor e sazonalidade
        regressors = explode_regressors_seasonalities(regressors)

        if 'regressor_is_calc' in regressors.columns:

            is_calc_cols = regressors.loc[regressors['regressor_is_calc'],'regressor_calc_cols'].values
            calc_cols = []

            for col in is_calc_cols:
                calc_cols += col

            calc_cols = list(set(calc_cols))
            calc_regressors = list(regressors.loc[regressors['regressor_is_calc'], 'regressor_name'].unique())
        
        # Seleciona os regressores
        if 'regressor_name' in regressors.columns:
            for col in list(regressors['regressor_name'].dropna().unique()) + calc_cols:

                if col == 'y':
                    continue

                if col in calc_regressors:
                    df_propeth[col] = 1
                    continue

            if col in df.columns:
                df_propeth[col] = df[col].copy()
            else:
                logger.info(f'Did not found regressor {col} in dataframe')

        del df

        # Removendo colunas full vazias
        regressors = regressors.dropna(how='all', axis=1)

        # Preciso fazer um dump nas funcoes lambda das variaveis calculadas para poder deserializa-las depois
        if 'regressor_calc_func' in regressors.columns:
            regressors['regressor_calc_func'] = regressors['regressor_calc_func'].apply(lambda x: x if x is None else dill.dumps(x))

        # Removendo series com cache_feat == False
        regressors = regressors.loc[regressors['cache_feat']==False]

        # Filtra as series a serem modeladas e linka com regressores desejados
        if len(reg_group_cols) > 0:
            reg_groups = regressors[reg_group_cols].drop_duplicates()
            df_propeth = df_propeth.merge(reg_groups, on=reg_group_cols)

        df_propeth = df_propeth.sort_values(["ds"]+group_columns).dropna(subset=['ds'])
        df_propeth['INITIAL_DATE'] = pd.to_datetime(df_propeth.dropna(subset=['y'])['ds'].max())

        # Se a previsao possui mais de uma coluna que defina granularidade
        if len(group_columns) > 0:

            df_propeth['IS_EMPTY'] = df_propeth.groupby(group_columns)['y'].transform(lambda x: x.isna().sum()==len(x))
            df_propeth = df_propeth.loc[~df_propeth['IS_EMPTY']].drop(columns=['IS_EMPTY'])

            if 'data_gran' not in regressors.columns:
                data_gran = reg_group_cols
            else:
                data_gran = regressors.drop_duplicates(['target_col'])['data_gran'].values[0]

            gran_intersec = [col for col in group_columns if col in data_gran]
            gran_not_int = [col for col in  group_columns if col not in data_gran]

            df_enter = df_propeth.drop_duplicates(subset=['ds']+gran_intersec, keep='last').drop(columns=gran_not_int, errors='ignore')
            reg_cols = [col for col in regressors.columns if col in ['target_col','regressor_name','seasonality_name']]

            if len(gran_intersec) > 0:
                gran_int_reg = [col for col in gran_intersec if col in regressors.columns]

                if len(gran_int_reg+reg_cols) > 0:
                    regressors = regressors.drop_duplicates(subset=gran_int_reg+reg_cols, keep='last')

                regressors = regressors.drop(columns=gran_not_int, errors='ignore')
                predictions = applyParallel(dfGrouped=df_enter.groupby(gran_intersec),
                                            func=TimeSeriesForecast,
                                            variable=target_col,
                                            regressors=regressors,
                                            prop_params=prop_params,
                                            n_jobs=n_jobs,
                                            backend=backend)
                predictions = predictions.rename_axis(gran_intersec+['index'])
                predictions['key'] = 1

            else:
                regressors = regressors.drop(columns=gran_not_int, errors='ignore')
                regressors['GROUP'] = 'group'
                regressors = regressors.drop_duplicates(subset=['GROUP']+reg_cols).drop(columns=['GROUP'])
                predictions = TimeSeriesForecast(df_enter, target_col, regressors, prop_params).assign(**{'key':1})

            del df_enter

            if len(gran_not_int) > 0:
                grans = df_propeth[gran_not_int].drop_duplicates(subset=gran_not_int).assign(**{'key':1})
                predictions = predictions.reset_index().merge(grans, on='key').drop(columns='key')

            else:
                predictions = predictions.reset_index().drop(columns='key')

        else:

            predictions = TimeSeriesForecast(df=df_propeth, variable=target_col,
                                        regressors=regressors, prop_params=prop_params)

            predictions = predictions.reset_index(drop=True)

        del df_propeth

        # Renomeia a coluna de data de volta pro nome original dela
        predictions = predictions.rename(columns={'DS':date_col})

        #Evita que na old_predictions tenha qualquer registro de grupo previsto na nova onda
        if len(predictions) > 0:

            if (len(group_columns) > 0) & (len(old_predictions) > 0):
                pred_series = predictions[group_columns].drop_duplicates().assign(**{'NEW':1})
                old_predictions = old_predictions.merge(pred_series, on=group_columns, how='outer')
                old_predictions = old_predictions.loc[old_predictions['NEW'] != 1].drop(columns=['NEW'])
            else:
                old_predictions = pd.DataFrame()

        predictions = pd.concat([old_predictions, predictions],sort=True, ignore_index=True)
        predictions = predictions.drop_duplicates(subset=[date_col]+group_columns, keep='last')

        assert len(predictions) > 0, "Seu dataframe do prophet encontra-se vazio"

        # Caso haja interesse em persistir os resultados, persiste na pasta desejada.
        if write:
            if io_method == pd:
                predictions.to_parquet(path_join(f'{folder[0]}','{target_col}',f'_{folder[1]}{output_version}.parquet'))
            else:
                io_method.to_parquet(predictions,
                                     path_join(f'{folder[0]}','{target_col}',f'_{folder[1]}{output_version}.parquet'))

    else:
        predictions = old_predictions.copy()

    return predictions

def build_regressors(reg_dict):

    """
    Função que constroi o dataframe de regressores necessários para dar como input a TimeSeriesManager.
    Args:
        reg_dict (List[dict]): Lista de dicionários, aonde cada dicionário corresponde a um conjunto de séries históricas que será modelada com aquelas configurações.
        Você pode passar como argumentos para cada dicionário:

            Argumentos obrigatórios:

                'target_col' (str): nome da coluna correspondente ao target.
                'gran' (list): Lista com strings que definem granularidade, para permitir que granularidades diferentes
                               possuam configurações diferentes. Essa granularidade saira no nome final do output.
                'data_gran' (list): Lista com colunas que definem a granularidade máxima do dado.

            Argumentos alternativos:
                'unit': (Dict(List)): Dicionario com chave sendo a coluna que se quer filtrar e valor sendo a lista de valores da coluna que se quer filtrar.
                                      É quem define para quais series a configuração sera aplicada.
                                      Se vazio, todas as series estarao enquadradas nessa configuração.
                                      Atenção! Modelo gerará erro se voce gerar duas configurações para a mesma série.

                'prophet_mode' (str): Modo da equação base do modelo.
                                      Pode ser 'multiplicative' (default)
                                      (termos externos serão adicionados como multiplicadores ao trend por default)
                                      ou 'additive' (termos externos serão aditivos ao trend por default)

                'regressors' (List[dict]): Regressores a serem utilizados na previsão.
                                           É uma lista de dicionarios, na qual cada regressor é um dicionario.
                                           Argumentos obrigatórios caso passado:
                                               'name': Nome do regressor
                                           Argumentos alternativos:
                                           Pode se passar qualquer parametro que a funcao add_regressor do prophet permita.
                                           São eles:
                                                'mode' (str): additive/multiplicative.
                                                        Se o regressor sera adicionado como termo aditivo/multiplicativo.
                                                        Se não passado, respeitará o que tiver em prophet_mode.
                                                'prior_scale' (int): Escala a priori que indica pro otimizador
                                                                     um conhecimento inicial do peso do coeficiente.
                                                                     Se não passado, valor padrão é 10.
                                                'standardize' (bool): Se voce quer normalizar ou não a variável.
                                                                      Se a variável for binária, o modelo assume como padrao False.
                                                                      Se for não-binária, padrão True.
                                                'is_calc' (bool): Se o regressor é um calculo/consequencia de outras variaveis ou não.
                                                'calc_cols' (list): Lista de colunas que compõem o cálculo da variável.
                                                'calc_func' (function): Função a ser aplicada ao dataframe para calcular o regressor consequencia.

                'manual_changepoints' (List): Lista com datas no formato '%Y-%m-%d' correspondente a changepoints manuais desejados no fit.
                                              Ele entrara no modelo no algoritmo de definição de changepoints criado pela área.

                'big_changers' (List): Lista com datas no formato '%Y-%m-%d' correspondente a momentos de ruptura de patamar da série histórica.
                                       Ele adicionará ao modelo uma função degrau, na qual settará 0 pro antes dessa data e 1 pra data e o pós dela.

                'special_events' (List): Lista com eventos especiais que se repetem. Ela deve ser passada no seguinte formato:
                                        [{'name': 'nome-do-evento', 'dates': [{'date': 'YYYY-MM-DD', 'value': valor}]}]
                                        Campo 'date' em 'dates' (Union[str, Tuple[str, str], List[Union[str, Tuple[str, str]]]]):
                                        * Caso seja passada uma str, uma data pontual será considerada
                                        * Caso seja passada uma tupla com duas str, será considerada como um intervalo fechado de datas
                                        * Caso seja passada uma lista, então ela pode conter os dois casos acima
                                        Caso voce queira customizar o regressor, voce pode passar os parametros padrao de adicao de regressor do prophet no dicionario (mode, prior_scale, standardize...)

                'seasonalities' (List[dict]): Sazonalidades a serem utilizados na previsão.
                                           É uma lista de dicionarios, na qual cada regressor é uma sazonalidade.
                                           Argumentos obrigatórios caso passado:
                                               'name': Nome da sazonalidade
                                           Argumentos alternativos:
                                           Pode se passar qualquer parametro que a funcao add_regressor do prophet permita.
                                           São eles:
                                                'mode' (str): additive/multiplicative.
                                                        Se a sazonalidade sera adicionado como termo aditivo/multiplicativo.
                                                        Se não passado, respeitará o que tiver em prophet_mode.
                                                'prior_scale' (int): Escala a priori que indica pro otimizador
                                                                     um conhecimento inicial do peso do coeficiente da sazonalidade.
                                                                     Se não passado, valor padrão é 10.
                                                'fourier_order' (int): Numero de frequencias harmonicas a serem criadas como features para o modelo capturar sazonalidade.
                                                                       Se não passado, valor padrão depende da frequencia temporal do dado.
                                                                       Para 'Y': fourier_order = 1
                                                                       Para 'M': fourier_order = 3
                                                                       Para 'D': fourier_order = 10

                                'trend' (str): Qual tipo de sinal de tendencia sera modelado. Pode ser 'linear' (default), 'flat' ou 'logistic'.
                                               Caso voce escolha logistic, voce precisa passar nos regressors os parametros de cap e floor.
                                               Voce pode passa-los em forma de dicionário com chave = data e valor = valor, caso queira que varie no tempo
                                               ou passar um valor inteiro, caso queira que seja constante na série.
                                               Ver documentação da API do Prophet para mais detalhes.
                                'cache_feat' (bool): Se o usuário deseja carregar o que ja tem em cache da previsão da série correspondente.
                                                     Padrão é False.
                                'freq': (str): Frequencia temporal do dado. Deve-se usar uma frequency alias padrão do Pandas.
                                               No momento somente há implementação para 'Y'(anual), 'M' (mensal) ou 'D' (diário)
    Returns:
        regressors (pd.DataFrame): DataFrame de regressores que sera dado como input as previsões
    """

    #Transforma a lista de dicionarios em dataframe e explode a lista passada em granularidades em linhas diferentes.
    regressors = pd.json_normalize(reg_dict)
    regressors.columns = regressors.columns.str.replace('unit.','')
    regressors = regressors.explode(column='gran', ignore_index=True)

    return regressors

def explode_regressors_seasonalities(regressors):

    """
    Função que realiza a explosão das celulas com listas dos dataframes de regressores em linhas diferentes.
    Args:
        regressors (pd.DataFrame): DataFrame com regressores armazenados
    Returns:
        pd.DataFrame: DataFrame explodido em linhas por regressor de cada modelo
    """

    #Explode a lista de regressores em linhas diferentes
    if 'regressors' in regressors.columns:
        regressors = regressors.explode(column='regressors', ignore_index=True)
        regressors['regressor_mode'] = regressors['regressors'].str.get('mode')
        regressors['regressor_prior_scale'] = regressors['regressors'].str.get('prior_scale')
        regressors['regressor_prior_mean'] = regressors['regressors'].str.get('prior_mean')
        regressors['regressor_standardize'] = regressors['regressors'].str.get('standardize')
        regressors['regressor_constraint'] = regressors['regressors'].str.get('constraint')
        regressors['regressor_name'] = regressors['regressors'].str.get('name')
        regressors['regressor_is_calc'] = regressors['regressors'].str.get('is_calc').fillna(False)
        regressors['regressor_calc_cols'] = regressors['regressors'].str.get('calc_cols')
        regressors['regressor_calc_func'] = regressors['regressors'].str.get('calc_func')

    #Explode a lista de sazonalidades em linhas diferentes
    if 'seasonalities' in regressors.columns:
        regressors = regressors.explode(column='seasonalities', ignore_index=True)
        regressors['seasonality_mode'] = regressors['seasonalities'].str.get('mode')
        regressors['seasonality_prior_scale'] = regressors['seasonalities'].str.get('prior_scale')
        regressors['seasonality_standardize'] = regressors['seasonalities'].str.get('standardize')
        regressors['seasonality_name'] = regressors['seasonalities'].str.get('name')
        regressors['seasonality_period'] = regressors['seasonalities'].str.get('period')
        regressors['seasonality_fourier_order'] = regressors['seasonalities'].str.get('fourier_order')
        regressors['seasonality_tipo'] = regressors['seasonalities'].str.get('tipo')
        regressors['seasonality_num_cycles'] = regressors['seasonalities'].str.get('num_cycles')
        regressors['seasonality_create_binaries'] = regressors['seasonalities'].str.get('create_binaries')
        regressors['seasonality_remove_unvariant_cycles'] = regressors['seasonalities'].str.get('remove_unvariant_cycles')

    regressors = regressors.drop(columns=['regressors','seasonalities'], errors='ignore')

    return regressors