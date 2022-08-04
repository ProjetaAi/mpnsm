import pandas as pd
import ast
from typing import List, Dict, Union, Tuple


def generate_full_tree(df: pd.DataFrame, config_dict: List[Dict],
                       group_columns: List[str]) -> pd.DataFrame:

    """
    Functions that fills the configurator with non-defined units.
    Args:
        df (pd.DataFrame): DataFrame with time series
        config_dict (List[Dict]): List of forecast configurators
        group_columns (List[str]): List of columns that define a time series unit. 
    """

    config_df = pd.json_normalize(config_dict)
    for col in [col for col in config_df.columns if col.startswith('unit.')]:
        config_df = config_df.explode(col)

    groups: pd.DataFrame = df[group_columns].drop_duplicates()
    groups.columns = 'unit.' + groups.columns

    with_config = pd.DataFrame()
    group_config = groups.copy()
    config_table = config_df.copy()

    for ind in range(len(group_columns) + 1):

        if ind == 0:
            group_unit = group_columns.copy()
        else:
            group_unit = group_columns[:-(ind)]

        group_cols = ['unit.' + unit for unit in group_columns]
        group_unit = ['unit.' + unit for unit in group_unit]
        not_in_unit = [unit for unit in group_cols if unit not in group_unit]

        config_gran = config_table.dropna(subset=group_unit)

        if len(not_in_unit) > 0:
            config_gran = config_gran.drop(columns=not_in_unit)

        if len(group_unit) == 0:
            group_config['key'] = 1
            config_gran['key'] = 1
            config_table['key'] = 1
            group_unit = ['key']

        config_gran = group_config.merge(config_gran,
                                         on=group_unit).set_index(group_unit)

        group_config = group_config.set_index(group_unit)

        group_config = group_config.loc[~group_config.index.isin(
            config_gran.index.to_list())].reset_index()

        config_table = config_table.set_index(group_unit)
        config_table = config_table.loc[~config_table.index.isin(
            config_gran.index.to_list())].reset_index()

        config_gran = config_gran.reset_index()

        with_config = pd.concat([with_config, config_gran],
                                sort=True,
                                ignore_index=True)

        if group_unit == ['key']:
            with_config = with_config.drop(columns=['key'])

    assert len(with_config) == len(
        groups
    ), f'Missing configuration for units: {group_config.set_index(group_columns).index.to_list()}'

    return with_config


def get_columns_needed(config: Union[pd.DataFrame, Dict], layer: str) -> List:

    """
    Functions that get the columns needed to do an explode in dataframe
    Args:
        config_dict (List[Dict]): Model Configurator
        layer (str): Can be 'forecast' or 'target'
    """

    if layer == 'forecast':
        prefix = 'targets.'
        a: pd.DataFrame = pd.json_normalize(
            config.explode('targets').to_dict(orient='records'))

    if layer == 'target':
        prefix = ''
        a: pd.DataFrame = pd.json_normalize(config)

    a = a.explode(f'{prefix}regressors')
    a = pd.json_normalize(a.to_dict(orient='records'))

    if f'{prefix}regressors.calc_cols' not in a.columns:
        a[f'{prefix}regressors.calc_cols'] = '[]'
    else:
        a[f'{prefix}regressors.calc_cols'] = a[
            f'{prefix}regressors.calc_cols'].astype('str').replace(
                'nan', '[]')

    a[f'{prefix}regressors.calc_cols'] = a[
        f'{prefix}regressors.calc_cols'].apply(ast.literal_eval)
    a = a.explode(f'{prefix}regressors.calc_cols')

    cols = list(set(a[f'{prefix}target_col'].unique().tolist()+ \
                    a[f'{prefix}regressors.name'].unique().tolist()+ \
                    a[f'{prefix}regressors.calc_cols'].dropna().unique().tolist()))

    if 'date.date_col' in a.columns:
        cols = a['date.date_col'].unique().tolist() + cols

    return cols


def generate_df_input(df_feat: pd.DataFrame, group_columns: List[str],
                      config_df: pd.DataFrame) -> pd.DataFrame:

    """
    Functions that generate the input dataframe
    Args:
        df (pd.DataFrame): DataFrame with time series
        group_columns (List[str]): List of columns that define a time series unit.
        config_dict (List[Dict]): List of forecast configurators
    """

    cols = get_columns_needed(config_df, layer='forecast')

    df_final = df_feat.loc[:, df_feat.columns.isin(cols + group_columns)]
    df_final = df_final.rename(
        columns={col: 'unit.' + col
                 for col in group_columns})
    group_cols = ['unit.' + col for col in group_columns]
    df_final = df_final.set_index(['ds'] + group_cols).reset_index()

    return df_final


def generate_dicts(df_final: pd.DataFrame, group_columns: List[str],
                   with_config: pd.DataFrame):

    """
    Functions that generate the inputs to TimeSeriesForecast
    Args:
        df_final (pd.DataFrame): DataFrame with time series
        group_columns (List[str]): List of columns that define a time series unit.
        with_config (pd.DataFrame): Dataframe configurator
    """

    group_cols = ['unit.' + col for col in group_columns]

    dict_dist: Dict[Tuple, Dict[str, Union[pd.DataFrame, Dict]]] = {}

    for index, group in df_final.groupby(group_cols):
        group: pd.DataFrame
        dict_dist[index] = {'data': group.reset_index(drop=True)}
    for index, config in with_config.groupby(group_cols):
        config: pd.DataFrame
        dict_dist[index].update({'config': config.to_dict(orient='records')})

    return dict_dist