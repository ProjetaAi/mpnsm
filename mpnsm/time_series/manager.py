import pickle
import json
import pandas as pd

from io import BytesIO
from joblib import Parallel, delayed

from dags import config
from ipp_ds.io import io
from ipp_ds.data_cleaning.path import path_join

from dags.models.time_series.forecast import TimeSeriesForecast
from dags.models.time_series.config import (generate_full_tree,
                                            generate_df_input, generate_dicts)


class TimeSeriesManager():
    """Time Series Manager Class"""

    def __init__(self, df_feat: pd.DataFrame, group_columns: list,
                 config_dict: list, n_jobs: int = -1):
        """
        Time Series Manager Class
        Args:
            df_feat (pd.DataFrame): DataFrame with targets and features data. Features data must already have future data.
            group_columns (List): List of columns that defines granularities
            config_dict (Dict): Dictionary with model configurations
            n_jobs (int): Number of workers desired in parallelism
        """

        self.df_feat = df_feat
        self.group_columns = group_columns
        self.config_dict = config_dict
        self.n_jobs = n_jobs

        self.dict_final = self.create_unit_variables(
            df_feat=self.df_feat,
            config_dict=self.config_dict,
            group_columns=self.group_columns)

    def write_params(self, forecaster: TimeSeriesForecast):
        """
        Method that write parameters already optimized.
        Args:
            forecaster (TimeSeriesForecast): Forecaster object desired
        """

        params = forecaster.best_params

        targets = [target.target_col for target in forecaster.targets]

        units_path = '/'.join(forecaster.units) + '/' if len(
            forecaster.units) else ''

        for target in targets:

            value = params[target]

            try:  #buscando arquivo pré existente no lake
                files = io.glob(
                    path_join(config.RAW_FOLDER,
                              f'best_params/{units_path}{target}_v*.json'))
                current_version = max(
                    [int(file.split(f'{target}_v')[1][:-5]) for file in files])

                old_dict = io.read_any(
                    path_join(
                        config.RAW_FOLDER,
                        f'best_params/{units_path}{target}_v{str(current_version)}.json'
                    ), json.load)

                if value == old_dict:
                    print('Não houve modificações no ', target)
                else:
                    value_encoded = json.dumps(value).encode('utf-8')
                    byte_stream = io.BytesIO(value_encoded)
                    print('Gerando arquivo ', target, ' version: ',
                          str(current_version + 1))
                    io.to_any(
                        byte_stream,
                        path_join(
                            config.RAW_FOLDER,
                            f'best_params/{units_path}{target}_v{str(current_version + 1)}.json'
                        ),
                        upload_mode='full')

            except:
                print('Gerando arquivo v1 para ', target)

                value_encoded = json.dumps(value).encode('utf-8')
                byte_stream = io.BytesIO(value_encoded)
                io.to_any(byte_stream,
                          path_join(
                              config.RAW_FOLDER,
                              f'best_params/{units_path}{target}_v1.json'),
                          upload_mode='full')

    def load_params(self, forecaster: TimeSeriesForecast):
        """
        Method that load parameters already optimized.
        Args:
            forecaster (TimeSeriesForecast): Forecaster object desired
        Returns:
            best_params (dict): Dict with best_params
        """

        targets = [target.target_col for target in forecaster.targets]

        units_path = '/'.join(forecaster.units) + '/' if len(
            forecaster.units) else ''

        best_params = {}
        for target in targets:
            try:
                files = io.glob(
                    path_join(config.RAW_FOLDER,
                              f'best_params/{units_path}{target}_v*.json'))
                current_version = max(
                    [int(file.split(f'{target}_v')[1][:-5]) for file in files])
                print('Lendo de ', target, 'versão: ', current_version)

                base_folder = path_join(
                    config.RAW_FOLDER,
                    f'best_params/{units_path}{target}_v{str(current_version)}.json'
                )
                output = io.read_any(base_folder, json.load)

                best_params[target] = output
            except:
                print('Não foram encontrados hiperparâmetros para o target ',
                      target, 'revertendo para padrão')

        return best_params

    @staticmethod
    def create_unit_variables(df_feat, config_dict, group_columns):
        """Method create units dicts that are not explictly passed"""

        with_config = generate_full_tree(df=df_feat,
                                         config_dict=config_dict,
                                         group_columns=group_columns)

        df_final = generate_df_input(df_feat=df_feat,
                                     group_columns=group_columns,
                                     config_df=with_config)

        dict_final = generate_dicts(df_final, group_columns, with_config)

        return dict_final

    def run_forecasts(self, unit, dict_config):
        """
        Method that run forecasts for one time series.
        Args:
            unit (List): List of values that defines a time series unit
            dict_config (Dict): Dictionary with data and configuration
        """

        data, model_config = dict_config['data'], dict_config['config']
        model_config = model_config[0]

        gran = model_config['data_gran']

        if len(gran):
            gran.sort()
            self.gran = gran
            units = ['unit.' + data_gran for data_gran in self.gran]
            units = [model_config[unit] for unit in units]
        else:
            gran = []
            self.gran = gran
            units = []

        forecaster = TimeSeriesForecast(
            data=data,
            forecast_unit=model_config['forecast_unit'],
            freq=model_config['date.freq'],
            date_col=model_config['date.date_col'],
            predict_mode=model_config['predict_mode'],
            targets=model_config['targets'],
            cv=model_config['cv'][0],
            units=units)

        forecaster.best_params = self.load_params(forecaster)

        units_path = '/'.join(units) + '/' if len(units) else ''

        if model_config['run_mode'] == 'fit_predict':
            fcst, model, future = forecaster.fit_predict()

            unit = list(unit)

            for target_dict in model_config['targets']:
                target = target_dict['target_col']

                renaming_cols = {
                    col: f'{col}_real'
                    for col in future[target].columns
                    if col not in ['ds', 'y_real', 'is_train']
                }
                new_future = future[target].rename(columns=renaming_cols)
                new_fcst = model[target].regressor_coefficients(fcst[target])
                grans = pd.DataFrame(data=[unit], columns=self.gran)
                new_fcst = pd.concat([new_fcst, grans], axis=1).ffill()
                new_fcst.loc[:, self.gran] = new_fcst.loc[:, self.gran].ffill()
                combined = new_future.merge(new_fcst, on=['ds'], how='outer')

                io.to_parquet(
                    combined,
                    path_join(config.RAW_FOLDER,
                              f'run_results/{units_path}{target}.parquet'))

                model_json = model[target].save_model()

                model_byte_stream = BytesIO(pickle.dumps(model_json))

                io.to_any(model_byte_stream,
                          path_join(config.RAW_FOLDER,
                                    f'run_models/{units_path}{target}.json'),
                          upload_mode='full')

        if model_config['run_mode'] == 'fit':
            full_df, train_df, model = forecaster.initialize(
                full_df=None,
                last_index=None,
                hyperparams=forecaster.best_params)
            model, train_df = forecaster.fit(model=model, train_df=train_df)

            for target_dict in model_config['targets']:
                target = target_dict['target_col']

                model_json = model[target].save_model()

                model_byte_stream = BytesIO(pickle.dumps(model_json))

                io.to_any(model_byte_stream,
                          path_join(
                              config.RAW_FOLDER,
                              f'fit_data/model/{units_path}{target}.json'),
                          upload_mode='full')

                io.to_parquet(
                    full_df[target],
                    path_join(
                        config.RAW_FOLDER,
                        f'fit_data/full_df/{units_path}{target}.parquet'))

        if model_config['run_mode'] == 'predict':

            models = {}
            full_dfs = {}

            for target_dict in model_config['targets']:
                target = target_dict['target_col']
                model = target_dict['model']

                prophet_model = forecaster.initialize_model(
                    model=model,
                    target_col=target,
                    hyperparams=forecaster.best_params.pop(target, {}))

                new_model = io.read_any(
                    path_join(config.RAW_FOLDER,
                              f'fit_data/model/{units_path}{target}.json'),
                    pickle.load)
                models[target] = prophet_model.load_model(new_model)

                full_dfs[target] = io.read_parquet(
                    path_join(
                        config.RAW_FOLDER,
                        f'fit_data/full_df/{units_path}{target}.parquet'))

            fcst, model, future = forecaster.predict(full_df=full_dfs,
                                                     model=models)

            unit = list(unit)

            for target_dict in model_config['targets']:
                target = target_dict['target_col']

                renaming_cols = {
                    col: f'{col}_real'
                    for col in future[target].columns
                    if col not in ['ds', 'y_real', 'is_train']
                }
                new_future = future[target].rename(columns=renaming_cols)
                new_fcst = model[target].regressor_coefficients(fcst[target])
                grans = pd.DataFrame(data=[unit], columns=self.gran)
                new_fcst = pd.concat([new_fcst, grans], axis=1).ffill()
                new_fcst.loc[:, self.gran] = new_fcst.loc[:, self.gran].ffill()
                combined = new_future.merge(new_fcst, on=['ds'], how='outer')

                print(f'saving combined in {units_path}{target}')
                io.to_parquet(
                    combined,
                    path_join(config.RAW_FOLDER,
                              f'run_results/{units_path}{target}.parquet'))

                model_json = model[target].save_model()

                model_byte_stream = BytesIO(pickle.dumps(model_json))

                io.to_any(model_byte_stream,
                          path_join(config.RAW_FOLDER,
                                    f'run_models/{units_path}{target}.json'),
                          upload_mode='full')

    def run_all(self, verbose=1):
        """Method that run all forecasts in parallel"""

        Parallel(n_jobs=self.n_jobs,
                 verbose=verbose)(delayed(self.run_forecasts)(unit, config)
                                  for unit, config in self.dict_final.items())
