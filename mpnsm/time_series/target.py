from typing import Dict, List, Union, Tuple

from dags.models.time_series import DEFAULT_METRICS, DEFAULT_OUTLIERS_CONFIG


class Target:

    """Class abstraction for a target unit in forecast"""
    def __init__(self,
                 target_col: str,
                 model: Dict[str, Union[str, Dict]],
                 metrics: List[Dict[str, str]] = DEFAULT_METRICS,
                 regressors: List[Dict[str, Union[str, List[str], int]]] = [],
                 outliers_config: Dict[str, Union[float, int]] = DEFAULT_OUTLIERS_CONFIG,
                 drop_intervals: List[Tuple] = []):

        """
        Class abstraction for a target unit in forecast.
        Args:
            target_col (str): Name of column that represents the target
            model (Dict[str, Union[str, Dict]]): Model configuration Dict. 
                                                 Mandatory arguments:
                                                    - type (str): Model Class name.
                                                 Optional arguments:
                                                    - **model_params: Arguments of the model class. 
                                                                    See the corresponding model class doc for more details
                                                    hyperparams_kwargs (Dict): 
                                                         Mandatory arguments:
                                                            - type (str): Hyperparameter tuning Class.
                                                            - **hyperparmeter_tuning_kwargs: 
                                                                See Hyperparameter tuning class doc for more details
            metrics (List[Dict[str, str]]): List of metrics to be usen in fitting. 
                                            Each metric can be specified as a Dict in the following way:
                                                Mandatory arguments:
                                                    - type: Metric Class name
                                                    **metric_params: Arguments of the Metric Class.
                                                        See the corresponding Metric class doc for more details.
            regressors (List[Dict[str, Union[str, List[str], int]]]): List of regressors to be usen in fitting. 
                                            Each regressor can be specified as a Dict in the following way:
                                                Mandatory arguments:
                                                    - name: Regressor name
                                                    - type: Regressor Class name
                                                    **regressor_params: Arguments of the Regressor Class.
                                                        See the corresponding Regressor class doc for more details.
            outliers_config (Dict[str,str]): Outlier configuration dict. Outlier can be specified as a Dict in the following way:
                                             Mandatory arguments:
                                                - type: Outlier Class name
                                                **outliers_params: Arguments of the Outlier Class.
                                                    See the corresponding Outlier class doc for more details.
            drop_intervals (List[Tuple]): List of intervals undesired in training set. Must be passed as [('YYYY-MM-DD','YYYY-MM-DD')]
        """

        self.target_col = target_col
        self.model = model
        self.metrics = metrics
        self.regressors = regressors
        self.outliers_config = outliers_config      
        self.drop_intervals = drop_intervals
