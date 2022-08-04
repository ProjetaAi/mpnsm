from abc import abstractmethod

import pandas as pd

class CrossValidation:

    """Abstract class for Cross Validation."""

    @abstractmethod
    def split(self, X: pd.DataFrame):
        """Abstract method for split folds"""
        pass


class TimeSeriesCV(CrossValidation):

    """Class for make Time Series Cross Validation"""

    def __init__(self,
                 n_splits: int,
                 horizon: int,
                 jump: int = 1,
                 min_train_size: int = None,
                 min_test_size: int = None):
        
        """
        Class for make Time Series Cross Validation
        Args:
            n_splits (int): Number of splits desired in data
            horizon (int): Horizon desired in each fold
            jump (int): Size of jump step in each fold. Defaults to 1.
            min_train_size (int): Minimum size of train data. Defaults to None (equals to horizon)
            min_test_size (int): Minimum size of test data. Defaults to None (equals to horizon)

        """

        self.n_splits = n_splits
        self.horizon = horizon
        self.jump = jump
        assert self.jump > 0, 'jump can not be lower or equal than zero'
        self.min_train_size = self.horizon if min_train_size is None else min_train_size
        self.min_test_size = self.horizon if min_test_size is None else min_test_size

    def split(self, X: pd.DataFrame, date_col: str = None):

        """
        Method to split data according to CrossValidation configuration
        Args:
            X (pd.DataFrame): Data
            date_col (str): Column that represents date
        """

        if date_col is not None:
            assert X.equals(X.sort_values(
                date_col,
                ascending=True)), f'Dataframe is not properly sorted in time'

        assert len(
            X
        ) >= self.min_test_size, f'Data has less then min_test_size ({self.min_test_size}) rows'
        assert len(
            X
        ) >= self.min_train_size, f'Data has less then min_train_size ({self.min_train_size}) rows'
        assert len(
            X
        ) >= self.min_train_size + self.min_test_size, f'Data has less then min_train_size+min_test_size ({self.min_train_size+self.min_test_size}) rows'

        start = len(X) - self.min_test_size
        stop = max(start - (self.jump * self.n_splits), self.min_train_size)

        for i in range(start, stop - 1, -self.jump):
            yield X.iloc[:i].index.to_list(
            ), X.iloc[i:i + self.horizon].index.to_list()
