import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

    
class FeatureNormalizer(BaseEstimator, TransformerMixin):
    games_started = 4
    games_played = 3
    team_games_played = 40
    mpg = 6
    per_game_stats = [ 9, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    max_stats = [30, 31, 32, 33, 34]
    average_stats = [12, 13, 14, 15, 16, 19]
    
    def __init__(self, keep_old=False):
        self.keep_old = keep_old
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        games_started_norm = X[:, self.games_started]/X[:, self.games_played]
        games_played_norm = X[:, self.games_played]/X[:, self.team_games_played]
        all_per_game_norm = X[:, self.per_game_stats]/X[:, self.mpg][:, None]
        max_per_average = X[:, self.max_stats[:-1]]/(X[:, self.average_stats[:-2]] + 0.1)
        max_per_average_rebs = X[:, self.max_stats[-1]]/(X[:, self.average_stats[-2]] + X[:, self.average_stats[-1]] + 0.1)
        if not self.keep_old:
            X[:, self.games_started] = games_started_norm
            X[:, self.games_played] = games_played_norm
            X[:, self.per_game_stats] = all_per_game_norm
            X[:, self.max_stats[:-1]] = max_per_average
            X[:, self.max_stats[-1]] = max_per_average_rebs
            return X
        return np.c_[X, games_started_norm, games_played_norm,
                     all_per_game_norm, max_per_average, max_per_average_rebs]
    
    
class FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, to_drop):
        self.to_drop = to_drop
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = np.array(X)
        X = np.delete(X, self.to_drop, axis=1)
        return X
    
    
class CustomImputer(BaseEstimator, TransformerMixin):
    fill_with_median = [1, 2]
    fill_with_zeros = [ 0,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                       19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                       36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    ap_poll_rank = 52
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        X = np.array(X, dtype=object)
        self.zero_imputer = SimpleImputer(strategy='constant', fill_value=0)
        self.median_imputer = SimpleImputer(strategy='median')
        self.median_imputer.fit(X[:, self.fill_with_median])
        return self
    
    def transform(self, X, y=None):
        X = np.array(X)
        X[:, self.fill_with_zeros] = self.zero_imputer.fit_transform(X[:, self.fill_with_zeros])
        X[:, self.fill_with_median] = self.median_imputer.transform(X[:, self.fill_with_median])
        X[:, self.ap_poll_rank] = [int(not np.isnan(rank)) for rank in X[:, self.ap_poll_rank]] 
        return X