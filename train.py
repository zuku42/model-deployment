import os
import numpy as np
import pandas as pd
import pickle

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

from custom_transformers import FeatureNormalizer, FeatureDropper, CustomImputer


def load_players_data(file_path, file_name, index_col=0):
    csv_path = os.path.join(file_path, file_name)
    return pd.read_csv(csv_path, index_col=index_col)

READ_PATH = 'data/'
DATA_FILE = 'ncaa_players.csv'

dataset = load_players_data(file_path=READ_PATH, file_name=DATA_FILE)

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
y = y.apply(lambda x: int(x < 4))

RS = 101

numeric_features = X.select_dtypes(include=np.number).columns
cat_features = X.select_dtypes(include=object).columns
to_drop_num_names = ['season']
to_drop_cat_names = ['name', 'school']
to_drop_num = np.where(np.isin(numeric_features, to_drop_num_names))[0]
to_drop_cat = np.where(np.isin(cat_features, to_drop_cat_names))[0]


num_pipeline = Pipeline([('imputer', CustomImputer()),
                         ('normalizer', FeatureNormalizer(keep_old=True)),
                         ('dropper', FeatureDropper(to_drop=to_drop_num)),
                         ('scaler', StandardScaler()),
                        ])

cat_pipeline = Pipeline([('dropper', FeatureDropper(to_drop=to_drop_cat)),
                         ('encoder', OneHotEncoder()),
                        ])

feature_transformer = ColumnTransformer([('numeric_pipeline', num_pipeline, numeric_features),
                                         ('categorical_pipeline', cat_pipeline, cat_features),
                                        ])

X = feature_transformer.fit_transform(X)
X, y = SMOTE(random_state=RS).fit_resample(X, y)

gbc = GradientBoostingClassifier(max_depth=5, min_samples_leaf=2, min_samples_split=2, random_state=RS)
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10, min_samples_leaf=2, min_samples_split=2),
                                  n_estimators=75, random_state=RS)
net = MLPClassifier(hidden_layer_sizes=(100,), learning_rate='constant', learning_rate_init=0.001,
                             activation='logistic', max_iter=10000, random_state=RS)

final_model = VotingClassifier([('gbc', gbc),
                                ('ada', ada),
                                ('net', net),
                               ], voting='soft')


final_model.fit(X, y)
full_pipeline = Pipeline([('feature transformations', feature_transformer),
                          ('final model', final_model),
                         ])


filepath = 'binary_model.sav'
pickle.dump(full_pipeline, open(filepath, 'wb'))