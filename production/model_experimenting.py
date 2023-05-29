import os
import os.path as op
import shutil
import json

# standard third party imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer,TransformedTargetRegressor
from sklearn.metrics import mean_squared_error,r2_score


# impute missing values
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from category_encoders import TargetEncoder
import warnings

warnings.filterwarnings('ignore', message="pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.", 
                        category=FutureWarning)
warnings.filterwarnings('ignore', message="pandas.Float64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.",
                        category=FutureWarning)
warnings.filterwarnings('ignore', message="pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.", 
                        category=DeprecationWarning)
warnings.filterwarnings('ignore', message="pandas.Float64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.",
                        category=DeprecationWarning)

# standard code-template imports
from ta_lib.core.api import (
    create_context, get_dataframe, get_feature_names_from_column_transformer, string_cleaning,
    get_package_path, display_as_tabs, save_pipeline, load_pipeline, initialize_environment,
    load_dataset, save_dataset, DEFAULT_ARTIFACTS_PATH
)

import ta_lib.eda.api as eda
from xgboost import XGBRegressor
from ta_lib.regression.api import SKLStatsmodelOLS
from ta_lib.regression.estimators import CustomTargetTransformer
from ta_lib.regression.api import RegressionComparison, RegressionReport
import ta_lib.reports.api as reports
from ta_lib.data_processing.api import Outlier

initialize_environment(debug=False, hide_warnings=True)
artifacts_folder = DEFAULT_ARTIFACTS_PATH
config_path = op.join('conf', 'config.yml')
context = create_context(config_path)
with open("./conf/model_config.json") as cf_file:
    xgboost_config = json.loads( cf_file.read() )
train_X = load_dataset(context, 'train/housing/features')
train_y = load_dataset(context, 'train/housing/target')



test_X = load_dataset(context, 'test/housing/features')
test_y = load_dataset(context, 'test/housing/target')


# collecting different types of columns for transformations
cat_columns = train_X.select_dtypes('object').columns
num_columns = train_X.select_dtypes('number').columns
outlier_transformer = Outlier(method='mean')

train_X = outlier_transformer.fit_transform(train_X)


tgt_enc_simple_impt = Pipeline([
    ('target_encoding', TargetEncoder(return_df=False)),
    ('simple_impute', SimpleImputer(strategy='most_frequent')),
])


# NOTE: the list of transformations here are not sequential but weighted 
# (if multiple transforms are specified for a particular column)
# for sequential transforms use a pipeline as shown above.
features_transformer = ColumnTransformer([
    
    ## categorical columns
    ('tgt_enc', TargetEncoder(return_df=False),
     list(set(cat_columns) - set(['technology', 'functional_status', 'platforms']))),
    
    # NOTE: if the same column gets repeated, then they are weighed in the final output
    # If we want a sequence of operations, then we use a pipeline but that doesen't YET support
    # get_feature_names. 
    ('tgt_enc_sim_impt', tgt_enc_simple_impt, ['technology', 'functional_status', 'platforms']),
        
    ## numeric columns
    ('med_enc', SimpleImputer(strategy='median'), num_columns),
    
])

sample_X = train_X.sample(frac=0.1, random_state=context.random_seed)
sample_y = train_y.loc[sample_X.index]

# sample_train_X = get_dataframe(
#     features_transformer.fit_transform(sample_X, sample_y), 
#     get_feature_names_from_column_transformer(features_transformer)
# )

# nothing to do for target
sample_train_y = sample_y
out = eda.get_correlation_table(train_X)
out[out["Abs Corr Coef"] > 0.6]
# saving the list of relevant columns
save_pipeline(train_X, op.abspath(op.join(artifacts_folder, 'curated_columns.joblib')))

# save the feature pipeline
save_pipeline(features_transformer, op.abspath(op.join(artifacts_folder, 'features.joblib')))
cols = list(train_X.columns)
vif = eda.calc_vif(train_X)
while max(vif.VIF) > 15:
    #removing the largest variable from VIF
    cols.remove(vif[(vif.VIF==vif.VIF.max())].variables.tolist()[0])
    vif = eda.calc_vif(train_X[cols])
reg_vars = vif.query('VIF < 15').variables
reg_vars = list(reg_vars)
# Custom Transformations like these can be utilised
def _custom_data_transform(df, cols2keep=None):
    """Transformation to drop some columns in the data
    
    Parameters
    ----------
        df - pd.DataFrame
        cols2keep - columns to keep in the dataframe
    """
    cols2keep = cols2keep or []
    if len(cols2keep):
        return (df
                .select_columns(cols2keep))
    else:
        return df
reg_ppln_ols = Pipeline([
    ('',FunctionTransformer(_custom_data_transform, kw_args={'cols2keep':reg_vars})),
    ('estimator', SKLStatsmodelOLS())
])
reg_ppln_ols.fit(train_X,np.ravel(train_y))

reg_ppln_ols['estimator'].summary()
reg_ppln = Pipeline([
    ('', FunctionTransformer(_custom_data_transform, kw_args={'cols2keep':reg_vars})),
    ('Linear Regression', SKLStatsmodelOLS())
])
print(reg_ppln_ols['estimator'].summary())
# let's find features for some decent defaults
estimator = XGBRegressor()
xgb_training_pipe_init = Pipeline([
    ('XGBoost', XGBRegressor())
])
xgb_training_pipe_init.fit(train_X, train_y)
# let's find features for some decent defaults
imp_features=['ocean_proximity_INLAND','median_income','population_per_household','ocean_proximity_NEAR BAY','ocean_proximity_ISLAND']

estimator = XGBRegressor()
xgb_training_pipe2 = Pipeline([
    ('', FunctionTransformer(_custom_data_transform, kw_args={'cols2keep':imp_features})),
    ('XGBoost', XGBRegressor())
])
parameters = xgboost_config
est = XGBRegressor()

xgb_grid = GridSearchCV(est,
                        parameters,
                        cv = 2,
                        n_jobs = 4,
                        verbose=True)

xgb_grid.fit(train_X, train_y)

print(f"xgb best score {xgb_grid.best_score_}")
print(f"xgb best parameters {xgb_grid.best_params_}")
warnings.filterwarnings('ignore', message="pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.", 
                        category=FutureWarning)
warnings.filterwarnings('ignore', message="pandas.Float64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.",
                        category=FutureWarning)

model_xgb=TransformedTargetRegressor(regressor=xgb_grid,transformer=CustomTargetTransformer())
model_xgb.fit(train_X,train_y)
model_xgb.get_params
pred=model_xgb.predict(test_X)
rmse=np.sqrt(mean_squared_error(test_y,pred))
r2=r2_score(test_y,pred)

print(f"rmse on test data {rmse}")
print(f"r2 value  on test data {r2}")

