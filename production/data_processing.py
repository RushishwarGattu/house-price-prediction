from pprint import pprint
import os
import os.path as op
import shutil

# standard third party imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
pd.options.mode.use_inf_as_na = True
# standard code-template imports
from ta_lib.core.api import (
    create_context, get_dataframe, get_feature_names_from_column_transformer, get_package_path,
    display_as_tabs, string_cleaning, merge_info, initialize_environment,
    list_datasets, load_dataset, save_dataset
)
import ta_lib.eda.api as eda
import warnings

warnings.filterwarnings('ignore', message="The default value of regex will change from True to False in a future version.", 
                        category=FutureWarning)
initialize_environment(debug=False, hide_warnings=True)
config_path = op.join('conf', 'config.yml')
context = create_context(config_path)
pprint(list_datasets(context))
housing_df = load_dataset(context, 'raw/housing')
housing_df["rooms_per_household"] = housing_df["total_rooms"]/housing_df["households"]
housing_df["bedrooms_per_room"] = housing_df["total_bedrooms"]/housing_df["total_rooms"]
housing_df["population_per_household"] = housing_df["population"] / \
    housing_df["households"]
X = housing_df.drop('median_house_value', axis=1)
X['total_bedrooms'].fillna(X['total_bedrooms'].mean(), inplace=True)
X['bedrooms_per_room'].fillna(X['bedrooms_per_room'].mean(), inplace=True)
X = pd.get_dummies(X, drop_first=True)
y = housing_df['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
print("generated train and test datasets successfully")
save_dataset(context, X_train, 'train/housing/features')
save_dataset(context, y_train, 'train/housing/target')

save_dataset(context, X_test, 'test/housing/features')
save_dataset(context, y_test, 'test/housing/target')