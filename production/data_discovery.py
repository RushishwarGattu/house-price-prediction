
# Third-party imports
import os.path as op
import pandas as pd
import great_expectations as ge

# Project imports
from ta_lib.core.api import display_as_tabs, initialize_environment

# Initialization
initialize_environment(debug=False, hide_warnings=True)
from ta_lib.core.api import create_context, list_datasets, load_dataset
config_path = op.join('conf', 'config.yml')
context = create_context(config_path)
# load datasets
housing_df = load_dataset(context, 'raw/housing')
# Import the eda API
import ta_lib.eda.api as eda
display_as_tabs([('housing', housing_df.shape)])
sum1 = eda.get_variable_summary(housing_df)


display_as_tabs([('housing', sum1)])
housing_df.isna().sum()
sum1, plot1 = eda.get_data_health_summary(housing_df, return_plot=True)


display_as_tabs([('housing', plot1)])
sum1 = eda.get_duplicate_columns(housing_df)


display_as_tabs([('housing', sum1)])