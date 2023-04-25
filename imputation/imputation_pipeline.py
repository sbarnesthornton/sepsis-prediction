import random       
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
import pandas as pd

def fill_early_values(input_df):
    df = input_df.copy()
    for c in df.columns:
        if not df[c].isna().all():
            for i in df[c].index:
                if not np.isnan(df[c][i]):
                    value = df[c][i]
                    index = i
                    break
            for i in range(df[c].index[0], index):
                df.at[i, c] = value
    return df

def worker(index, r, df):
    imp_mean = IterativeImputer(random_state=r, max_iter=50, missing_values=np.nan, verbose=2, imputation_order='ascending')
    imputed_data = imp_mean.fit_transform(df)
    imputed_df = pd.DataFrame(imputed_data, columns=imp_mean.get_feature_names_out())
    return imputed_df

print('Reading in Data')
# filtered_data = pre-processed
filtered_data = pd.read_csv('imputation_data/filtered_data.csv')

# Input data should not have columns: Unit1, Unit2, 'Unnamed: 0', Hou and SepsisLabel
# These columns can be added back in afterwards as the order of the rows is not changed during imputation
filtered_data = filtered_data.drop([filtered_data.columns[0], 'Unit1', 'Unit2'], axis=1)
impute_data = filtered_data.drop(['Hour', 'SepsisLabel'], axis=1)

print('Grouping data by patient and backfilling from first reading in all features')
group_impute_data = impute_data.groupby('Patient_ID', as_index=False, sort=False)
group_filled_data = group_impute_data.apply(fill_early_values)
print('Regrouping by patient ID and performing interpolation')
group_impute_data = group_filled_data.groupby('Patient_ID')
group_imputed_data = group_impute_data.apply(lambda x: x.interpolate(method='linear'))

print('Performing multivariate imputation on very sparse features')
# If you want to leave out particular columns, they could be dropped here and added back afterwards
# e.g. If you think adding an extra feature for last recording or something similar would be more beneficial
group_imputed_data = group_imputed_data.drop(['Patient_ID'], axis=1)
imputed_data = worker(0, random.randint(0, 2000), group_imputed_data)

# Example of adding back in columns
imputed_data['Patient_ID'] = impute_data['Patient_ID']
imputed_data['Hour'] = filtered_data['Hour']

print('Head of data:')
print(imputed_data.head())

print('Outputting to csv...')
imputed_data.to_csv('imputed_data_example.csv')