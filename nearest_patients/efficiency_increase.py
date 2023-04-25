import pandas as pd
import numpy as np

data = pd.read_csv('../no_additional_features/test_set_interpolation_with_multivariate.csv').drop(['Hour', 'SepsisLabel'], axis=1)
patients = data['Patient_ID'].unique()

full_df = pd.read_csv('full_nearest_neighbours_test_mutlivariate.csv', index_col=0)
full_df.columns = full_df.columns.astype(int)
full_df.index = full_df.index.astype(int)

nearest_patients_df = {}
for p in patients:
    if int(p) % 100 == 0:
        print(p)
    nearest_patients_df[p] = full_df.nsmallest(5, p)[p].index.tolist()

nearest_patients_df = pd.DataFrame.from_dict(nearest_patients_df)
print(nearest_patients_df.head())
nearest_patients_df.to_csv('nearest_neighbours_efficient.csv')