import pandas as pd
import numpy as np
from hausdorff import hausdorff_distance
import itertools

data = pd.read_csv('no_additional_features/test_set_interpolation_with_multivariate.csv').drop(['Hour', 'SepsisLabel'], axis=1)
patients = data['Patient_ID'].unique()
print(len(patients))
patients_splits = len(patients) / 11
patients_splits = [(patients[0+i], patients[898 + i]) for i in range(0, len(patients), 899)]
for s in range(len(patients_splits)):
    p_dists = {}
    for p in patients[s*899:(s+1)*899]:
        p_dists[p] = {}
    p_pairs = itertools.combinations(patients, 2)
    p = 0
    index = 0
    for pair in p_pairs:
        if pair[0] >= patients_splits[s][0]:
            if pair[0] != p:
                print(pair[0])
                p = pair[0]
            if pair[0] > patients_splits[s][1]:
                break
            distance = hausdorff_distance(
                    data[data['Patient_ID'] == pair[0]].drop(['Patient_ID'], axis=1).to_numpy(),
                    data[data['Patient_ID'] == pair[1]].drop(['Patient_ID'], axis=1).to_numpy()
                )
            p_dists[pair[0]][pair[0]] = np.nan
            p_dists[pair[0]][pair[1]] = distance
        
    print(p_dists)
        
    df = pd.DataFrame.from_dict(p_dists)
    df.to_csv('neighbours_test_multivariate_split_'+str(s)+'.csv')

    df = pd.read_csv('neighbours_test_multivariate_split_'+str(s)+'.csv', index_col=0)
    df.columns = df.columns.astype(int)
    print(df.head(50))