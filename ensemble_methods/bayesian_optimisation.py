import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Implementing U SCORE
def compute_prediction_utility(labels, predictions, dt_early=-12, dt_optimal=-6, dt_late=3.0, max_u_tp=1, min_u_fn=-2, u_fp=-0.05, u_tn=0):
    # Check inputs for errors.
    # Does the patient eventually have sepsis?
    if np.any(labels):
        is_septic = True
        t_sepsis = np.argmax(labels) - dt_optimal 
    else:
        is_septic = False
        t_sepsis = float('inf')

    n = len(labels)

    # Define slopes and intercept points for utility functions of the form
    # u = m * t + b.
    m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal

    # Compare predicted and true conditions.
    u = np.zeros(n)
    # loop over each hour and evaluate the prediction
    for t in range(n):
        if t <= t_sepsis + dt_late:
            # TP
            if is_septic and predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                elif t <= t_sepsis + dt_late:
                    u[t] = m_2 * (t - t_sepsis) + b_2
            # FP
            elif not is_septic and predictions[t]:
                u[t] = u_fp
            # FN
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0
                elif t <= t_sepsis + dt_late:
                    u[t] = m_3 * (t - t_sepsis) + b_3
            # TN
            elif not is_septic and not predictions[t]:
                u[t] = u_tn

    # Find total utility for patient.
    return np.sum(u)


# create class rebalancing function
# Group dataframe by patient ID
def class_balancer(df):
  grouped_df = df.groupby('Patient_ID')
  # Check if all sepsis label values for each patient are 1
  admitted_with_sepsis = grouped_df['SepsisLabel'].all()
  # Dataframe of patients who are admitted with sepsis
  admitted_with_sepsis_df = df[df['Patient_ID'].isin(admitted_with_sepsis[admitted_with_sepsis].index)]
  # list of patients admitted with sepsis
  admitted_with_sepsis_list = admitted_with_sepsis_df.Patient_ID.unique()
  # list of patients who have sepsis
  septic_patients_list =df['Patient_ID'][df['SepsisLabel']==1].unique()
  # dataframe of septic patients
  septic_df  = df[df.Patient_ID.isin(septic_patients_list)]
  mask = ~septic_df['Patient_ID'].isin(admitted_with_sepsis_list)
  developed_sepsis_df = septic_df[mask]
  # only get septic data and the 10 hours before this from the patients who developed sepsis
  result = pd.DataFrame(columns=developed_sepsis_df.columns)
  for patient_id, group in developed_sepsis_df.groupby('Patient_ID'):
    # find the index of the first row where SepsisLabel is 1
    sepsis_index = group.index[group['SepsisLabel'] == 1]
    # select the rows for this patient, starting 10 rows before sepsis_index
    start_index = max(sepsis_index.min() - 10, 0)
    end_index = sepsis_index.max()
    selected_rows = group.loc[start_index:]
    result = pd.concat([result, selected_rows], axis=0)
  # never get sepsis patients
  num_of_zeros = result.shape[0] + admitted_with_sepsis_df.shape[0]
  nosepsis = df[~df['Patient_ID'].isin(septic_patients_list)].sample(n=num_of_zeros)
  
  return pd.concat([admitted_with_sepsis_df, result, nosepsis]).reset_index(drop=True)

# hyperparameter optimization
# def rf_val(n_estimators, max_depth, min_samples_split, min_samples_leaf):
  # clf = RandomForestClassifier(
  #       n_estimators=int(n_estimators),
  #       max_depth=int(max_depth),
  #       min_samples_split=int(min_samples_split),
  #       min_samples_leaf=int(min_samples_leaf),
  #       n_jobs=-1
  #   )
def objective_function(n_estimators, colsample_bytree, gamma, learning_rate, max_depth):
  model =  xgb.XGBClassifier(objective="binary:logistic", 
                             n_estimators = int(n_estimators),
                             colsample_bytree = colsample_bytree,
                             max_depth = int(max_depth),
                             gamma = gamma,
                             learning_rate = learning_rate,
                             random_state=42)
  model.fit(train_X, train_y.astype('int'))
  y_pred = model.predict(val_X)

  # Group Patients by ID
  val_y_withID = val_y.to_frame().join(val_X['Patient_ID'])
  grouped = val_y_withID.groupby('Patient_ID').groups
  u_list=[]

  # Initilalise Utilities
  num_patients = val_y_withID['Patient_ID'].nunique()

  observed_utilities = np.zeros(num_patients)
  best_utilities = np.zeros(num_patients)
  inaction_utilities = np.zeros(num_patients)

  k = 0
  # iterate through the preditions and ground truth labels of each patient
  for id, idx in grouped.items():
    patient_actual = val_y[idx[0]:idx[-1]+1]
    patient_pred = y_pred[idx[0]:idx[-1]+1]

    best_predictions = np.zeros(len(patient_actual))
    inaction_predictions = np.zeros(len(patient_actual))

    # create best prediction vector
    if np.any(patient_actual):
      t_sepsis = np.argmax(patient_actual) + 6
      best_predictions[max(0, t_sepsis - 12) : min(t_sepsis + 3 + 1, len(best_predictions))] = 1

    # compute the utility scores 
    observed_utilities[k] = compute_prediction_utility(patient_actual,patient_pred)
    best_utilities[k] = compute_prediction_utility(patient_actual,best_predictions)
    inaction_utilities[k] = compute_prediction_utility(patient_actual, inaction_predictions)

    k += 1 

  # sum the u-score for all patients 
  unnormalized_observed_utility = np.sum(observed_utilities)
  unnormalized_best_utility = np.sum(best_utilities)
  unnormalized_inaction_utility = np.sum(inaction_utilities)

  # normalise the u-score
  print((unnormalized_observed_utility - unnormalized_inaction_utility) / (unnormalized_best_utility - unnormalized_inaction_utility))

  return -(1-(unnormalized_observed_utility - unnormalized_inaction_utility) / (unnormalized_best_utility - unnormalized_inaction_utility))


# load in the validtion and training data
val_set = pd.read_csv('C:/Users/Callum Paton/Downloads/temp_val_data.csv')
train_set = pd.read_csv('C:/Users/Callum Paton/Downloads/temp_train_data.csv')
#test_set = pd.read_csv('C:/Users/Callum Paton/Downloads/temp_test_data.csv')
train_set = train_set.drop('dPatient_ID', axis=1)

# balance the training set
train_set = class_balancer(train_set)

# prepare training set, fill na with -1000 as fail safe
train_y = train_set['SepsisLabel']
train_X = train_set.drop('SepsisLabel', axis = 1)
train_X = train_X.fillna(-1000)

# prepare training set, fill na with -1000 as fail safe
val_y = val_set['SepsisLabel']
val_X = val_set.drop('SepsisLabel', axis = 1)
val_X = val_set.fillna(-1000)
val_X = val_X.drop('SepsisLabel',axis=1)

# set bounds for search (Random Forest)
# pbounds = {
#     'n_estimators': (1500, 1900),
#     'max_depth': (3, 8),
#     'min_samples_split': (2, 8),
#     'min_samples_leaf': (2, 9)
# }

# Bounds for search (Ensemble methods)
pbounds = {
    "colsample_bytree": (0.3, 0.7),
    "gamma": (0, 0.5),
    "learning_rate": (0.03, 0.3), # default 0.1 
    "max_depth": (2, 10), # default 3
    "n_estimators": (100, 150), # default 100
}

# prepare and run the optimisation
optimizer = BayesianOptimization(
    f=objective_function,
    pbounds=pbounds,
    random_state=42
)

optimizer.maximize(init_points=8, n_iter=30)