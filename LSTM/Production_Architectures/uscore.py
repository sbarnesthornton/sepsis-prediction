import numpy as np
import pandas as pd

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

def Uscore(y_pred, y_actual, X_actual):
    # Group Patients by ID
    val_y_withID = y_actual.to_frame().join(X_actual['Patient_ID'])
    grouped = val_y_withID.groupby('Patient_ID').groups
    u_list=[]

    # Initilalise Utilities
    num_patients = val_y_withID['Patient_ID'].nunique()

    observed_utilities = np.zeros(num_patients)
    best_utilities = np.zeros(num_patients)
    inaction_utilities = np.zeros(num_patients)

    k = 0

    for id, idx in grouped.items():
        patient_actual = y_actual[idx[0]:idx[-1]+1]
        patient_pred = y_pred[idx[0]:idx[-1]+1]

        best_predictions = np.zeros(len(patient_actual))
        inaction_predictions = np.zeros(len(patient_actual))

        if np.any(patient_actual):
            t_sepsis = np.argmax(patient_actual) + 6
            best_predictions[max(0, t_sepsis - 12) : min(t_sepsis + 3 + 1, len(best_predictions))] = 1

        observed_utilities[k] = compute_prediction_utility(patient_actual,patient_pred)
        best_utilities[k] = compute_prediction_utility(patient_actual,best_predictions)
        inaction_utilities[k] = compute_prediction_utility(patient_actual, inaction_predictions)

        k += 1 

    unnormalized_observed_utility = np.sum(observed_utilities)
    unnormalized_best_utility = np.sum(best_utilities)
    unnormalized_inaction_utility = np.sum(inaction_utilities)

    return (unnormalized_observed_utility - unnormalized_inaction_utility) / (unnormalized_best_utility - unnormalized_inaction_utility)