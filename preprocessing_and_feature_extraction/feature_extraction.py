import os
import pandas as pd
import numpy as np
from itertools import chain

def missingness_features(patient, sep_columns):
    """
    measurement frequency feature
    """
    ret = np.array(patient)
    for sep_column in sep_columns:
        sep_data = np.array(patient[sep_column])
        nan_pos = np.where(~np.isnan(sep_data))[0]
        interval_f1 = sep_data.copy()
        interval_f2 = sep_data.copy()
        if len(nan_pos) == 0:
            interval_f1[:] = 0
            ret = np.column_stack((ret, interval_f1))
            interval_f2[:] = -1
            ret = np.column_stack((ret, interval_f2))
        else:
            interval_f1[: nan_pos[0]] = 0
            for p in range(len(nan_pos)-1):
                interval_f1[nan_pos[p]: nan_pos[p+1]] = p + 1
            interval_f1[nan_pos[-1]:] = len(nan_pos)
            ret = np.column_stack((ret, interval_f1))

            interval_f2[:nan_pos[0]] = -1
            for q in range(len(nan_pos) - 1):
                length = nan_pos[q+1] - nan_pos[q]
                for l in range(length):
                    interval_f2[nan_pos[q] + l] = l

            length = len(patient) - nan_pos[-1]
            for l in range(length):
                interval_f2[nan_pos[-1] + l] = l
            ret = np.column_stack((ret, interval_f2))

    return ret

def empiric_score(empiric_data):
    """
    emperic risk score feature
    """
    risk_score = np.zeros((len(empiric_data), 8))
    for ii in range(len(empiric_data)):

        Temp = empiric_data[ii, 2]
        if Temp == np.nan:
            Temp_score = np.nan
        elif Temp <= 35:
            Temp_score = 3
        elif Temp >= 39.1:
            Temp_score = 2
        elif (35.1 <= Temp <= 36.0) | (38.1 <= Temp <= 39.0):
            Temp_score = 1
        else:
            Temp_score = 0
        risk_score[ii, 1] = Temp_score

        HR = empiric_data[ii, 0]
        if HR == np.nan:
            HR_score = np.nan
        elif (HR <= 40) | (HR >= 131):
            HR_score = 3
        elif 111 <= HR <= 130:
            HR_score = 2
        elif (41 <= HR <= 50) | (91 <= HR <= 110):
            HR_score = 1
        else:
            HR_score = 0
        risk_score[ii, 0] = HR_score


        Creatinine = empiric_data[ii, 19]
        if Creatinine == np.nan:
            Creatinine_score = np.nan
        elif Creatinine < 1.2:
            Creatinine_score = 0
        elif Creatinine < 2:
            Creatinine_score = 1
        elif Creatinine < 3.5:
            Creatinine_score = 2
        else:
            Creatinine_score = 3
        risk_score[ii, 3] = Creatinine_score

        Resp = empiric_data[ii, 6]
        if Resp == np.nan:
            Resp_score = np.nan
        elif (Resp < 8) | (Resp > 25):
            Resp_score = 3
        elif 21 <= Resp <= 24:
            Resp_score = 2
        elif 9 <= Resp <= 11:
            Resp_score = 1
        else:
            Resp_score = 0
        risk_score[ii, 2] = Resp_score

        Platelets = empiric_data[ii, 30]
        if Platelets == np.nan:
            Platelets_score = np.nan
        elif Platelets <= 50:
            Platelets_score = 3
        elif Platelets <= 100:
            Platelets_score = 2
        elif Platelets <= 150:
            Platelets_score = 1
        else:
            Platelets_score = 0
        risk_score[ii, 6] = Platelets_score

        MAP = empiric_data[ii, 4]
        if MAP == np.nan:
            MAP_score = np.nan
        elif MAP >= 70:
            MAP_score = 0
        else:
            MAP_score = 1
        risk_score[ii, 4] = MAP_score

        SBP = empiric_data[ii, 3]
        Resp = empiric_data[ii, 6]
        if SBP + Resp == np.nan:
            qsofa = np.nan
        elif (SBP <= 100) & (Resp >= 22):
            qsofa = 1
        else:
            qsofa = 0
        risk_score[ii, 5] = qsofa

        Bilirubin = empiric_data[ii, 25]
        if Bilirubin == np.nan:
            Bilirubin_score = np.nan
        elif Bilirubin < 1.2:
            Bilirubin_score = 0
        elif Bilirubin < 2:
            Bilirubin_score = 1
        elif Bilirubin < 6:
            Bilirubin_score = 2
        else:
            Bilirubin_score = 3
        risk_score[ii, 7] = Bilirubin_score

    return risk_score

if __name__ == "__main__":

    file = "Dataset.csv"
    df = pd.read_csv(file)
    patient = df.loc[df["Patient_ID"] == 19877]
    pid = patient.drop(columns=['Bilirubin_direct', 'TroponinI', 'Fibrinogen', 'SepsisLabel'])
    ret = missingness_features(pid, ['Temp'])
    temp = pd.DataFrame(ret)
    temp = pid.fillna(method='ffill')
    risk_score = empiric_score(temp)
    print(risk_score)
    #temp.to_csv("temp.csv")
    df = pid.drop(["SepsisLabel"], axis=1)
    correlation = df.corrwith(pid["SepsisLabel"]).sort_values(ascending=False)