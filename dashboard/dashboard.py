import pandas as pd
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from dash_bootstrap_templates import load_figure_template
import igraph
from igraph import Graph, EdgeSeq
from hausdorff import hausdorff_distance
import itertools
import math
import dash_daq as daq

TOP_HALF_HEIGHT = 85
CHART_MEASURE = 'HR'


SOFA_FEATURES = ['HR', 'Temp', 'Resp', 'Creatinine', 'MAP', 'qSOFA', 'Platelets', 'Bilirubin_total']

data = (
    pd.read_csv("dashboard_multivariate_set.csv")
)

sepsis_probs = (
    pd.read_csv('test_set_prediction_probabilities_v1.csv')
)

nearest_patients_df = (
    pd.read_csv('nearest_neighbours_efficient.csv', index_col=0)
)

patients = data['Patient_ID'].unique()

df = round(100*(data.isnull().sum()/len(data.index)),2)
df = pd.DataFrame({'Measurement':df.index, 'No. of Missing Values':df.values})
app = Dash(__name__, external_stylesheets=[dbc.themes.SOLAR, dbc.icons.BOOTSTRAP])
load_figure_template(['solar'])

SCORE_RANGES = {SOFA_FEATURES[0]:(0, 150), SOFA_FEATURES[1]:(30, 42), SOFA_FEATURES[2]:(0,40),SOFA_FEATURES[3]:(0,6),SOFA_FEATURES[4]:(0, 100),SOFA_FEATURES[5]:(0,1),SOFA_FEATURES[6]:(0,200),SOFA_FEATURES[7]:(0,7)}
RISK_COLOURS= {0:'#238823', 1:'#fab733', 2:'#ff8e15', 3:'#d2222d'}
PREV_PATIENT = None
NEAREST_PATIENTS = None


def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]

def empiric_risk_score(patient, hour):
    ''' 
    finds the empiric (risk) score for HR, SBP, MAP, Resp, Temp, creatine,
    platelets and total bilirubin according to scoring system of NEWS,
    SOFA and qSOFA 
    (https://github.com/Meicheng-SEU/EASP/blob/master/feature_engineering.py)
    '''
    scores = {}
    ii = hour
    HR = patient.at[ii, SOFA_FEATURES[0]]
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
    scores[SOFA_FEATURES[0]] = HR_score

    Temp = patient.at[ii, SOFA_FEATURES[1]]
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
    scores[SOFA_FEATURES[1]] = Temp_score

    Resp = patient.at[ii, SOFA_FEATURES[2]]
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
    scores[SOFA_FEATURES[2]] = Resp_score

    Creatinine = patient.at[ii, SOFA_FEATURES[3]]
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
    scores[SOFA_FEATURES[3]] = Creatinine_score

    MAP = patient.at[ii, SOFA_FEATURES[4]]
    if MAP == np.nan:
        MAP_score = np.nan
    elif MAP >= 70:
        MAP_score = 0
    else:
        MAP_score = 1
    scores[SOFA_FEATURES[4]] = MAP_score

    SBP = patient.at[ii, 'SBP']
    Resp = patient.at[ii, 'Resp']
    if SBP + Resp == np.nan:
        qsofa = np.nan
    elif (SBP <= 100) & (Resp >= 22):
        qsofa = 1
    else:
        qsofa = 0
    scores[SOFA_FEATURES[5]] = qsofa

    Platelets = patient.at[ii, SOFA_FEATURES[6]]
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
    scores[SOFA_FEATURES[6]] = Platelets_score

    Bilirubin = patient.at[ii, SOFA_FEATURES[7]]
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
    scores[SOFA_FEATURES[7]] = Bilirubin_score
    return scores

def empiric_risk_labels(intervals, feature):
    ''' 
    finds the empiric (risk) score for HR, SBP, MAP, Resp, Temp, creatine,
    platelets and total bilirubin according to scoring system of NEWS,
    SOFA and qSOFA 
    (https://github.com/Meicheng-SEU/EASP/blob/master/feature_engineering.py)
    '''
    custom = {}
    if feature == SOFA_FEATURES[0]:
        for HR in intervals:
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
            custom[HR] = {'label':str(HR_score)}
    elif feature == SOFA_FEATURES[1]:
        for Temp in intervals:
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
            custom[Temp] = {'label':str(Temp_score)}
    elif feature == SOFA_FEATURES[2]:
        for Resp in intervals:
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
            custom[Resp] = {'label':str(Resp_score)}
    elif feature == SOFA_FEATURES[3]:
        for Creatinine in intervals:
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
            custom[Creatinine_score] = {'label':str(Creatinine_score)}
    elif feature == SOFA_FEATURES[4]:
        for MAP in intervals:
            if MAP == np.nan:
                MAP_score = np.nan
            elif MAP >= 70:
                MAP_score = 0
            else:
                MAP_score = 1
            custom[MAP] = {'label':str(MAP_score), 'style':{'font-size':'1px'}}
    elif feature == SOFA_FEATURES[5]:
        custom[0] = {'label:':str(0)}
        custom[1] = {'label':str(1)}
    elif feature == SOFA_FEATURES[6]:
        for Platelets in intervals:
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
            custom[Platelets] = {'label':str(Platelets_score)}
    elif feature == SOFA_FEATURES[7]:
        for Bilirubin in intervals:
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
            custom[Bilirubin] = {'label':str(Bilirubin_score)}
    return custom

def get_score_gauge(title, width, risk, val):
    custom = empiric_risk_labels(range(SCORE_RANGES[title][0], SCORE_RANGES[title][1], 1), title)
    risk = risk.tolist()[0]
    colour = RISK_COLOURS[val]
    if title == 'qSOFA':
        colour = RISK_COLOURS[3]
    return daq.Gauge(
        id=title+'-gauge',
        value=risk,
        label=title,
        style={'width': str(width * 100) + '%', 'height': str(width * 100) + '%'},
        size=width * 275,
        min=SCORE_RANGES[title][0],
        max=SCORE_RANGES[title][1],
        scale={'interval':(SCORE_RANGES[title][1]-SCORE_RANGES[title][0])/20, 'labelInterval':(SCORE_RANGES[title][1]-SCORE_RANGES[title][0])/20},
        color=colour
    )

def create_gauges(features, patient_id, hour):
    if len(features) > 4:
        line_1=create_gauges(features[:4], patient_id, hour)
        line_2=create_gauges(features[4:], patient_id, hour)
        return [line_1[0], line_1[1], line_2[0], line_2[1]]
    width = 1/len(features)
    scores = empiric_risk_score(patient_id, hour)
    patient_id['qSOFA'] = np.zeros(len(patient_id))
    patient_id.loc[patient_id['Hour']==hour,'qSOFA'] = scores['qSOFA']
    return [
        dbc.Row([
            get_score_gauge(f, width, patient_id[patient_id['Hour']==hour][f], scores[f]) for f in features
        ]),
        dbc.Row([
            html.P(
                str(scores[f]),
                className="text-center",
                style={'width':str(width*100) + '%', 'padding':'{0,0,0,0}', 'font-weight':'bold'}
            ) for f in features
        ])
        ]
    
def get_patient_labels():
    sepsis_patients = data[data['SepsisLabel']==1]['Patient_ID'].unique()
    labels = []
    for p in patients:
        d = {}
        if p in sepsis_patients:
            d['label'] = html.Span([str(p) + '  ', html.I(className="bi bi-exclamation-triangle-fill me-2")])
        else:
            d['label'] = str(p)
        d['value'] = p
        labels.append(d)
    return labels

def get_hour_predictions(hour, patient):
    patient_sepsis_probs = sepsis_probs[sepsis_probs['Patient_ID'] == patient].reset_index(drop=True)
    preds = []
    for h in range(hour+1):
        preds.append(patient_sepsis_probs.at[h, 'SepsisProb'] * 100)
    return preds


header = html.H4(
    "Visualisation of Sepsis Prediction", className="bg-primary text-white p-2 mb-2 mt-2 text-center rounded-2"
)

app.title = 'Sepsis Predictor'

app.layout = dbc.Container(
    [
        header,
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Container(
                            [
                                html.P(
                                    "Percentage Chance of Sepsis:",
                                    className="lead text-center"
                                ),
                                dbc.Container(
                                    [
                                        dbc.Progress(value=0, color='#ffff00', style={'height':'100%', 'color': 'white'}, id='sepsis-prediction-value')
                                    ],
                                    className='mb-2',
                                    style={'height':'25%'}
                                ),
                                dbc.Container(
                                    dcc.Graph(style={'height':'100%'}, id='predictions-graph', config=dict(displayModeBar=False)),
                                    style={'height':'30%', 'color': 'rgba{0,0,0,0}'}
                                )
                            ],
                            fluid=True,
                            className="py-3 bg-light rounded-2",
                            style={'height':str(TOP_HALF_HEIGHT * 1/3) + 'vh'}
                        ),
                        dbc.Card(
                            [
                                html.P(
                                    "Empirical Risk Scores",
                                    className="lead text-center"
                                ),
                                dbc.Container(
                                    [],        
                                    id='gauges-card'
                                )    
                            ],
                            className='rounded-2 mb-2 mt-2',
                            body=True,
                            style={'height':'calc(' + str(TOP_HALF_HEIGHT * 2/3) + 'vh - 0.5rem)'}
                        )                                    
                    ],
                    width=5
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            className='rounded-2',
                            body=True,
                            style={'height':str(TOP_HALF_HEIGHT) + 'vh'},
                            children=[
                                dbc.Row(style={'flex-wrap':'no-wrap'},children=[
                                    dcc.Dropdown(get_patient_labels(), id='patient-dropdown', value=data['Patient_ID'].unique()[0], clearable=False, style={'width': '40%'})
                                ]),
                                html.Div(id='nearest-patient'),
                                dbc.Row([
                                    dbc.Col(
                                        [
                                            dcc.Tabs(id='tabs-patient-graphs', style={'height':'80%'}, value='HR', vertical = True, children=[
                                                dcc.Tab(label=m, value = m, style={'font-size':'1.1vh', 'padding': '0.5px'},
                                                        selected_style={'font-size':'1.1vh', 'padding': '0.5px'}) for m in data.drop(['SepsisLabel', 'Hour', 'Gender', 'Patient_ID', 'Age', 'ICULOS', 'HospAdmTime'], axis=1).columns
                                            ]),
                                        ],
                                        width=1
                                    ),
                                    dbc.Col([
                                        html.Div(id='patient-graph-container', style={'height': '90%'})
                                        ],
                                        width=9
                                    ),
                                    dbc.Col([
                                        html.Div(
                                            ['Nearest Patients:'],
                                            style={
                                                'padding-bottom':'10px'
                                            }
                                        ),
                                        dcc.Checklist(
                                            patients[:3],
                                            [],
                                            style={
                                                'margin':'auto'
                                            },
                                            id='nearest-patients-checklist'
                                        )
                                        ],
                                        width=2
                                    )
                                ],
                                style={'height': '100%'})
                            ]
                        )
                    ],
                    width=7
                )
            ]
        ),
        dbc.Row([
            dbc.Col(id='slider-container', children= [
                dcc.Slider(0, data['Hour'].max(), 1, id='hour-slider', value=0)
            ],
            width=12)
        ])
    ],
    fluid=True,
    className="dbc"
)

def get_time_until_sepsis(patient, hour):
    patient_data = data[data['Patient_ID'] == patient]
    sepsis_patient_data = patient_data[patient_data['SepsisLabel'] == 1]
    sepsis_hour = sepsis_patient_data['Hour'].min()
    if hour == None:
        hour = 0
    if not pd.isna(sepsis_hour):
        difference = int(sepsis_hour)-int(hour)
        if difference < 0:
            difference = 0
        return html.Span(['(' + str(difference) + ' hrs to sepsis)'], style={'color':'red'})
    else:
        return html.Span(['(Not septic)'])


@app.callback(
    Output('hour-slider', 'max'),
    Input('patient-dropdown', 'value')
)
def update_slider(patient):
    return data[data['Patient_ID'] == patient]['Hour'].max()

@app.callback(
    Output('nearest-patients-checklist', 'options'),
    Output('nearest-patients-checklist', 'value'),
    Input('patient-dropdown', 'value'),
    Input('hour-slider', 'value'),
    Input('nearest-patients-checklist', 'value')
)
def update_nearest_patients(patient, hour, selected_nearest_patients):
    nearest_patients = nearest_patients_df[str(patient)].tolist()
    sepsis_times = [get_time_until_sepsis(p, hour) for p in nearest_patients]
    nearest_patients_list = [{'label':html.Span([html.Span([str(nearest_patients[i])], style={'font-weight':'bold'}), html.Br(), sepsis_times[i]], style={'padding-left':'5px'}), 'value': nearest_patients[i]} for i in range(len(nearest_patients))]
    value =[]
    for p in selected_nearest_patients:
        if p in nearest_patients:
            value.append(p)
    return nearest_patients_list, value

@app.callback(
    Output('sepsis-prediction-value', 'color'),
    Input('sepsis-prediction-value', 'value')
)
def update_prediction_color(value):
    color_scale = ['#ff0d0d', '#ff4e11', '#ff8e15', '#fab733', '#acc334', '#7cf334'][::-1]
    color_scale = get_color_gradient('#007000', '#238823', 25) + get_color_gradient('#238823', '#FFBF00', 25) + get_color_gradient('#FFBF00', '#D2222D', 50)
    increment = 100 / len(color_scale)
    increments = math.floor(value / increment)
    return color_scale[increments]

@app.callback(
    Output('patient-graph-container', 'children'),
    Output('sepsis-prediction-value', 'value'),
    Output('sepsis-prediction-value', 'label'),
    Output('predictions-graph', 'figure'),
    Output('gauges-card', 'children'),
    Input('hour-slider', 'value'),
    Input('tabs-patient-graphs', 'value'),
    Input('patient-dropdown', 'value'),
    Input('nearest-patients-checklist', 'value'))
def update_graph(value, measure, patient, nearest_patients):
    try:
        graph_df = data[data['Hour'] <= int(value)]
    except:
        value = 0
        graph_df = data[data['Hour'] <= 0]
    no_lim_patient_df = data[data['Patient_ID'] == patient]
    if value > no_lim_patient_df['Hour'].max():
        value = no_lim_patient_df['Hour'].max()
    sepsis_prob = sepsis_probs[sepsis_probs['Patient_ID'] == patient].reset_index(drop=True).at[value, 'SepsisProb']
    
    graph_df = graph_df.isnull().sum()
    prediction_graph_layout = {
        'xaxis':{
            'showgrid':False,
            'showticklabels':False,
            'range': [0,no_lim_patient_df['Hour'].max()],
            'zeroline': False,
            'fixedrange': True
            }, 
        'yaxis':
            {
                'showgrid':False,
                'showticklabels':True,
                'range': [0,100],
                'zeroline': False,
                'fixedrange': True
            },
        'showlegend': False,
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'margin': {'l':0, 'r':0, 't':10, 'b':10},
        'dragmode': False
        }
    patient_df = no_lim_patient_df[no_lim_patient_df['Hour'] <= value]
    pred_figure = px.line(get_hour_predictions(int(value), patient))
    pred_figure.layout = prediction_graph_layout
    CHART_MEASURE = measure
    y_range = [no_lim_patient_df[CHART_MEASURE].min()-np.abs(no_lim_patient_df[CHART_MEASURE].min()/10), no_lim_patient_df[CHART_MEASURE].max() + np.abs(no_lim_patient_df[CHART_MEASURE].max()/10)]
    for n in nearest_patients:
        n_df = data[data['Patient_ID'] == n]
        min_val = n_df[CHART_MEASURE].min() - np.abs(n_df[CHART_MEASURE].min()/10)
        max_val = n_df[CHART_MEASURE].max() + np.abs(n_df[CHART_MEASURE].max()/10)
        if min_val < y_range[0]:
            y_range[0] = min_val
        if max_val > y_range[1]:
            y_range[1] = max_val
    if value == 0:
        output_figure = px.scatter(patient_df, x='Hour', y=CHART_MEASURE, template='solar').update_xaxes(range=[0, no_lim_patient_df['Hour'].max() + no_lim_patient_df['Hour'].max()/10]).update_yaxes(range=y_range)
    else:
        output_figure = px.line(patient_df, x='Hour', y=CHART_MEASURE, template='solar').update_xaxes(range=[0, no_lim_patient_df['Hour'].max() + no_lim_patient_df['Hour'].max()/10]).update_yaxes(range=y_range)
    if len(nearest_patients) > 0:
        for p in nearest_patients:
            n_df = data[data['Patient_ID'] == p]
            n_df = n_df[n_df['Hour'] < no_lim_patient_df['Hour'].max()]
            output_figure.add_trace(go.Scatter(x=n_df['Hour'], y=n_df[CHART_MEASURE], mode='lines', name=p))
    output_figure.update_traces(hovertemplate=CHART_MEASURE + ": %{y}<extra></extra>")
    output_figure.update_layout(hovermode="x unified")
    output_graph = dcc.Graph(style={'height': '100%'}, figure=output_figure)
    return (
        output_graph,
        sepsis_prob * 100, 
        str(round(sepsis_prob*100, 2)) + '%',
        pred_figure,
        create_gauges(SOFA_FEATURES, no_lim_patient_df.reset_index(), value)
    )


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)