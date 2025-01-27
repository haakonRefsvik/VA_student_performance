import dash
from dash import dcc, html, Input, Output
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # Import make_subplots

# Load and preprocess the data
df = pd.read_csv('data/student_data.csv')

## Preprocessing data.
df = pd.get_dummies(df, columns=['Mjob', 'Fjob', 'reason', 'guardian'], drop_first=True)

df['sex'] = df['sex'].map({'F': 0, 'M': 1})
df['school'] = df['school'].map({'GP': 0, 'MS': 1})
df['address'] = df['address'].map({'U': 0, 'R': 1})
df['Pstatus'] = df['Pstatus'].map({'T': 0, 'A': 1})
df['famsup'] = df['famsup'].map({'yes': 1, 'no': 0})
df['schoolsup'] = df['schoolsup'].map({'yes': 1, 'no': 0})
df['famsize'] = df['famsize'].map({'GT3': 1, 'LE3': 0})
df['activities'] = df['activities'].map({'yes': 1, 'no': 0})
df['paid'] = df['paid'].map({'yes': 1, 'no': 0})
df['higher'] = df['higher'].map({'yes': 1, 'no': 0})
df['nursery'] = df['nursery'].map({'yes': 1, 'no': 0})
df['internet'] = df['internet'].map({'yes': 1, 'no': 0})
df['romantic'] = df['romantic'].map({'yes': 1, 'no': 0})

boolean_columns = [
    'Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher',
    'Fjob_health', 'Fjob_other', 'Fjob_services', 'Fjob_teacher',
    'reason_home', 'reason_other', 'reason_reputation',
    'guardian_mother', 'guardian_other'
]

df[boolean_columns] = df[boolean_columns].astype(int)

##numeric_columns = df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14,15,16,17, 18,19,20,21,22,23, 24, 25, 26, 27, 28, 29, 30,31,32]]
numeric_columns = df.select_dtypes(include=['number'])


'''
0 school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)
1 sex - student's sex (binary: 'F' - female or 'M' - male)
2 age - student's age (numeric: from 15 to 22)
3 address - student's home address type (binary: 'U' - urban or 'R' - rural)
4 famsize - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)
5 Pstatus - parent's cohabitation status (binary: 'T' - living together or 'A' - apart)
6 Medu - mother's education (numeric: 0 - none,  1 - primary education (4th grade), 2 â€“ 5th to 9th grade, 3 â€“ secondary education or 4 â€“ higher education)
7 Fedu - father's education (numeric: 0 - none,  1 - primary education (4th grade), 2 â€“ 5th to 9th grade, 3 â€“ secondary education or 4 â€“ higher education)
8 Mjob - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
9 Fjob - father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
10 reason - reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')
11 guardian - student's guardian (nominal: 'mother', 'father' or 'other')
12 traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
13 studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
14 failures - number of past class failures (numeric: n if 1<=n<3, else 4)
15 schoolsup - extra educational support (binary: yes or no)
16 famsup - family educational support (binary: yes or no)
17 paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
18 activities - extra-curricular activities (binary: yes or no)
19 nursery - attended nursery school (binary: yes or no)
20 higher - wants to take higher education (binary: yes or no)
21 internet - Internet access at home (binary: yes or no)
22 romantic - with a romantic relationship (binary: yes or no)
23 famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
24 freetime - free time after school (numeric: from 1 - very low to 5 - very high)
25 goout - going out with friends (numeric: from 1 - very low to 5 - very high)
26 Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
27 Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
28 health - current health status (numeric: from 1 - very bad to 5 - very good)
29 absences - number of school absences (numeric: from 0 to 93)
30 G1 - first period grade (numeric: from 0 to 20)
31 G2 - second period grade (numeric: from 0 to 20)
32 G3 - final grade (numeric: from 0 to 20, output target)
'''

def explain_attribute(attribute, bin_value):
    attribute_details = {
        'Medu': ['None', 'Primary education (4th grade)', '5th to 9th grade', 'Secondary education', 'Higher education'],
        'Fedu': ['None', 'Primary education (4th grade)', '5th to 9th grade', 'Secondary education', 'Higher education'],
        'reason': ['Close to home', 'School reputation', 'Course preference', 'Other'],
        'traveltime': ['<15 min.', '15 to 30 min.', '30 min. to 1 hour', '>1 hour'],
        'studytime': ['<2 hours', '2 to 5 hours', '5 to 10 hours', '>10 hours'],
        'failures': ['None', '1', '2', '3 or more'],
        'famrel': ['Very bad', 'Bad', 'Neutral', 'Good', 'Excellent'],
        'freetime': ['Very low', 'Low', 'Neutral', 'High', 'Very high'],
        'goout': ['Very low', 'Low', 'Neutral', 'High', 'Very high'],
        'Dalc': ['Very low', 'Low', 'Neutral', 'High', 'Very high'],
        'Walc': ['Very low', 'Low', 'Neutral', 'High', 'Very high'],
        'health': ['Very bad', 'Bad', 'Neutral', 'Good', 'Very good'],
        'absences': [
            f'{int((93 / 5) * bin_value)} - {int((93 / 5) * (bin_value + 1) - 1)} absences' for bin_value in range(5)
        ],
        'G1': [
            f'{int((20 / 5) * bin_value)} - {int((20 / 5) * (bin_value + 1) - 1)} grade' for bin_value in range(5)
        ],
        'G2': [
            f'{int((20 / 5) * bin_value)} - {int((20 / 5) * (bin_value + 1) - 1)} grade' for bin_value in range(5)
        ],
        'G3': [
            f'{int((20 / 5) * bin_value)} - {int((20 / 5) * (bin_value + 1) - 1)} grade' for bin_value in range(5)
        ]
    }

    # Check if the attribute exists and bin_value is within the range
    if attribute in attribute_details and 0 <= bin_value < len(attribute_details[attribute]):
        return attribute_details[attribute][bin_value]
    else:
        return "Invalid attribute or bin value."


histogramWidth = 240
histogramHeight = 400
histogramTitleFontSize = 16
data_standardized = StandardScaler().fit_transform(numeric_columns)

pca = PCA(n_components=5)
df_pca = pca.fit_transform(data_standardized)

tsne = TSNE(n_components=2, perplexity=15, learning_rate=200, random_state=42)
tsne_results = tsne.fit_transform(df_pca)

df['tsne-1'] = tsne_results[:, 0]
df['tsne-2'] = tsne_results[:, 1]

x_margin = (df['tsne-1'].max() - df['tsne-1'].min()) * 0.1
y_margin = (df['tsne-2'].max() - df['tsne-2'].min()) * 0.1

x_min, x_max = df['tsne-1'].min() - x_margin, df['tsne-1'].max() + x_margin
y_min, y_max = df['tsne-2'].min() - y_margin, df['tsne-2'].max() + y_margin

app = dash.Dash(__name__)

@app.callback(
    Output('tsne-plot', 'figure'),
    [Input('gender-histogram', 'selectedData'),
     Input('wants-higher-histogram', 'selectedData'),
     Input('parents-together-histogram', 'selectedData'),
    Input('grade-histogram', 'selectedData')
    ]
)
def update_tsne_plot(studytime_selected, wants_higher_selected, parents_together_selected, grade_selected):    
    selected_indices = set(range(len(df)))

    if studytime_selected and 'points' in studytime_selected:
        studytime_indices = set()
        for point in studytime_selected['points']:
            studytime_indices.update(point['pointNumbers']) 
        selected_indices.intersection_update(studytime_indices)

    if wants_higher_selected and 'points' in wants_higher_selected:
        higher_indices = set()
        for point in wants_higher_selected['points']:
            higher_indices.update(point['pointNumbers'])  
        selected_indices.intersection_update(higher_indices)

    if parents_together_selected and 'points' in parents_together_selected:
        parents_together_indices = set()
        for point in parents_together_selected['points']:
            parents_together_indices.update(point['pointNumbers'])  
        selected_indices.intersection_update(parents_together_indices)

    if grade_selected and 'points' in grade_selected:
        grades_indices = set()
        for point in grade_selected['points']:
            grades_indices.update(point['pointNumbers'])  
        selected_indices.intersection_update(grades_indices)

    if(len(selected_indices) != 0):
        df['highlight'] = 0.2 

    df.loc[list(selected_indices), 'highlight'] = 1  # Set opacity to fully visible for selected points

    # Create the scatter plot
    fig = px.scatter(
        df, x='tsne-1', y='tsne-2', color='G3',
        opacity=df['highlight'],  # Use the 'highlight' column for opacity
        title="t-SNE Visualization",
        labels={'G3': 'Final Grade'},
        hover_data={'tsne-1': False, 'tsne-2': False, 
                    'age': True, 'Medu': True, 'Fedu': True},  
        color_continuous_scale='Viridis',  
        range_color=[0, 20]  
    )

    fig.update_layout(
        height=600,
        width=800,
        dragmode='select',  # Set default to box select tool
        title="Filtered t-SNE Visualization"
    )
    
    return fig


categories = ['Mother Education (Medu)', 
              'Father Education (Fedu)', 
              'Study Time', 
              "Travel time", 
              "Failures",
              "Weekend Alcohol Consuption",
              "Workday Alcohol Consuption",
              ]

@app.callback(
    Output('heatmap', 'figure'),
    [Input('selected-points', 'data'),
    Input('grade-histogram', 'selectedData')
     ]
)
def create_heatmap(selected_points, grade_points):
    selected_df = df  # Show full dataset if no points are selected

    if grade_points and grade_points['points']:
        l = list()
        for points in grade_points['points']:
            for p in points['pointNumbers']:
                l.append(p)

        selected_df = df.iloc[l]
    if selected_points:
        selected_df = df.iloc[selected_points]

    attributes = ['Medu', 'Fedu', 'failures', 'studytime', 'traveltime', 'Walc', 'Dalc', 'health', 'famrel', 'goout', 'freetime']
    
    custom_titles = {
        'Medu': 'Mother Education Level',
        'Fedu': 'Father Education Level',
        'failures': 'Number of Failures',
        'studytime': 'Weekly Study Time',
        'traveltime': 'Travel Time to School',
        'Walc': 'Weekend Alcohol Consumption',
        'Dalc': 'Workday Alcohol Consumption',
        'health': 'Quality of health',
        'famrel': 'Quality of Family Relationships',
        'goout': 'Going Out with Friends',
        'freetime': 'Freetime after school'
    }
    
    num_bins = {
        'Medu': 5, 'Fedu': 5, 'failures': 4, 'studytime': 4, 'traveltime': 4,
        'Walc': 5, 'Dalc': 5, 'health': 5, 'famrel': 5, 'goout': 5, 'freetime': 5
    }
    
    heatmap_data = pd.DataFrame(columns=attributes, index=[0, 1, 2, 3, 4])  
    text_data = pd.DataFrame(columns=attributes, index=[0, 1, 2, 3, 4])  
    hover_text = pd.DataFrame(columns=attributes, index=[0, 1, 2, 3, 4])  

    for attribute in attributes:
        bins = num_bins[attribute]
        discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')

        try:
            binned_data = discretizer.fit_transform(selected_df[[attribute]])
        except ValueError:
            binned_data = np.zeros((selected_df.shape[0], 1))

        bin_counts = pd.Series(binned_data.flatten()).value_counts().sort_index()
        bin_counts = bin_counts.reindex(range(bins)).fillna(0).astype(int)
        total_count = bin_counts.sum()
        if total_count > 0:
            bin_counts_normalized = bin_counts / total_count
        else:
            bin_counts_normalized = bin_counts  

        heatmap_data.loc[0:bins - 1, attribute] = bin_counts_normalized
        text_data.loc[0:bins - 1, attribute] = bin_counts_normalized.apply(lambda x: f"{x:.1f}")
        hover_text.loc[0:bins - 1, attribute] = [
            f"{explain_attribute(attribute, bin_idx)}: {float(distr) * 100:.1f}% of students"
            for bin_idx, distr in enumerate(bin_counts_normalized)
        ]

    heatmap_data.fillna(0, inplace=True)

    fig = go.Figure(go.Heatmap(
        z=heatmap_data.T.values, 
        x=heatmap_data.index,  
        y=[custom_titles[attr] for attr in heatmap_data.columns], 
        colorscale='Inferno',  
        zmin=0, 
        zmax=1,  
        colorbar=dict(title='Relative Distribution'),
        showscale=True,

        texttemplate='%{text}',  
        hoverinfo='text',  
        hovertext=hover_text.T.values,  
    ))

    fig.update_layout(
        title="Heatmap of Students' Attribute Bins (Normalized to Max 1)",
        xaxis_title="Attribute Bins (0-4)",
        yaxis_title="",
        height=600,
        width=800,
    )

    return fig


# App layout
##  ------------------------------------------------------------------------------

app.layout = html.Div([
    html.H1("Interactive t-SNE Visualization"),
    html.Div([
        dcc.Graph(id='gender-histogram', style={'height': '150px', 'width': '240px'}),
        dcc.Graph(id='wants-higher-histogram', style={'height': '150px', 'width': '240px'}),
        dcc.Graph(id='parents-together-histogram', style={'height': '150px', 'width': '240px'}),
        dcc.Graph(id='grade-histogram', style={'height': '150px', 'width': '540px'}),
    ], style={'display': 'flex', 'flex-direction': 'row', 'height': '350px'}),
    html.Div([
        dcc.Graph(id='tsne-plot', 
                    figure=update_tsne_plot([], [], [], []), 
                    style={'height': '600px', 'width': '800px'},
                    config={'displayModeBar': True},  
                  ),
        dcc.Graph(id='heatmap', style={'height': '600px', 'width': '600px'}), 
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    dcc.Store(id='selected-points', data=[]),  
    html.Div(id='selection-output'), 
])

##  ------------------------------------------------------------------------------

@app.callback(
    Output('selected-points', 'data'),
    [Input('tsne-plot', 'selectedData'),
     ]
)
def store_selected_points(selected_data):

    if selected_data:
        selected_points = [point['pointIndex'] for point in selected_data['points']]
        return selected_points
    
    return []
    

@app.callback(
    Output('selection-output', 'children'),
    Input('selected-points', 'data')
)
def display_selected_points(selected_points):
    if selected_points:
        return f"Selected Points: {len(selected_points)} ({100 * (len(selected_points) / len(numeric_columns)):.2f}%)"
    return "No points selected."


@app.callback(
    Output('wants-higher-histogram', 'figure'),
    Input('selected-points', 'data')
)

def update_higher_histogram(selected_points):
    if selected_points:
        selected_df = df.iloc[selected_points]
    else:
        selected_df = df

    wants_higer_fig = px.histogram(
        selected_df,
        x='higher',
        title="Wants higher education",
        labels={'higher': 'Wants'},
    )

    wants_higer_fig.update_layout(
        height=histogramHeight,
        width=histogramWidth,
        title_x=0.5,
        xaxis=dict(
            title = "",
            tickvals=[0, 1],
            ticktext=["No", "Yes"],
        ),
        yaxis=dict(
            title="",
        ),
        dragmode='select'
    )

    wants_higer_fig.update_traces(
        marker=dict(color='blue')
    )

    return wants_higer_fig

@app.callback(
    Output('gender-histogram', 'figure'),
    Input('selected-points', 'data')
)

def update_gender_histogram(selected_points):
    if selected_points:
        selected_df = df.iloc[selected_points]
    else:
        selected_df = df

    studytime_fig = px.histogram(
        selected_df,
        x='sex',
        title="Gender",
        labels={'sex': 'Gender'},
    )

    studytime_fig.update_layout(
        height=histogramHeight,
        width=histogramWidth,
        title_x=0.5,  
        xaxis=dict(
            title = "",
            tickvals=[0, 1],
            ticktext=["Female", "Male"],
        ),
        yaxis=dict(
            title="",
        ),
        dragmode='select'
    )

    studytime_fig.update_traces(
        marker=dict(color='blue')
    )

    return studytime_fig


@app.callback(
    Output('parents-together-histogram', 'figure'),
    Input('selected-points', 'data')
)

def update_cohibition_histogram(selected_points):
    if selected_points:
        selected_df = df.iloc[selected_points]
    else:
        selected_df = df

    cohibition_fig = px.histogram(
        selected_df,
        x='Pstatus',
        title="Parents together",
        labels={'Pstatus': 'Parents together'},
    )

    cohibition_fig.update_layout(
        height=histogramHeight,
        width=histogramWidth,
        title_x=0.5,
        xaxis=dict(
            title = "",
            tickvals=[0, 1],
            ticktext=["Yes", "No"],
        ),
        yaxis=dict(
            title="",
        ),
        dragmode='select'
    )

    cohibition_fig.update_traces(
        marker=dict(color='blue')
    )

    return cohibition_fig

@app.callback(
    Output('grade-histogram', 'figure'),
    Input('selected-points', 'data')
)

def update_grade_histogram(selected_points):
    if selected_points:
        selected_df = df.iloc[selected_points]
    else:
        selected_df = df

    cohibition_fig = px.histogram(
        selected_df,
        x='G3',
        title="Final grade",
        labels={'Final grade': 'Final grade'},
    )

    cohibition_fig.update_layout(
        height=histogramHeight,
        width=histogramWidth * 2,
        title_x=0.5,
        xaxis=dict(
            title = "",
        ),
        yaxis=dict(
            title="",
        ),
        dragmode='select'
    )

    cohibition_fig.update_traces(
        marker=dict(color='blue')
    )

    return cohibition_fig

if __name__ == '__main__':
    app.run_server(debug=True)
