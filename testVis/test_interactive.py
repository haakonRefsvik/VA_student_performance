import dash
from dash import dcc, html, Input, Output
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import plotly.express as px

# Load and preprocess the data
df = pd.read_csv('student_data.csv')

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
print(numeric_columns.iloc[:, 0:10].head())
print(numeric_columns.iloc[:, 10:20].head())
print(numeric_columns.iloc[:, 20:30].head())
print(numeric_columns.iloc[:, 30:40].head())
print(numeric_columns.iloc[:, 40:50].head())
'''


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

data_standardized = StandardScaler().fit_transform(numeric_columns)

# Reduce dimensions with PCA (use 10 components as an example)
pca = PCA(n_components=5)
df_pca = pca.fit_transform(data_standardized)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=15, learning_rate=200, random_state=42)
tsne_results = tsne.fit_transform(df_pca)


# Add t-SNE results back to the dataframe
df['tsne-1'] = tsne_results[:, 0]
df['tsne-2'] = tsne_results[:, 1]

# Compute margins for zooming out
x_margin = (df['tsne-1'].max() - df['tsne-1'].min()) * 0.1
y_margin = (df['tsne-2'].max() - df['tsne-2'].min()) * 0.1

x_min, x_max = df['tsne-1'].min() - x_margin, df['tsne-1'].max() + x_margin
y_min, y_max = df['tsne-2'].min() - y_margin, df['tsne-2'].max() + y_margin

# Initialize Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("Interactive t-SNE Visualization"),
    html.Div([
        html.Label("Mother's Education Level"),
        dcc.RangeSlider(0, 4, 1, value=[0, 4], id='mother-education-slider',
                        marks={i: str(i) for i in range(5)}),
        html.Label("Father's Education Level"),
        dcc.RangeSlider(0, 4, 1, value=[0, 4], id='father-education-slider',
                        marks={i: str(i) for i in range(5)}),
        html.Label("Final Grade (G3)"),
        dcc.RangeSlider(0, 20, 1, value=[0, 20], id='final-grade-slider',
                        marks={i: str(i) for i in range(0, 21, 2)}),
    ], style={'margin-bottom': '20px', 'width': '50%'}),
    html.Div([
        dcc.Graph(id='tsne-plot', style={'height': '600px', 'width': '800px'}),
        dcc.Graph(id='age-bargraph', style={'height': '600px', 'width': '400px'}),
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    dcc.Store(id='selected-points', data=[]),  # Store for selected points
    html.Div(id='selection-output'),  # Div to display selected points
])

# Callback to update the t-SNE scatter plot based on slider inputs
@app.callback(
    Output('tsne-plot', 'figure'),
    [Input('mother-education-slider', 'value'),
     Input('father-education-slider', 'value'),
     Input('final-grade-slider', 'value')]
)
def update_plot(mother_education_range, father_education_range, grade_range):
    fig = px.scatter(
        df, x='tsne-1', y='tsne-2', color='G3',
        title="t-SNE Visualization",
        labels={'G3': 'Final Grade'},
        hover_data=['age', 'Medu', 'Fedu'],
        color_continuous_scale='Viridis',  # Consistent color scale
        range_color=[0, 20]  # Fixed color range from 0 to 20 (minimum to maximum grade)
    )
    fig.update_layout(
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max]),
        height=600,
        width=800,
        dragmode='select'  # Set default to box select tool
    )
    return fig

# Callback to store selected points
@app.callback(
    Output('selected-points', 'data'),
    Input('tsne-plot', 'selectedData')
)
def store_selected_points(selected_data):
    if selected_data:
        selected_points = [point['pointIndex'] for point in selected_data['points']]
        return selected_points
    return []  # Return an empty list if no points are selected

# Callback to display selected points
@app.callback(
    Output('selection-output', 'children'),
    Input('selected-points', 'data')
)
def display_selected_points(selected_points):
    if selected_points:
        return f"Selected Points: {len(selected_points)} ({100 * (len(selected_points) / len(numeric_columns)):.2f}%)"
    return "No points selected."

# Callback to update the age distribution bar graph
@app.callback(
    Output('age-bargraph', 'figure'),
    Input('selected-points', 'data')
)
def update_age_bargraph(selected_points):
    if selected_points:
        selected_df = df.iloc[selected_points]
    else:
        selected_df = df

    # Create the age distribution bar graph
    fig = px.histogram(
        selected_df, x='age', nbins=len(selected_df['age'].unique()),
        title="Age Distribution",
        labels={'age': 'Age'},
    )
    fig.update_layout(height=600, width=400)
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
