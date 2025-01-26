import dash
from dash import dcc, html, Input, Output
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, OneHotEncoder, StandardScaler
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
    # Define the attributes and their descriptions
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

@app.callback(
    Output('tsne-plot', 'figure'),
    [Input('studytime-boxplot', 'selectedData'),
     Input('wants-higher-boxplot', 'selectedData'),
     Input('parents-together-boxplot', 'selectedData')
    ]
)
def update_tsne_plot(studytime_selected, wants_higher_selected, parents_together_selected):
    # Determine which points are selected based on the boxplot selections
    selected_indices = set(range(len(df)))

    if studytime_selected and 'points' in studytime_selected:
        studytime_indices = set()
        for point in studytime_selected['points']:
            studytime_indices.update(point['pointNumbers'])  # Use 'pointNumbers' for histogram bins
        selected_indices.intersection_update(studytime_indices)

    if wants_higher_selected and 'points' in wants_higher_selected:
        higher_indices = set()
        for point in wants_higher_selected['points']:
            higher_indices.update(point['pointNumbers'])  # Use 'pointNumbers' for histogram bins
        selected_indices.intersection_update(higher_indices)

    if parents_together_selected and 'points' in parents_together_selected:
        parents_together_indices = set()
        for point in parents_together_selected['points']:
            parents_together_indices.update(point['pointNumbers'])  # Use 'pointNumbers' for histogram bins
        selected_indices.intersection_update(parents_together_indices)

    # If no points are selected, show the full data
    if not selected_indices:
        selected_df = df
    else:
        # Filter the dataframe to only show selected points
        selected_df = df.iloc[list(selected_indices)]
    
    # Create a scatter plot based on the selected points
    fig = px.scatter(
        selected_df, x='tsne-1', y='tsne-2', color='G3',
        title="t-SNE Visualization",
        labels={'G3': 'Final Grade'},
        hover_data={'tsne-1': False, 'tsne-2': False,  # Exclude tsne-1 and tsne-2
                    'age': True, 'Medu': True, 'Fedu': True},  # Show only age, Medu, Fedu
        color_continuous_scale='Viridis',  # Consistent color scale
        range_color=[0, 20]  # Fixed color range from 0 to 20 (minimum to maximum grade)
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

# Function to create the heatmap
@app.callback(
    Output('heatmap', 'figure'),
    Input('selected-points', 'data')
)
def create_heatmap(selected_points):
    # If selected points are available, filter the dataframe accordingly
    if selected_points:
        selected_df = df.iloc[selected_points]
    else:
        selected_df = df  # Show full dataset if no points are selected

    # Define the attributes to include in the heatmap
    attributes = ['Medu', 'Fedu', 'failures', 'studytime', 'traveltime', 'Walc', 'Dalc', 'health', 'famrel', 'goout', 'freetime']
    
    # Define custom titles for the attributes
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
    
    # Define the number of bins for each attribute
    num_bins = {
        'Medu': 5, 'Fedu': 5, 'failures': 4, 'studytime': 4, 'traveltime': 4,
        'Walc': 5, 'Dalc': 5, 'health': 5, 'famrel': 5, 'goout': 5, 'freetime': 5
    }
    
    # Create dataframes to hold the heatmap data and text
    heatmap_data = pd.DataFrame(columns=attributes, index=[0, 1, 2, 3, 4])  # Maximum 5 bins
    text_data = pd.DataFrame(columns=attributes, index=[0, 1, 2, 3, 4])  # For text labels inside cells
    hover_text = pd.DataFrame(columns=attributes, index=[0, 1, 2, 3, 4])  # For hover text

    # Bin and count for each attribute
    for attribute in attributes:
        # Get the number of bins for the current attribute
        bins = num_bins[attribute]
        
        # Create a binning transformer with the appropriate number of bins
        discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')

        # Reshape the data to fit the discretizer and transform it into bins
        try:
            binned_data = discretizer.fit_transform(selected_df[[attribute]])
        except ValueError:
            # Handle attributes with no variation or insufficient data
            binned_data = np.zeros((selected_df.shape[0], 1))

        # Count how many students fall into each bin for the current attribute
        bin_counts = pd.Series(binned_data.flatten()).value_counts().sort_index()

        # Reindex to ensure all bins (0 to bins-1) are present, filling missing bins with 0
        bin_counts = bin_counts.reindex(range(bins)).fillna(0).astype(int)

        # Normalize the bin counts for each attribute (relative scale)
        total_count = bin_counts.sum()
        if total_count > 0:
            bin_counts_normalized = bin_counts / total_count
        else:
            bin_counts_normalized = bin_counts  # No normalization if no data is available

        # Fill the heatmap dataframe with the normalized bin counts
        heatmap_data.loc[0:bins - 1, attribute] = bin_counts_normalized
        # Display normalized bin counts with 1 digit after the decimal point
        
        text_data.loc[0:bins - 1, attribute] = bin_counts_normalized.apply(lambda x: f"{x:.1f}")

        # Fill the hover text with custom explanations
        hover_text.loc[0:bins - 1, attribute] = [
            f"{explain_attribute(attribute, bin_idx)}: {float(distr) * 100:.1f}% of students"
            for bin_idx, distr in enumerate(bin_counts_normalized)
        ]

    # Replace any remaining NaN values in heatmap_data with 0
    heatmap_data.fillna(0, inplace=True)

    # Create the heatmap figure
    fig = go.Figure(go.Heatmap(
        z=heatmap_data.T.values,  # Values for the heatmap (transposed to fit the layout)
        x=heatmap_data.index,  # Binned values (0, 1, 2, 3, 4)
        y=[custom_titles[attr] for attr in heatmap_data.columns],  # Custom titles for attributes
        colorscale='Inferno',  # Color scale
        zmin=0,  # Ensure the color scale starts at 0
        zmax=1,  # Ensure the color scale is capped at 1
        colorbar=dict(title='Relative Distribution'),
        showscale=True,

        texttemplate='%{text}',  # Use the text inside the cells
        hoverinfo='text',  # Show the text on hover
        hovertext=hover_text.T.values,  # Custom hover text
    ))

    # Update layout
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
        dcc.Graph(id='studytime-boxplot', style={'height': '150px', 'width': '240px'}),
        dcc.Graph(id='wants-higher-boxplot', style={'height': '150px', 'width': '240px'}),
        dcc.Graph(id='parents-together-boxplot', style={'height': '150px', 'width': '240px'}),
    ], style={'display': 'flex', 'flex-direction': 'row', 'height': '350px'}),
    html.Div([
        dcc.Graph(id='tsne-plot', 
                    figure=update_tsne_plot([], [], []), 
                    style={'height': '600px', 'width': '800px'},
                    config={'displayModeBar': True},  # Enable tools for selection
                  ),
        dcc.Graph(id='heatmap', style={'height': '600px', 'width': '600px'}),  # heatmap
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    dcc.Store(id='selected-points', data=[]),  # Store for selected points
    html.Div(id='selection-output'),  # Div to display selected points
])

##  ------------------------------------------------------------------------------

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


@app.callback(
    Output('wants-higher-boxplot', 'figure'),
    Input('selected-points', 'data')
)

def update_higher_histogram(selected_points):
    # Filter the dataframe based on selected points if they exist
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

    # Update layout to set specific tick values and labels
    wants_higer_fig.update_layout(
        height=histogramHeight,
        width=histogramWidth,
        title_x=0.5,  # Center the title
        xaxis=dict(
            title = "",
            tickvals=[0, 1],
            ticktext=["No", "Yes"],
            showticklabels=False
        ),
        yaxis=dict(
            title="",
        ),
        dragmode='select'
    )

    # Customize the appearance of the histogram bars
    wants_higer_fig.update_traces(
        marker=dict(color='blue')
    )

    return wants_higer_fig

@app.callback(
    Output('studytime-boxplot', 'figure'),
    Input('selected-points', 'data')
)

def update_gender_histogram(selected_points):
    # Filter the dataframe based on selected points if they exist
    if selected_points:
        selected_df = df.iloc[selected_points]
    else:
        selected_df = df

    # Create the histogram for studytime
    studytime_fig = px.histogram(
        selected_df,
        x='sex',
        title="Gender",
        labels={'sex': 'Gender'},
    )

    # Update layout to set specific tick values and labels
    studytime_fig.update_layout(
        height=histogramHeight,
        width=histogramWidth,
        title_x=0.5,  # Center the title
        xaxis=dict(
            title = "",
            tickvals=[0, 1],
            ticktext=["Female", "Male"],
            showticklabels=False
        ),
        yaxis=dict(
            title="",
        ),
        dragmode='select'
    )

    # Customize the appearance of the histogram bars
    studytime_fig.update_traces(
        marker=dict(color='blue')
    )

    return studytime_fig

@app.callback(
    Output('parents-together-boxplot', 'figure'),
    Input('selected-points', 'data')
)

def update_cohibition_histogram(selected_points):
    # Filter the dataframe based on selected points if they exist
    if selected_points:
        selected_df = df.iloc[selected_points]
    else:
        selected_df = df

    # Create the histogram for studytime
    cohibition_fig = px.histogram(
        selected_df,
        x='Pstatus',
        title="Parents together",
        labels={'Pstatus': 'Parents together'},
    )

    # Update layout to set specific tick values and labels
    cohibition_fig.update_layout(
        height=histogramHeight,
        width=histogramWidth,
        title_x=0.5,  # Center the title
        xaxis=dict(
            title = "",
            tickvals=[0, 1],
            ticktext=["True", "False"],
            showticklabels=False
        ),
        yaxis=dict(
            title="",
        ),
        dragmode='select'
    )

    # Customize the appearance of the histogram bars
    cohibition_fig.update_traces(
        marker=dict(color='blue')
    )

    return cohibition_fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
