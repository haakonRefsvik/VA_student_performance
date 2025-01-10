import dash
from dash import dcc, html, Input, Output
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, OneHotEncoder, StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # Import make_subplots

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
    [Input('medu-boxplot', 'selectedData'),
     Input('fedu-boxplot', 'selectedData'),
     Input('studytime-boxplot', 'selectedData')
    ]
)
def update_tsne_plot(medu_selected, fedu_selected, studytime_selected):
    # Determine which points are selected based on the boxplot selections
    selected_indices = set(range(len(df)))

    # Process selections from the Medu histogram
    if medu_selected and 'points' in medu_selected:
        medu_indices = set()
        for point in medu_selected['points']:
            medu_indices.update(point['pointNumbers'])  # Use 'pointNumbers' for histogram bins

        selected_indices.intersection_update(medu_indices)

    if fedu_selected and 'points' in fedu_selected:
        fedu_indices = set()
        for point in fedu_selected['points']:
            fedu_indices.update(point['pointNumbers'])  # Use 'pointNumbers' for histogram bins
        selected_indices.intersection_update(fedu_indices)

    if studytime_selected and 'points' in studytime_selected:
        studytime_indices = set()
        for point in studytime_selected['points']:
            studytime_indices.update(point['pointNumbers'])  # Use 'pointNumbers' for histogram bins
        selected_indices.intersection_update(studytime_indices)

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
    attributes = ['Medu', 'Fedu', 'failures', 'studytime', 'traveltime', 'Walc', 'Dalc', 'health', 'famrel', 'goout']
    
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
        'famrel': 'quality of family relationships',
        'goout': 'going out with friends'
    }
    
    # Create a binning transformer to discretize values into 5 bins
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

    # Create a dataframe to hold the counts of students per bin per attribute
    heatmap_data = pd.DataFrame(columns=attributes, index=[0, 1, 2, 3, 4])
    text_data = pd.DataFrame(columns=attributes, index=[0, 1, 2, 3, 4])  # For text labels inside cells

    # Bin and count for each attribute
    for attribute in attributes:
        # Reshape the data to fit the discretizer and transform it into bins
        binned_data = discretizer.fit_transform(selected_df[[attribute]])

        # Count how many students fall into each bin for the current attribute
        bin_counts = pd.Series(binned_data.flatten()).value_counts().sort_index()

        # Reindex to ensure all bins (0-4) are present, filling missing bins with 0
        bin_counts = bin_counts.reindex([0, 1, 2, 3, 4]).fillna(0).astype(int)

        # Fill the heatmap dataframe with the bin counts
        heatmap_data[attribute] = bin_counts

        # Fill the text dataframe with the bin counts as text
        text_data[attribute] = bin_counts.astype(str)  # Convert counts to string for text

    # Create the heatmap figure
    fig = go.Figure(go.Heatmap(
        z=heatmap_data.T.values,  # Values for the heatmap (transposed to fit the layout)
        x=heatmap_data.index,  # Binned values (0, 1, 2, 3, 4)
        y=[custom_titles[attr] for attr in heatmap_data.columns],  # Custom titles for attributes
        colorscale='Inferno',  # Color scale
        colorbar=dict(title='Number of Students'),
        showscale=True,
        text=text_data.T.values,  # The text for each cell
        texttemplate='%{text}',  # Use the text inside the cells
        hoverinfo='text',  # Show the text on hover
    ))

    # Update layout
    fig.update_layout(
        title="Heatmap of Students' Attribute Bins with Counts",
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
        dcc.Graph(id='medu-boxplot', style={'height': '150px', 'width': '240px'}),
        dcc.Graph(id='fedu-boxplot', style={'height': '150px', 'width': '240px'}),
        dcc.Graph(id='studytime-boxplot', style={'height': '150px', 'width': '240px'}),
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
    Output('studytime-boxplot', 'figure'),
    Input('selected-points', 'data')
)

def update_studytime_histogram(selected_points):
    # Filter the dataframe based on selected points if they exist
    if selected_points:
        selected_df = df.iloc[selected_points]
    else:
        selected_df = df

    # Create the histogram for studytime
    studytime_fig = px.histogram(
        selected_df,
        x='studytime',
        title="Weekly studytime",
        labels={'studytime': 'Study Time'},
    )

    # Update layout to set specific tick values and labels
    studytime_fig.update_layout(
        height=histogramHeight,
        width=histogramWidth,
        title_x=0.5,  # Center the title
        xaxis=dict(
            title = "",
            tickvals=[1, 2, 3, 4],
            ticktext=["<2 hours", "2-5 hours", "5-10 hours", ">10 hours"],
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
    Output('medu-boxplot', 'figure'),
    Output('fedu-boxplot', 'figure'),
    Input('selected-points', 'data')
)
def update_education_histograms(selected_points):
    # Filter the dataframe based on selected points if they exist
    if selected_points:
        selected_df = df.iloc[selected_points]
    else:
        selected_df = df

    # Create the medu histogram
    medu_fig = px.histogram(
        selected_df,
        x='Medu',
        title="Mother's Education",
        labels={'Medu': "Mother's Education"},
        nbins=5,  # To align with the 5 education levels
    )
    medu_fig.update_layout(
        height=histogramHeight,
        width=histogramWidth,
        title_x=0.5,  # Center the title
        title_font=dict(size=histogramTitleFontSize),  # Set the title font size
        xaxis=dict(
            tickvals=[0, 1, 2, 3, 4],
            ticktext=["None", "Primary", "5th-9th", "Secondary", "Higher"],
            title="",
            showticklabels=False
        ),
        yaxis=dict(
            title="",
        ),
        showlegend=False,  # Remove legend if not needed
        dragmode='select'
    )
    medu_fig.update_traces(
        marker=dict(color='blue')
    )

    # Create the fedu histogram
    fedu_fig = px.histogram(
        selected_df,
        x='Fedu',
        title="Father's Education",
        labels={'Fedu': "Father's Education"},
        nbins=5,  # To align with the 5 education levels
    )
    fedu_fig.update_layout(
        height=histogramHeight,
        width=histogramWidth,
        title_x=0.5,  # Center the title
        title_font=dict(size=histogramTitleFontSize),  # Set the title font size
        xaxis=dict(
            tickvals=[0, 1, 2, 3, 4],
            ticktext=["None", "Primary", "5th-9th", "Secondary", "Higher"],
            title="",
            showticklabels=False
        ),
        yaxis=dict(
            title="",
        ),
        showlegend=False,  # Remove legend if not needed
        dragmode='select'
    )
    fedu_fig.update_traces(
        marker=dict(color='blue')
    )

    return medu_fig, fedu_fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
