import dash
from dash import dcc, html, Input, Output
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import plotly.express as px

barFigHeight=800
barFigWidth=800
scatterFigHeight = 800
scatterFigWidth = 800

def getBarChartMaxRange(max_percentage: float) -> int:
    if max_percentage <= 20:
        return 25  # Max y-range is 25 for lower percentages
    elif max_percentage > 45:
        return 100  # Max y-range is 100 for higher percentages
    else:
        return 50  # Default max range for intermediate percentages
    
# Load and preprocess the data
df = pd.read_csv('student_data.csv')
numeric_columns = df.iloc[:, [2, 6, 7, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29, 32]]
data_standardized = StandardScaler().fit_transform(numeric_columns)

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
tsne_results = tsne.fit_transform(data_standardized)

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
        # Gender filter with RadioItems
    html.Div([
        html.Label("Select Gender"),
        dcc.RadioItems(
            id='gender-filter',
            options=[
                {'label': 'Boys', 'value': 'M'},
                {'label': 'Girls', 'value': 'F'},
                {'label': 'Both', 'value': 'Both'}
            ],
            value='Both',  # Default: show both boys and girls
            labelStyle={'display': 'block'},  # Make the labels stack vertically
        ),
    ], style={'margin-bottom': '20px'}),

    html.Div([
        html.Label("Mother's Education Level"),
        dcc.RangeSlider(0, 4, 1, value=[0, 4], id='mother-education-slider',
                        marks={i: str(i) for i in range(5)}, updatemode="drag"),
        html.Label("Father's Education Level"),
        dcc.RangeSlider(0, 4, 1, value=[0, 4], id='father-education-slider',
                        marks={i: str(i) for i in range(5)}, updatemode="drag"),
    ], style={'margin-bottom': '20px', 'width': '300px'}),
    html.Div([
        dcc.Graph(id='tsne-plot', style={'height': '600px', 'width': '800px'}),
        dcc.Graph(id='grade-distribution', style={'height': '600', 'width': '800px'}),

    ], style={'display': 'flex'})

])

# Callback to update the scatter plot and bar chart
@app.callback(
    [Output('tsne-plot', 'figure'),
     Output('grade-distribution', 'figure')],
    [Input('mother-education-slider', 'value'),
     Input('father-education-slider', 'value'),
     Input('gender-filter', 'value')]  # Add gender filter input
)

def update_plots(mother_education_range, father_education_range, selected_gender):

    # Filter data based on slider ranges
    # Filter data based on slider ranges and selected gender
    if selected_gender == 'Both':
        filtered_df = df[
            (df['Medu'] >= mother_education_range[0]) & (df['Medu'] <= mother_education_range[1]) &
            (df['Fedu'] >= father_education_range[0]) & (df['Fedu'] <= father_education_range[1])
        ]
    else:
        filtered_df = df[
            (df['Medu'] >= mother_education_range[0]) & (df['Medu'] <= mother_education_range[1]) &
            (df['Fedu'] >= father_education_range[0]) & (df['Fedu'] <= father_education_range[1]) &
            (df['sex'] == selected_gender)  # Filter by gender
        ]
    # Handle empty dataset
    if filtered_df.empty:
        # Create an empty scatter plot
        scatter_fig = px.scatter(
            title="t-SNE Visualization (No Data Available)"
        )
        scatter_fig.update_layout(
            xaxis=dict(range=[x_min, x_max], title="tsne-1"),
            yaxis=dict(range=[y_min, y_max], title="tsne-2"),
            height=scatterFigHeight,
            width=scatterFigWidth,
            coloraxis_showscale=False  # Hide color scale for empty plot
        )
        scatter_fig.update_layout(height=scatterFigHeight, width=scatterFigWidth)

        # Create an empty bar chart
        bar_fig = px.bar(
            title="Grade Distribution (No Data Available)",
            labels={'x': 'Grade', 'y': 'Percentage of Students'}
        )
        bar_fig.update_layout(height=barFigHeight, width=barFigWidth)

        return scatter_fig, bar_fig
    
    percentage_displayed = (len(filtered_df) / len(df)) * 100

    # Create the scatter plot with fixed color scale
    scatter_fig = px.scatter(
        filtered_df, x='tsne-1', y='tsne-2', color='G3',
        title=f"t-SNE Visualization ({percentage_displayed:.2f}% of Data Shown)",
        labels={'G3': 'Final Grade'},
        hover_data=['age', 'Medu', 'Fedu'],
        color_continuous_scale='Viridis',
        range_color=[0, 20]
    )
    scatter_fig.update_layout(
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max]),
        height=scatterFigHeight,
        width=scatterFigWidth,
    )

    # Create the bar chart showing grade distribution as percentages
    grade_distribution = filtered_df['G3'].value_counts(normalize=True).sort_index() * 100  # Normalize to percentage
    max_percentage = grade_distribution.max()
    y_barchart_max_range = getBarChartMaxRange(max_percentage)

    bar_fig = px.bar(
        x=grade_distribution.values, y=grade_distribution.index,
        labels={'y': 'Grade', 'x': 'Percentage of Students'},
        title="Grade Distribution (Percentage)",
        color=grade_distribution.index,  # Color bars by percentage values
        color_continuous_scale='Viridis',  # Apply Viridis color scale
        orientation='h',
        range_color=[0, 20],
    )

    # Set static x-axis and y-axis ranges for the bar chart
    bar_fig.update_layout(
        coloraxis_showscale=False,  # Hides the color scale legend
        height=barFigHeight,
        width=barFigWidth,

        xaxis=dict(
            title='Percentage of Students',
            range=[0, 50],  # Fix y-axis range from 0 to 100%
        ),
        yaxis=dict(
            title='Grade',
            range=[0, 20],  # Fix y-axis range from 0 to 100%
            tickvals=list(range(0, 21, 1))  # Show each grade from 0 to 20
        ),

    )

    return scatter_fig, bar_fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
