import dash
from dash import dcc, html, Input, Output
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import plotly.express as px

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
    dcc.Graph(id='tsne-plot', style={'height': '600px', 'width': '800px'}),
])

# Callback to update the plot based on slider inputs
@app.callback(
    Output('tsne-plot', 'figure'),
    [Input('mother-education-slider', 'value'),
     Input('father-education-slider', 'value'),
     Input('final-grade-slider', 'value')]
)
def update_plot(mother_education_range, father_education_range, grade_range):
    # Filter data based on slider ranges
    filtered_df = df[
        (df['Medu'] >= mother_education_range[0]) & (df['Medu'] <= mother_education_range[1]) &
        (df['Fedu'] >= father_education_range[0]) & (df['Fedu'] <= father_education_range[1]) &
        (df['G3'] >= grade_range[0]) & (df['G3'] <= grade_range[1])
    ]

    # Create the plot with fixed color scale
    fig = px.scatter(
        filtered_df, x='tsne-1', y='tsne-2', color='G3',
        title="t-SNE Visualization",
        labels={'G3': 'Final Grade'},
        hover_data=['age', 'Medu', 'Fedu'],
        color_continuous_scale='Viridis',  # Consistent color scale
        range_color=[0, 20]  # Fixed color range from 0 to 20 (minimum to maximum grade)
    )

    # Fix the axis range with zoomed-out margins
    fig.update_layout(
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max]),
        height=600,
        width=800,
    )

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)