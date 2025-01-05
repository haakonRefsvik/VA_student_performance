import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import plotly.express as px

scatterFigHeight = 800
scatterFigWidth = 800

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
    dcc.Store(id='stored-selection', data=[]),  # Persistent store for selected points
    html.Div([
        dcc.Graph(id='tsne-plot', style={'height': '600px', 'width': '800px'}),
    ])
])

# Callback to store the selected data
@app.callback(
    Output('stored-selection', 'data'),
    Input('tsne-plot', 'selectedData'),
    State('stored-selection', 'data')
)
def store_selection(selected_data, stored_data):
    if selected_data:
        selected_points = [point['pointIndex'] for point in selected_data['points']]
        stored_data = list(set(stored_data + selected_points))  # Ensure no duplicates
        return stored_data
    return stored_data  # If no new selection, retain the previous selection

# Callback to update the scatter plot
@app.callback(
    Output('tsne-plot', 'figure'),
    [Input('stored-selection', 'data')]
)
def update_scatterplot(selected_points):
    filtered_df = df.copy()
    # Handle selection (lasso tool)
    if selected_points:
        filtered_df['opacity'] = filtered_df.index.map(
            lambda x: 1.0 if x in selected_points else 0.2
        )
    else:
        filtered_df['opacity'] = 1.0

    # Create the scatter plot with fixed color scale and varying opacity
    scatter_fig = px.scatter(
        filtered_df, x='tsne-1', y='tsne-2', color='G3',
        title="t-SNE Visualization",
        labels={'G3': 'Final Grade'},
        hover_data=['age', 'Medu', 'Fedu'],
        color_continuous_scale='Viridis',
        range_color=[0, 20]
    )
    scatter_fig.update_traces(marker=dict(opacity=filtered_df['opacity']))
    scatter_fig.update_layout(
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max]),
        height=scatterFigHeight,
        width=scatterFigWidth,
        dragmode='select'  # Set default to box select tool
    )

    return scatter_fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
