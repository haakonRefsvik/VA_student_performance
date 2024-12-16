import numpy as np
import pandas as pd
import plotly.express as px
from dash import dcc, html, Input, Output
import dash
from sklearn.discriminant_analysis import StandardScaler
from sklearn.manifold import TSNE

# Create a sample dataframe (you would load your dataset here)
df = pd.read_csv('student_data.csv')

numeric_columns = df.iloc[:, [2, 6, 7, 12,
                              13, 14, 23, 24, 25, 26, 27, 28, 29, 32]]
data_standardized = StandardScaler().fit_transform(numeric_columns)

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=10000)
tsne_results = tsne.fit_transform(data_standardized)

# Add t-SNE results back to the dataframe
df['tsne-1'] = tsne_results[:, 0]
df['tsne-2'] = tsne_results[:, 1]

# Compute margins for zooming out
x_margin = (df['tsne-1'].max() - df['tsne-1'].min()) * 0.1
y_margin = (df['tsne-2'].max() - df['tsne-2'].min()) * 0.1

x_min, x_max = df['tsne-1'].min() - x_margin, df['tsne-1'].max() + x_margin
y_min, y_max = df['tsne-2'].min() - y_margin, df['tsne-2'].max() + y_margin

# Calculate Q1, Q2 (median), Q3, and IQR
Q1 = df['G3'].quantile(0.25)
Q2 = df['G3'].median()  # Median is the 50th percentile (Q2)
Q3 = df['G3'].quantile(0.75)
IQR = Q3 - Q1

# Calculate the outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Define the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Checklist(
        id='grade-range',
        options=[
            {'label': 'Outliers', 'value': 'outliers'},
            {'label': 'Lower 25%', 'value': 'lower_25'},
            {'label': 'Lower Middle 50%', 'value': 'lower_middle'},
            {'label': 'Upper Middle 50%', 'value': 'upper_middle'},
            {'label': 'Upper 25%', 'value': 'upper_25'},
            {'label': 'Within IQR', 'value': 'within_IQR'},
        ],
        value=['within_IQR'],  # Default value
        inline=True,
        style={'margin': '10px'}
    ),
    dcc.Graph(id='tsne-plot'),
])


@app.callback(
    Output('tsne-plot', 'figure'),
    [Input('grade-range', 'value')]
)
def update_plot(selected_ranges):
    # Initialize filtered_df to be the entire dataset
    filtered_df = df

    # Filter the data based on the selected ranges
    if 'outliers' in selected_ranges:
        # Filter out values outside the outlier bounds
        filtered_df = df[(df['G3'] < lower_bound) | (df['G3'] > upper_bound)]
    elif 'lower_25' in selected_ranges:
        filtered_df = df[df['G3'] < Q1]
    elif 'lower_middle' in selected_ranges:
        filtered_df = df[(df['G3'] >= Q1) & (df['G3'] < Q2)]
    elif 'upper_middle' in selected_ranges:
        filtered_df = df[(df['G3'] >= Q2) & (df['G3'] < Q3)]
    elif 'upper_25' in selected_ranges:
        filtered_df = df[df['G3'] > Q3]
    elif 'within_IQR' in selected_ranges:
        filtered_df = df[(df['G3'] >= Q1) & (df['G3'] <= Q3)]

    # Create the scatter plot
    fig = px.scatter(
        filtered_df, x='tsne-1', y='tsne-2', color='G3',
        title="Filtered t-SNE Visualization Based on Final Grade (G3)",
        labels={'G3': 'Final Grade'},
        hover_data=['age', 'Medu', 'Fedu'],
        color_continuous_scale='Viridis',  # Consistent color scale
        # Fixed color range from 0 to 20 (minimum to maximum grade)
        range_color=[0, 20]
    )

    # Fix the axis range with zoomed-out margins
    fig.update_layout(
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max]),
        height=600,
        width=800,
    )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
