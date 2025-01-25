import pandas as pd
import plotly.express as px

# Assuming 'df' is your DataFrame containing the dataset
# Load and preprocess the data
df = pd.read_csv('data/student_data.csv')

fig = px.parallel_coordinates(
    df,
    dimensions=[
        'studytime', 
        'famsup',
        'internet',
        'failures', 
        'romantic', 
        'famrel', 
        'freetime', 
        'goout', 
        'Dalc',
        'absences', 
        'traveltime', 
        'G3'
    ],
    labels={
        'studytime': "Study Time",
        'failures': "Past Failures",
        'famsup': "Family Support",
        'internet': "Internet Access",
        'traveltime': "Travel Time",
        'romantic': "Romantic Relationship",
        'famrel': "Family Relationship",
        'freetime': "Free Time",
        'goout': "Going Out",
        'health': "Health Status",
        'Dalc': "Workday Alcohol Consumption",
        'absences': "Absences",
        'G3': "Final Grade"
    },
    title="Parallel Coordinates Plot with Balanced Dimensions"
)

# Update opacity and layout
fig.update_layout(
    height=300,
    width=800,
    font=dict(size=10),
    coloraxis_colorbar=dict(
        title="Final Grade",
        tickvals=[0, 5, 10, 15, 20],
        ticktext=["0", "5", "10", "15", "20"]
    )
)

fig.show()
