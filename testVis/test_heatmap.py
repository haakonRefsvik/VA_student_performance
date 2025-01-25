import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Example: load the dataset (replace with your actual DataFrame)
df = pd.read_csv("data/student_data.csv")

# Step 1: Encode categorical variables to numeric values (if necessary)
# For binary categorical columns, you can use Label Encoding
df['sex'] = df['sex'].map({'F': 0, 'M': 1})
df['school'] = df['school'].map({'GP': 0, 'MS': 1})
df['address'] = df['address'].map({'U': 0, 'R': 1})
df['Pstatus'] = df['Pstatus'].map({'T': 0, 'A': 1})
df['schoolsup'] = df['schoolsup'].map({'yes': 1, 'no': 0})
df['famsup'] = df['famsup'].map({'yes': 1, 'no': 0})
df['famsize'] = df['famsize'].map({'GT3': 1, 'LE3': 0})
df['activities'] = df['activities'].map({'yes': 1, 'no': 0})
df['higher'] = df['higher'].map({'yes': 1, 'no': 0})
df['internet'] = df['internet'].map({'yes': 1, 'no': 0})
df['romantic'] = df['romantic'].map({'yes': 1, 'no': 0})

# Step 2: Select relevant numeric and encoded columns
columns = ['address','Pstatus','famsize', 'age', 'sex', 'studytime', 'failures', 'Pstatus', 'Medu', 'Fedu', 
           'schoolsup', 'famsup', 'activities', 'higher', 'internet', 'romantic', 
           'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G3']

# Step 3: Compute correlation matrix
corr_matrix = df[columns].corr()

# Step 4: Plot heatmap using Plotly
fig = px.imshow(corr_matrix,
                title="Correlation Heatmap of Attributes vs Final Grade (G3)",
                labels={'x': 'Attributes', 'y': 'Attributes'},
                color_continuous_scale='Viridis')
fig.show()

# Optional: Use seaborn and matplotlib for a static version of the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='viridis', cbar=True, square=True)
plt.title("Correlation Heatmap of Attributes vs Final Grade (G3)")
plt.show()
