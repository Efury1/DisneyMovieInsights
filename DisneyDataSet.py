import kagglehub
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import os

# Print current working directory 
print("Current working directory:", os.getcwd())

# Download the latest version of the dataset
print("=" * 40)
print("\n Downloading dataset...")
print("=" * 40)
path = kagglehub.dataset_download("prateekmaj21/disney-movies")
print("Path to dataset files:", path)

# Save to a subfolder name output" in the current directory
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)
output_file_path = f"{output_folder}/disney_movies.csv"

# Define the file path to the dataset file and load it
file_path = f"{path}/disney_movies.csv"
print("\n dataset")
df = pd.read_csv(file_path)

df.to_csv(output_file_path, index=False)
print(f"Dataset saved to: {output_file_path}")

# Preview the dataset
print("=" * 40)
print("\n Preview Dataset")
print(df.head())
print("=" * 40)
# Explore and Prepare the Dataset

# print dataset structure
print("\n Data Structure")
print(df.info())
print("=" * 40)

# Check for missing values 
print("\n Check for missing values")
print(df.isnull().sum())
print("=" * 40)

# Get basic statistics 
print("\n Basic statistics")
print(df.describe())
print("=" * 40)

# Clean data
df['mpaa_rating'] = df['mpaa_rating'].fillna("Unknown")

# Convert release_date to datetime
df['release_date'] = pd.to_datetime(df['release_date'])

# Ensure numerical columns are properly formatted
df['total gross'] = pd.to_numeric(df['total_gross'], errors='coerce')
df['inflation_adjustred_gross'] = pd.to_numeric(df['inflation_adjusted_gross'])
print("=" * 40)

# Encode Categorical Variables
# Label encode MPAA rating
label_encoder = LabelEncoder()
df['mpaa_rating_encoding'] = label_encoder.fit_transform(df['mpaa_rating'])
print(df[['mpaa_rating', 'mpaa_rating_encoding']].head())
print("=" * 40)

# One-Hot Encoding for Non-ORdinal Data
df = pd.get_dummies(df, columns=['genre'], prefix='genre')
print(df.head())

# Decision tree classifier
# Prepare features and target
features = df[['total_gross', 'mpaa_rating_encoding']]
target = df['genre_Comedy']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Train the decision tree
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("=" * 40)