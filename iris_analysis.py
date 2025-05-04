# iris_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
try:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    print("First five rows of the dataset:")
    print(df.head())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    # Clean the dataset (no missing values, but keeping for consistency)
    df = df.dropna()
    print("\nDataset cleaned (if necessary).")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Task 2: Basic Data Analysis
print("\nBasic Statistics:")
print(df.describe())

print("\nMean of numerical features grouped by species:")
grouped_means = df.groupby('species').mean()
print(grouped_means)

# Observations
print("\nObservations:")
print("1. Setosa species has generally smaller petal sizes.\n"
      "2. Versicolor and Virginica have more overlap in measurements.\n"
      "3. Petal length and width vary significantly between species.")

# Task 3: Data Visualization

sns.set(style="whitegrid")

# Line Chart: Average petal length per species (mock time-series example)
plt.figure(figsize=(8, 5))
species_order = df['species'].unique()
plt.plot(species_order, grouped_means['petal length (cm)'], marker='o')
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.grid(True)
plt.tight_layout()
plt.savefig("line_chart_petal_length.png")
plt.show()

# Bar Chart: Average sepal width per species
plt.figure(figsize=(8, 5))
sns.barplot(x='species', y='sepal width (cm)', data=df, ci=None, palette="pastel")
plt.title("Average Sepal Width per Species")
plt.xlabel("Species")
plt.ylabel("Sepal Width (cm)")
plt.tight_layout()
plt.savefig("bar_chart_sepal_width.png")
plt.show()

# Histogram: Distribution of petal width
plt.figure(figsize=(8, 5))
plt.hist(df['petal width (cm)'], bins=20, color='lightblue', edgecolor='black')
plt.title("Distribution of Petal Width")
plt.xlabel("Petal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("histogram_petal_width.png")
plt.show()

# Scatter Plot: Sepal length vs Petal length
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title='Species')
plt.tight_layout()
plt.savefig("scatter_sepal_vs_petal.png")
plt.show()
