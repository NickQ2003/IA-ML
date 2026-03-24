import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import missingno as msno

# Load your dataset into a pandas DataFrame
df = pd.read_csv(r'C:\Users\nquevedo\Documents\My_Repos\IA&ML\limpieza-de-datos\dataset_clientes.csv')  # Replace 'your_dataset.csv' with your actual file path

# Display the first few rows of the dataset
print(df.head(15))

# Visualize missing data (optional)
msno.matrix(df)
print(df.head(15))
msno.heatmap(df)
print(df.head(15))

# Drop rows with missing values
df_cleaned = df.dropna()

print("Filas originales:", df.shape[0])
print("Filas sin NaN:   ", df_cleaned.shape[0])
print("Filas eliminadas:", df.shape[0] - df_cleaned.shape[0])

# Fill missing values   : mean for numeric columns, mode for categorical columns
df_filled = df.copy()
df_filled[df.select_dtypes(include=[np.number]).columns] = (
    df_filled.select_dtypes(include=[np.number])
    .fillna(df.mean(numeric_only=True))
)
df_filled[df.select_dtypes(include=['object', 'category']).columns] = (
    df_filled.select_dtypes(include=['object', 'category'])
    .fillna(df.select_dtypes(include=['object', 'category']).mode().iloc[0])
) # TypeError: can only concatenate str (not "int") to str
print(df_filled.isnull().sum())   # debe ser todo 0
print(df.mean(numeric_only=True)) 
# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print("Numéricas:   ", numerical_cols)
print("Categóricas: ", categorical_cols)

# Standardize numerical columns
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
print(df[numerical_cols].describe())
# Normalize numerical columns
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
print(df[numerical_cols].describe())


# Identify categorical columns by size (BEFORE get_dummies removes them)
large_categorical_cols = [col for col in categorical_cols if df[col].nunique() > 20]
small_categorical_cols = [col for col in categorical_cols if df[col].nunique() <= 20]

# Apply one-hot encoding to large categorical columns
df = pd.get_dummies(df, columns=large_categorical_cols, drop_first=True)


# Apply label encoding to small categorical columns
from sklearn.preprocessing import LabelEncoder
for col in small_categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])


# Identify outliers in numerical columns
outlier_threshold = 3  # Standard deviation threshold
for col in numerical_cols:
    mean = df[col].mean()
    std = df[col].std()
    df[col] = np.where(np.abs(df[col] - mean) > outlier_threshold * std, mean, df[col])


# Save the cleaned dataset
df.to_csv('cleaned_dataset.csv', index=False)
