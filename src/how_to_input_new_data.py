import pickle
import pandas as pd
import preprocess_data as prep_data

# Exemple de nouvelle donnée
# Example 1 = 1
# new_data = {
#     'age': [63],
#     'sex': ['Male'],
#     'dataset': ['Cleveland'],
#     'cp': ['typical angina'],
#     'fbs': [True],
#     'trestbps': [145],
#     'chol': [233],
#     'restecg': ['lv hypertrophy'],
#     'thalch': [150],
#     'exang': [False],
#     'oldpeak': [2.3],
#     'slope': ['downsloping']
# }

# Example 2 = 0
new_data = {
    'age': [67],
    'sex': ['Male'],
    'dataset': ['Cleveland'],
    'cp': ['asymptomatic'],
    'fbs': [False],
    'trestbps': [160],
    'chol': [286],
    'restecg': ['lv hypertrophy'],
    'thalch': [108],
    'exang': [True],
    'oldpeak': [1.5],
    'slope': ['flat'],
}



# Charger le modèle entraîné
with open('../models/randomforest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Charger le modèle PCA et les outils de prétraitement
with open('../tools/num_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
with open('../tools/cat_encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)
with open('../tools/pca.pkl', 'rb') as file:
    pca = pickle.load(file)

# Convertir en DataFrame
new_data_df = pd.DataFrame(new_data)
numeric_cols = new_data_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = new_data_df.select_dtypes(include=['object', 'bool']).columns.tolist()

# Appliquer le scaler aux colonnes numériques
new_data_numeric_scaled = scaler.transform(new_data_df[numeric_cols])
new_data_numeric_scaled_df = pd.DataFrame(new_data_numeric_scaled, columns=numeric_cols)

# Appliquer l'encoder aux colonnes catégorielles
new_data_categorical_encoded = encoder.transform(new_data_df[categorical_cols])
new_data_categorical_encoded_df = pd.DataFrame(new_data_categorical_encoded, columns=encoder.get_feature_names_out(categorical_cols))

# Combiner les données numériques et catégorielles transformées
new_data_transformed_df = pd.concat([new_data_numeric_scaled_df, new_data_categorical_encoded_df], axis=1)
# Appliquer PCA
new_data_pca_transformed = pca.transform(new_data_transformed_df)

# Faire la prédiction
prediction = model.predict(new_data_pca_transformed)

print(f"Prédiction de la variable cible 'num': {prediction[0]}")
