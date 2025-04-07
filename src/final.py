import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder

# Modifier les limitations d'affichage par défaut de Pandas pour faciliter l'exploration des données.
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 25)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

# Load the data
def get_data():
    data = pd.read_csv('../data/heart_disease_uci.csv')
    data = data.drop(['id'], axis=1)

    return data


def exploration(data, save_graphs, display_graphs):
    # show histograms
    data.hist(bins=50, figsize=(15, 10))
    if save_graphs: plt.savefig('../output/histograms.png')
    if display_graphs: plt.show()

    # show boxplot of numeric data
    data.select_dtypes(include=['int64', 'float64']).plot(kind='box', subplots=True, layout=(4, 6), figsize=(15, 10))
    if save_graphs: plt.savefig('../output/boxplots.png')
    if display_graphs: plt.show()

    # print the info of the data<
    print(data.info())
    # print number of missing values of each column
    print(data.isnull().sum())

    # return the description of the data
    print(data.describe(include='all'))


def preprocess_data_imputation(data):

    # Compter le nombre de valeurs manquantes par colonne
    missing_values_count = data.isnull().sum()

    # Identifier les colonnes avec un nombre de valeurs manquantes supérieur à 60% du nombre total de lignes
    columns_to_remove = missing_values_count[missing_values_count > 0.6 * data.shape[0]].index

    # Supprimer les colonnes identifiées
    data = data.drop(columns=columns_to_remove)

    # Séparer les colonnes numériques par type (int et float)
    int_columns = data.select_dtypes(include=['int64']).columns
    float_columns = data.select_dtypes(include=['float64']).columns
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns

    # Imputer les valeurs manquantes des colonnes entières par la médiane et arrondir
    for column in int_columns:
        if data[column].isnull().sum() > 0:
            # Utiliser la médiane et arrondir à l'entier le plus proche
            median_value = int(round(data[column].median()))
            data[column] = data[column].fillna(median_value).astype('int64')

    # Imputer les valeurs manquantes des colonnes flottantes par la moyenne
    for column in float_columns:
        if data[column].isnull().sum() > 0:
            data[column] = data[column].fillna(data[column].mean())

    # Traiter les colonnes catégorielles si elles existent
    if len(categorical_columns) > 0:

        # Encoder les variables catégorielles
        encoder = OrdinalEncoder()
        categorical_data = data[categorical_columns].copy()
        categorical_encoded = pd.DataFrame(
            encoder.fit_transform(categorical_data),
            columns=categorical_columns,
            index=data.index
        )

        # Appliquer KNN imputer sur les données encodées
        imputer = KNNImputer(n_neighbors=5)
        categorical_imputed = pd.DataFrame(
            imputer.fit_transform(categorical_encoded),
            columns=categorical_columns,
            index=data.index
        )

        # Arrondir les valeurs imputées à l'entier le plus proche
        categorical_imputed = categorical_imputed.round().astype(int)

        # Inverser l'encodage pour retrouver les catégories originales
        categorical_decoded = pd.DataFrame(
            encoder.inverse_transform(categorical_imputed),
            columns=categorical_columns,
            index=data.index
        )

        # Remplacer les colonnes catégorielles d'origine par les versions encodées et imputées
        data = data.drop(columns=categorical_columns)
        data = pd.concat([data, categorical_decoded], axis=1)

    return data


def handle_outliers_with_iqr(data, factor=1.5):

    # Sélectionner toutes les colonnes numériques
    columns = data.select_dtypes(include=['int64', 'float64']).columns

    # Traiter chaque colonne spécifiée
    for column in columns:

        # Calcul des quartiles
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1

        # Définir les limites
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        # Remplacer les outliers
        original_type = data[column].dtype

        # Gérer les valeurs aberrantes en les remplaçant par les limites
        data[column] = np.where(data[column] < lower_bound, lower_bound, data[column])
        data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])

        # Préserver le type de données original (pour les entiers)
        if original_type == 'int64':
            data[column] = data[column].round().astype('int64')

        return data

# def preprocess_data_encode_and_scale(data):
#     # Séparer les colonnes numériques par type (int et float)
#     num_columns = data.select_dtypes(include=['int64', 'float64']).columns
#     categorical_columns = data.select_dtypes(include=['object', 'category']).columns
#
#     # Encoder les variables catégorielles
#     encoder = LabelEncoder()
#     categorical_data = data[categorical_columns].copy()
#     categorical_encoded = pd.DataFrame(
#         encoder.fit_transform(categorical_data),
#         columns=categorical_columns,
#         index=categorical_data.index
#     )
#
#     # Normaliser les colonnes numériques
#     scaler = StandardScaler()
#     numeric_data = data[num_columns].copy()
#     numeric_scaled = pd.DataFrame(
#         scaler.fit_transform(numeric_data),
#         columns=num_columns,
#         index=numeric_data.index
#     )
#
#     # Combiner les données encodées et normalisées avec les colonnes non modifiées
#     data = pd.concat([numeric_scaled, categorical_encoded], axis=1)
#
#     return data

def preprocess_data_encode_and_scale(data, target_column='num'):
    # # Séparation X et y
    # X = df.drop(columns=[target_column])
    # y = df[target_column]

    # Identifier types de colonnes
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'bool']).columns.tolist()

    # --- Standardisation des colonnes numériques ---
    scaler = StandardScaler()
    X_numeric = pd.DataFrame(scaler.fit_transform(data[numeric_cols]), columns=numeric_cols)

    # --- Encodage des colonnes catégorielles ---
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    X_categorical = pd.DataFrame(
        encoder.fit_transform(data[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    # --- Fusionner ---
    data = pd.concat([X_numeric.reset_index(drop=True), X_categorical.reset_index(drop=True)], axis=1)

    return data

def main():
    data = get_data()
    data = preprocess_data_imputation(data)
    data = handle_outliers_with_iqr(data)
    data = preprocess_data_encode_and_scale(data)

    # print(exploration(data, save_graphs=False, display_graphs=False))
    print(data)


if __name__ == "__main__":
    main()