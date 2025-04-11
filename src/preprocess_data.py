import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

def create_binary_target(data, target_column='num'):
    """
    Fonction utilitaire qui permet de convertir la variable cible en binaire
    """
    data[target_column] = data[target_column].apply(lambda x: 1 if x > 0 else 0)
    print(data[target_column].value_counts())

    return data

def exploration(data, save_graphs, display_graphs):
    """
    Fonction utilitaire qui permet d'explorer les données
    """
    # Afficher les histogrammes des toutes les variables
    data.hist(bins=50, figsize=(15, 10))
    if save_graphs: plt.savefig('../output/histograms.png')
    if display_graphs: plt.show()

    # Afficher les boîtes à moustaches pour toutes les variables numériques
    # Cela permet de visualiser les valeurs aberrantes
    data.select_dtypes(include=['int64', 'float64']).plot(kind='box', subplots=True, layout=(4, 6), figsize=(15, 10))
    if save_graphs: plt.savefig('../output/boxplots.png')
    if display_graphs: plt.show()

    # Afficher les informations sur les données
    # Permet de voir le nombre de valeurs manquantes et les types de données
    print(data.info())
    print(data.isnull().sum())
    print(data.describe(include='all'))


def preprocess_data_imputation(data):
    """
    Fonction utilitaire qui permet de traiter les valeurs manquantes
    """
    # Supprimer les colonnes avec 50% ou plus de valeurs manquantes
    missing_values_count = data.isnull().sum()
    columns_to_remove = missing_values_count[missing_values_count > 0.5 * data.shape[0]].index
    data = data.drop(columns=columns_to_remove)

    # Séparer les colonnes numériques par type (int et float) ainsi que les colonnes catégorielles
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

    # Traiter les colonnes catégorielles
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
        # Cela permettra de conserver la nature discrète des variables catégorielles
        # et ainsi pouvoir inverser l'encodage pour obtenir les catégories originales
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


def handle_outliers_with_iqr(data, factor=1.2):
    """
    Fonction utilitaire qui permet de traiter les valeurs aberrantes
    """
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

def preprocess_data_encode_and_scale(data, target_column='num'):
    """
    Fonction utilitaire qui permet d'appliquer un encodage des valeurs catégorielles
    et une normalisation des valeurs numériques
    """
    # Séparation X et y
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Identifier types de colonnes
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()

    # Standardisation des colonnes numériques pour les mettre à l'échelle
    scaler = StandardScaler()
    X_numeric = pd.DataFrame(scaler.fit_transform(X[numeric_cols]), columns=numeric_cols)

    # Encodage des colonnes catégorielles, car la plupart des algorithmes de
    # ML ne peuvent pas traiter les variables catégorielles
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    X_categorical = pd.DataFrame(
        encoder.fit_transform(X[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    # Recréer le DataFrame avec les colonnes numériques, catégorielles avec la variable cible
    data = pd.concat([X_numeric.reset_index(drop=True), X_categorical.reset_index(drop=True)], axis=1)
    data = pd.concat([data, y.reset_index(drop=True)], axis=1)

    return data

def preprocess_data(data, target_column='num'):
    """
    Fonction utilitaire qui permet d'effectuer l'ensemble des pré-traitements sur les données
    """
    # Traiter les valeurs manquantes
    data = preprocess_data_imputation(data)

    # Gérer les outliers
    data = handle_outliers_with_iqr(data)

    # Appliquer un encodage des valeurs catégorielles (OneHotEncoder)
    # Normaliser des valeurs numériques (StandardScaler)
    data = preprocess_data_encode_and_scale(data, target_column=target_column)

    return data


def apply_pca(X, n_components=0.90, print_variance=False):
    """
    Fonction utilitaire qui permet d'appliquer PCA sur les données
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Enregistrer le modèle PCA, car il sera utilisé pour transformer les nouvelles données
    with open("../models/pca.pkl", 'wb') as file:
        pickle.dump(pca, file)

    # Afficher la variance expliquée par chaque composante principale
    if print_variance:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o')
        plt.title('Variance expliquée par chaque composante principale')
        plt.xlabel('Composantes principales')
        plt.ylabel('Variance expliquée')
        plt.grid()
        plt.savefig('../output/variance.png')
        plt.show()

    return X_pca

def resample_target_variable(X, y):
    """
    Fonction utilitaire qui permet de rééchantillonner la variable cible
    """
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X, y)

    return X_train_balanced, y_train_balanced