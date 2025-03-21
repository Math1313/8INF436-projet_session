import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

def print_separator():
    print("=====================================")
def import_data():
    df = pd.read_csv(f"data/heart_disease_uci.csv")
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
    print_separator()
    return train_set, test_set

def explore_data(df):
    print(df.head())
    print(df.shape)
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())

    data_num = df.select_dtypes(include=['int64', 'float64'])
    corr_matrix = data_num.corr()
    print(corr_matrix['num'].sort_values(ascending=False))

    print_separator()
    """
    Résultat de l'exploration des données
    Shape: 
        (736, 16)
    Info:
        <class 'pandas.core.frame.DataFrame'>
        Index: 736 entries, 880 to 102
        Data columns (total 16 columns):
        #   Column    Non-Null Count  Dtype  
        ---  ------    --------------  -----  
        0   id        736 non-null    int64  
        1   age       736 non-null    int64  
        2   sex       736 non-null    object 
        3   dataset   736 non-null    object 
        4   cp        736 non-null    object 
        5   trestbps  691 non-null    float64
        6   chol      711 non-null    float64
        7   fbs       656 non-null    object 
        8   restecg   735 non-null    object 
        9   thalch    694 non-null    float64
        10  exang     694 non-null    object 
        11  oldpeak   691 non-null    float64
        12  slope     498 non-null    object 
        13  ca        250 non-null    float64
        14  thal      346 non-null    object 
        15  num       736 non-null    int64  
        dtypes: float64(5), int64(3), object(8)
        memory usage: 97.8+ KB
        None
    Description:
                       id         age    trestbps        chol      thalch     oldpeak          ca         num
        count  736.000000  736.000000  691.000000  711.000000  694.000000  691.000000  250.000000  736.000000
        mean   461.173913   53.679348  131.668596  197.763713  137.383285    0.881187    0.688000    0.975543
        std    265.515183    9.226723   19.269226  110.881121   26.020584    1.111322    0.960388    1.142595
        min      1.000000   28.000000    0.000000    0.000000   60.000000   -2.600000    0.000000    0.000000
        25%    226.750000   47.000000  120.000000  175.000000  120.000000    0.000000    0.000000    0.000000
        50%    469.500000   54.000000  130.000000  222.000000  140.000000    0.500000    0.000000    1.000000
        75%    686.250000   60.000000  140.000000  266.000000  158.000000    1.500000    1.000000    2.000000
        max    920.000000   77.000000  200.000000  603.000000  202.000000    6.200000    3.000000    4.000000
    Null values:
        id            0
        age           0
        sex           0
        dataset       0
        cp            0
        trestbps     45
        chol         25
        fbs          80
        restecg       1
        thalch       42
        exang        42
        oldpeak      45
        slope       238
        ca          486
        thal        390
        num           0
        dtype: int64
    Correlation:
        num         1.000000
        ca          0.522610
        oldpeak     0.447453
        age         0.319181
        id          0.272929
        trestbps    0.121011
        chol       -0.222180
        thalch     -0.369152
    """

def visualize_data(df):
    # Convertir les colonnes numériques si nécessaire
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

    # Générer des boxplots pour détecter les données aberrantes
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(numeric_columns, 1):
        plt.subplot(3, 3, i)
        sns.boxplot(data=df, y=column)
        plt.title(f'Boxplot - {column}')
        plt.tight_layout()

    plt.savefig('./output/boxplots.png')  # Sauvegarder les boxplots
    plt.show()

    # Générer des histogrammes pour détecter des lois normales ou autres distributions
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(numeric_columns, 1):
        plt.subplot(3, 3, i)
        sns.histplot(data=df, x=column, kde=True, bins=20)
        plt.title(f'Histogram - {column}')
        plt.tight_layout()

    plt.savefig('./output/histograms.png')  # Sauvegarder les histogrammes
    plt.show()
    """
    Résultat de la visualisation des données
    Boxplots:
        - Les colonnes 'trestbps', 'chol' et 'oldpeak' contiennent des données aberrantes
    Histograms:
        - Les colonnes 'age', 'trestbps', 'chol', 'thalch' et 'oldpeak' semblent suivre une distribution normale""
    """

def nettoyage_outliers(X):
    """
    Fonction pour détecter et traiter les valeurs aberrantes en utilisant la méthode IQR.
    
    Args:
        X (array ou DataFrame): Les données à nettoyer
        
    Returns:
        DataFrame: Les données nettoyées sans valeurs aberrantes
    """
    # Conversion en DataFrame si c'est un array
    if not isinstance(X, pd.DataFrame):
        if hasattr(X, 'columns'):
            columns = X.columns
        else:
            columns = [f'col_{i}' for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=columns)
    
    # Calcul des quartiles et IQR
    Q25 = X.quantile(0.25)
    Q75 = X.quantile(0.75)
    IQR = Q75 - Q25
    
    # Calcul des seuils
    SeuilMin = (Q25 - 1.5 * IQR)
    SeuilMax = (Q75 + 1.5 * IQR)
    
    # Écrêtage des valeurs
    X_clean = X.clip(lower=SeuilMin, upper=SeuilMax, axis=1)
    
    return X_clean


def creer_preprocessor(colonnes_numeriques, colonnes_categorielles):
    """
    Crée un preprocessor pour le traitement des données.
    
    Args:
        colonnes_numeriques (list): Liste des noms des colonnes numériques
        colonnes_categorielles (list): Liste des noms des colonnes catégorielles
        
    Returns:
        ColumnTransformer: Le preprocessor configuré
    """
    # Création du pipeline pour les variables numériques
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('outliers', FunctionTransformer(nettoyage_outliers, validate=False))
    ])
    
    # Création du pipeline pour les variables catégorielles
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    
    # Création du ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, colonnes_numeriques),
            ('cat', categorical_pipeline, colonnes_categorielles)
        ],
            remainder='passthrough'
        )
    
    return preprocessor


def preprocess_data(df, colonnes_numeriques, colonnes_categorielles):
    """
    Prétraite un DataFrame en appliquant différentes transformations selon le type de variable.
    
    Args:
        df (DataFrame): Le DataFrame à prétraiter
        colonnes_numeriques (list): Liste des noms des colonnes numériques
        colonnes_categorielles (list): Liste des noms des colonnes catégorielles
        
    Returns:
        DataFrame: Le DataFrame prétraité
    """
    # Vérifier que les colonnes spécifiées sont présentes
    all_specified_columns = colonnes_numeriques + colonnes_categorielles
    missing_columns = [col for col in all_specified_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Les colonnes suivantes manquent dans le DataFrame: {missing_columns}")
    
    # Sauvegarder les types de données originaux
    dtypes_original = X_train.dtypes.to_dict()

    # Créer et appliquer le preprocessor
    preprocessor = creer_preprocessor(colonnes_numeriques, colonnes_categorielles)
    processed_data = preprocessor.fit_transform(df)
    
    # Récupérer toutes les colonnes, y compris celles non spécifiées
    remainder_columns = [col for col in df.columns if col not in all_specified_columns]
    all_columns = all_specified_columns + remainder_columns
    
    # Convertir le résultat en DataFrame
    processed_df = pd.DataFrame(processed_data, columns=all_columns)

     # Restaurer les types de données originaux
    for col in processed_df.columns:
        if col in dtypes_original:
            processed_df[col] = processed_df[col].astype(dtypes_original[col])
    
    return preprocessor, processed_df


if __name__ == '__main__':
    train_set, test_set = import_data()
    explore_data(train_set)
    # visualize_data(train_set)
    """
     À la suite de l'exploration de données, nous devrons effectuer les étapes suivantes:
        - Séparer la variable ciible des variables explicatives
        - Remplacer les valeurs manquantes
        - Traiter les données aberrantes
        - Encoder les variables catégorielles
        - Normaliser les données
    """
    colonnes_numeriques = ['trestbps', 'chol', 'thalch', 'oldpeak']
    colonnes_categorielles = ['sex', 'cp', 'fbs', 'restecg', 'ca', 'thal', 'exang', 'slope']
    X_train, y_train = train_set.drop('num', axis=1), train_set['num']
    preprocessor, processed_train = preprocess_data(X_train, colonnes_numeriques, colonnes_categorielles)

    # processed_train_with_target = pd.concat([processed_train, y_train.reset_index(drop=True)], axis=1)

    # data_num = processed_train.select_dtypes(include=['int64', 'float64'])
    # corr_matrix = data_num.corr()
    # print(corr_matrix['num'].sort_values(ascending=False))
    
    # visualize_data(processed_train)