import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_validate,learning_curve
from imblearn.over_sampling import SMOTE
import preprocess_data as prep_data
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

# Modifier les limitations d'affichage par défaut de Pandas pour faciliter l'exploration des données.
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 25)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)



def exploration(data, save_graphs, display_graphs, data_state):
    """
    Fonction utilitaire qui permet d'explorer les données
    """
    # Afficher les histogrammes des toutes les variables
    data.hist(bins=50, figsize=(15, 10))
    if save_graphs: plt.savefig(f"../output/histograms_{data_state}.png")
    if display_graphs: plt.show()

    # Afficher les boîtes à moustaches pour toutes les variables numériques
    # Cela permet de visualiser les valeurs aberrantes
    data.select_dtypes(include=['int64', 'float64']).plot(kind='box', subplots=True, layout=(4, 6), figsize=(15, 10))
    if save_graphs: plt.savefig(f'../output/boxplots_{data_state}.png')
    if display_graphs: plt.show()

    # Afficher les informations sur les données
    # Permet de voir le nombre de valeurs manquantes et les types de données
    print(data.info())
    print(data.isnull().sum())
    print(data.describe(include='all'))


def train_model_with_cv(X_train, y_train, model):
    """
    Fonction utilitaire qui permet d'entraîner un modèle avec validation croisée
    """
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    scores = cross_validate(model, X_train, y_train, scoring=scoring, cv=5, return_train_score=True)

    model.fit(X_train, y_train)
    return model, scores

def plot_learning_curves(estimator, X_train, X_test, y_train, y_test, model_name, show_graphs=False):
    """
    Fonction utilitaire qui permet de tracer les courbes d'apprentissage
    """
    train_sizes, train_scores, test_scores = learning_curve(estimator, X_train, y_train,
                                                            cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5),
                                                            scoring='accuracy', random_state=1)

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Test score')

    # Calculer et afficher le score de test final
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    plt.axhline(y=test_accuracy, color='b', linestyle='--', label='Cross-validation score')

    plt.title(f'Learning Curve for {model_name}')
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy Score')
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(f'../output/{model_name.lower()}_learning_curve.png')
    if show_graphs: plt.show()

    return test_accuracy



def main():
    """
    TRAITEMENT DES DONNÉES
    """
    # Charger les données
    # Éliminer l'identifiant unique, car il n'est pas pertinent pour l'analyse
    data = pd.read_csv('../data/heart_disease_uci.csv')
    data = data.drop(['id'], axis=1)

    # Eplorer rapidement les données
    # 1. Il y a des valeurs manquantes
    # 2. Il y a des valeurs aberrantes
    # Enregistrer les graphiques dans le dossier "output" et les afficher si possible
    print(exploration(data, save_graphs=True, display_graphs=False, data_state="raw"))

    # Comme la variable cible contient 5 catégories, nous allons la convertir en binaire
    # 0 = aucune maladie cardiaque
    # 1,2,3,4 = maladie cardiaque
    # Cela facilite l'analyse et la modélisation. Cela donnera également plus de poids à la classe 1
    data = prep_data.create_binary_target(data, target_column='num')

    # Pré-traiter les données en utilisant une fonction qui appellera les autres
    # fonctions nécessaires au prétraitement des données
    data = prep_data.preprocess_data(data, target_column='num')

    print(exploration(data, save_graphs=True, display_graphs=False, data_state="preprocessed"))

    # Séparer les variables explicatives et la variable cible
    X = data.drop(columns=['num'])
    y = data['num']

    # Appliquer PCA pour réduire la dimensionnalité
    X = prep_data.apply_pca(X, n_components=0.90, print_variance=False)

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)

    # Ré-échantillonner la variable cible pour équilibrer les classes si nécessaire
    # Le déséquilibre limite acceptable de la variable cible est de 35% - 65%
    target_counts = y_train.value_counts(normalize=True) * 100
    if min(target_counts) <= 35:
        X_train, y_train = prep_data.resample_target_variable(X_train, y_train)



    """
    ENTRAINEMENT DES MODÈLES DE CLASSIFICATION BINAIRE
    """

    models = {
        'RandomForest': RandomForestClassifier(
            random_state=2,
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
        ),
        'LogisticRegression': LogisticRegression(
            random_state=2,
            solver='lbfgs',
        ),
        'SVM': SVC(
            random_state=2,
            kernel='rbf'
        ),
        'GradientBoosting': GradientBoostingClassifier(
            random_state=2,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            min_samples_split=2,
            min_samples_leaf=1,
        )
    }

    # Créer un fichier CSV pour stocker les résultats
    csv_file_path = '../output/model_results.csv'
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            "Model",
            "Test Accuracy",
            "Precision (0)",
            "Recall (0)",
            "F1 Score (0)",
            "Precision (1)",
            "Recall (1)",
            "F1 Score (1)"
        ])

    for model_name, model in models.items():
        model, scores = train_model_with_cv(X_train, y_train, model)

        with open(f'../models/{model_name.lower()}_model.pkl', 'wb') as file:
            pickle.dump(model, file)

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        test_accuracy = accuracy_score(y_test, y_pred)

        # Ajouter les métriques au fichier CSV
        with open(csv_file_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([
                model_name,
                f"{test_accuracy:.4f}",
                f"{report['0']['precision']:.4f}",
                f"{report['0']['recall']:.4f}",
                f"{report['0']['f1-score']:.4f}",
                f"{report['1']['precision']:.4f}",
                f"{report['1']['recall']:.4f}",
                f"{report['1']['f1-score']:.4f}"
            ])

        # plot_learning_curves(scores, model_name, show_graphs=False)
        test_accuracy = plot_learning_curves(model, X_train, X_test, y_train, y_test, model_name, show_graphs=False)
    return 0

if __name__ == "__main__":
    main()