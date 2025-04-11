import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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



def train_model_with_cv(X_train, y_train, model):
    """
    Fonction utilitaire qui permet d'entraîner un modèle avec validation croisée
    """
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    scores = cross_validate(model, X_train, y_train, scoring=scoring, cv=5, return_train_score=True)

    for metric in scoring:
        print(f"{metric.capitalize()} : {np.mean(scores[f'test_{metric}']):.4f}")

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
    # print(exploration(data, save_graphs=True, display_graphs=True))

    # Comme la variable cible contient 5 catégories, nous allons la convertir en binaire
    # 0 = aucune maladie cardiaque
    # 1,2,3,4 = maladie cardiaque
    # Cela facilite l'analyse et la modélisation. Cela donnera également plus de poids à la classe 1
    data = prep_data.create_binary_target(data, target_column='num')

    # Pré-traiter les données en utilisant une fonction qui appellera les autres
    # fonctions nécessaires au prétraitement des données
    data = prep_data.preprocess_data(data, target_column='num')

    print(data)

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

    for model_name, model in models.items():
        model, scores = train_model_with_cv(X_train, y_train, model)

        with open(f'../models/{model_name.lower()}_model.pkl', 'wb') as file:
            pickle.dump(model, file)

        y_pred = model.predict(X_test)
        print(f"\\nRapport de classification sur l'ensemble de test pour {model_name} :")
        print(classification_report(y_test, y_pred))

        # plot_learning_curves(scores, model_name, show_graphs=False)
        test_accuracy = plot_learning_curves(model, X_train, X_test, y_train, y_test, model_name, show_graphs=False)
        print(f"Test Accuracy for {model_name}: {test_accuracy:.4f}")
    return 0

if __name__ == "__main__":
    main()