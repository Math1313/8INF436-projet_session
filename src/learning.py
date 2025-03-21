import pickle
import pandas as pd
# from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# def load_data():
#     # Charger le jeu de données Iris
#     data = load_iris()
#     X, y = data.data, data.target
#     return X, y

def preprocess_data(X, y):
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normaliser les données
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"X_train mean: {X_train.mean(axis=0)}, std: {X_train.std(axis=0)}")

    
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):

    # Entraîner les modèles
    model1 = LogisticRegression(max_iter=200)
    model1.fit(X_train, y_train)

    model2 = RandomForestClassifier()
    model2.fit(X_train, y_train)

    model3 = KNeighborsClassifier(n_neighbors=5)
    model3.fit(X_train, y_train)

    return model1, model2, model3

# Sauvegarder les modèles avec pickle
def save_model(model, model_name):
    with open(f"models/{model_name}", 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Modèle << {model_name} >> entrainé et sauvegardé avec succès !")

# Fonction pour calculer les métriques
def calculate_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    # print(f"Prédictions uniques du modèle {model}: {set(y_pred)}")
    # print(f"Distribution des prédictions : {pd.Series(y_pred).value_counts(normalize=True)}")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    return accuracy, precision, recall, f1

def save_metrics(metrics, model_name):
    metrics_df = pd.DataFrame([metrics], columns=['accuracy', 'precision', 'recall', 'f1'])
    metrics_df['model'] = model_name
    return metrics_df

if __name__ == '__main__':
    
    # X, y = load_data()
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_classes=3, random_state=42)

    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    model1, model2, model3 = train_models(X_train, y_train)

    model_names = ['model_logistic_regression.pkl', 'model_random_forest.pkl', 'model_knn.pkl']

    for i, model in enumerate([model1, model2, model3]):
        # Sauvegarder les modèles
        save_model(model, model_names[i])

    # Calculer les métriques
    metrics = []
    for i, model in enumerate([model1, model2, model3]):
        metrics.append(calculate_metrics(model, X_test, y_test))
    
    # Sauvegarder les métriques
    metrics_df = pd.concat([save_metrics(metrics[i], model_names[i]) for i in range(3)])
    metrics_df.to_csv('metrics.csv', index=False)
        
    pass