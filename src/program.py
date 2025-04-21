import os
import sys
import glob
import pickle
import pandas as pd

from PyQt5 import QtWidgets, QtCore

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Chargement des composants de transformation et des modèles
        self.load_components_and_models()

        self.setWindowTitle("Application de Prédiction et Métriques")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("""
            QWidget {
                background-color: #F0F8FF;
                font-family: Arial;
                font-size: 14px;
            }
            QLineEdit, QComboBox {
                padding: 5px;
                border: 1px solid #A9A9A9;
                border-radius: 5px;
            }
            QPushButton {
                background-color: #4682B4;
                color: white;
                padding: 10px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #5A9BD5;
            }
            QLabel {
                padding: 5px;
            }
            QTableWidget {
                border: 1px solid #A9A9A9;
                border-radius: 5px;
            }
        """)

        # Création d'un QTabWidget pour contenir les deux onglets
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # Onglet de prédiction
        self.prediction_tab = QtWidgets.QWidget()
        self.create_prediction_tab()
        self.tabs.addTab(self.prediction_tab, "Prédiction")

        # Onglet des métriques
        self.metrics_tab = QtWidgets.QWidget()
        self.create_metrics_tab()
        self.tabs.addTab(self.metrics_tab, "Métriques")

    def create_prediction_tab(self):
        layout = QtWidgets.QVBoxLayout(self.prediction_tab)
        form_layout = QtWidgets.QFormLayout()
        form_layout.setSpacing(10)
        layout.addLayout(form_layout)

        # Champs d'entrée de données
        self.age_edit = QtWidgets.QLineEdit()
        form_layout.addRow("Age (entier) :", self.age_edit)

        self.sex_combo = QtWidgets.QComboBox()
        self.sex_combo.addItems(["Homme", "Femme"])
        form_layout.addRow("Sexe :", self.sex_combo)

        self.location_combo = QtWidgets.QComboBox()
        self.location_combo.addItems(["Cleveland", "Hungary", "Switzerland", "VA Long Beach"])
        form_layout.addRow("Location :", self.location_combo)

        self.cp_combo = QtWidgets.QComboBox()
        self.cp_combo.addItems(["atypical angina", "asymptomatic", "non-anginal", "typical angina"])
        form_layout.addRow("Chest Pain (cp) :", self.cp_combo)

        self.fbs_checkbox = QtWidgets.QCheckBox("Fasting Blood Sugar > 120 mg/dl")
        form_layout.addRow("FBS :", self.fbs_checkbox)

        self.trestbps_edit = QtWidgets.QLineEdit()
        form_layout.addRow("Trestbps (pression artérielle) :", self.trestbps_edit)

        self.chol_edit = QtWidgets.QLineEdit()
        form_layout.addRow("Chol (cholestérol) :", self.chol_edit)

        self.restecg_combo = QtWidgets.QComboBox()
        self.restecg_combo.addItems(["st-t abnormality", "normal", "lv hypertrophy"])
        form_layout.addRow("Restecg :", self.restecg_combo)

        self.thalch_edit = QtWidgets.QLineEdit()
        form_layout.addRow("Thalch (pouls max) :", self.thalch_edit)

        self.exang_checkbox = QtWidgets.QCheckBox("Exercise Induced Angina")
        form_layout.addRow("Exang :", self.exang_checkbox)

        self.oldpeak_edit = QtWidgets.QLineEdit()
        form_layout.addRow("Oldpeak (dépression ST) :", self.oldpeak_edit)

        self.slop_combo = QtWidgets.QComboBox()
        self.slop_combo.addItems(["upsloping", "flat", "downsloping"])
        form_layout.addRow("Slope :", self.slop_combo)

        # Sélection du modèle à utiliser
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(list(self.models.keys()))
        form_layout.addRow("Sélection du modèle :", self.model_combo)

        # Bouton de prédiction
        self.predict_button = QtWidgets.QPushButton("Prédiction")
        self.predict_button.clicked.connect(self.make_prediction)
        layout.addWidget(self.predict_button)

        # Zone d'affichage du résultat
        self.result_label = QtWidgets.QLabel("Résultat de la prédiction s'affichera ici.")
        self.result_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.result_label)

    def create_metrics_tab(self):
        layout = QtWidgets.QVBoxLayout(self.metrics_tab)
        # Titre de l'onglet
        title = QtWidgets.QLabel("Métriques des Modèles")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 16px;")
        layout.addWidget(title)

        # Création du tableau pour afficher les métriques
        self.metrics_table = QtWidgets.QTableWidget()
        layout.addWidget(self.metrics_table)

        # Chargement des données du CSV et remplissage du tableau
        self.load_metrics()

    def load_metrics(self):
        # Chemin du fichier CSV
        csv_path = os.path.join("..", "output", "model_results.csv")
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Erreur", f"Erreur lors du chargement du CSV des métriques:\n{e}")
            return

        # Configuration du QTableWidget
        self.metrics_table.setRowCount(df.shape[0])
        self.metrics_table.setColumnCount(df.shape[1])
        self.metrics_table.setHorizontalHeaderLabels(df.columns.tolist())

        # Remplissage du tableau
        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                item = QtWidgets.QTableWidgetItem(str(df.iat[row, col]))
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.metrics_table.setItem(row, col, item)

        # Ajustement automatique de la largeur des colonnes
        self.metrics_table.resizeColumnsToContents()

    def load_components_and_models(self):
        """
        Charge scaler, encoder, pca depuis ../tools et les modèles depuis ../models.
        On suppose que dans ../tools, les fichiers se nomment :
          - scaler.pkl
          - encoder.pkl
          - pca.pkl
        Et dans ../models, les fichiers dont le nom contient 'model' sont les modèles.
        """
        # Chargement des composants depuis ../tools
        tools_path = os.path.join("..", "tools")
        self.scaler = None
        self.encoder = None
        self.pca = None

        for file in glob.glob(os.path.join(tools_path, "*.pkl")):
            filename = os.path.basename(file)
            try:
                with open(file, "rb") as f:
                    obj = pickle.load(f)
            except Exception as e:
                print(f"Erreur lors du chargement de {filename} depuis tools: {e}")
                continue

            if "scaler" in filename.lower():
                self.scaler = obj
            elif "encoder" in filename.lower():
                self.encoder = obj
            elif "pca" in filename.lower():
                self.pca = obj

        if self.scaler is None or self.encoder is None or self.pca is None:
            QtWidgets.QMessageBox.critical(self, "Erreur", "Les composants scaler, encoder ou pca n'ont pas pu être chargés depuis ../tools.")
            sys.exit(1)

        # Chargement des modèles depuis ../models
        models_path = os.path.join("..", "models")
        self.models = {}
        for file in glob.glob(os.path.join(models_path, "*.pkl")):
            filename = os.path.basename(file)
            if "model" not in filename.lower():
                continue
            try:
                with open(file, "rb") as f:
                    model_obj = pickle.load(f)
                self.models[filename] = model_obj
            except Exception as e:
                print(f"Erreur lors du chargement du modèle {filename}: {e}")
                continue

        if not self.models:
            QtWidgets.QMessageBox.critical(self, "Erreur", "Aucun modèle n'a été chargé depuis ../models.")
            sys.exit(1)

    def make_prediction(self):
        # Vérifier que tous les champs obligatoires sont remplis
        if not all([
            self.age_edit.text().strip(),
            self.trestbps_edit.text().strip(),
            self.chol_edit.text().strip(),
            self.thalch_edit.text().strip(),
            self.oldpeak_edit.text().strip()
        ]):
            QtWidgets.QMessageBox.warning(self, "Attention", "Veuillez remplir tous les champs obligatoires.")
            return

        try:
            # Construction du dictionnaire de nouvelles données
            new_data = {
                "age": int(self.age_edit.text().strip()),
                "sex": "Male" if self.sex_combo.currentText() == "Homme" else "Female",
                "dataset": self.location_combo.currentText(),
                "cp": self.cp_combo.currentText(),
                "fbs": self.fbs_checkbox.isChecked(),
                "trestbps": int(self.trestbps_edit.text().strip()),
                "chol": int(self.chol_edit.text().strip()),
                "restecg": self.restecg_combo.currentText(),
                "thalch": int(self.thalch_edit.text().strip()),
                "exang": self.exang_checkbox.isChecked(),
                "oldpeak": float(self.oldpeak_edit.text().strip()),
                "slope": self.slop_combo.currentText()
            }

            # Conversion en DataFrame
            new_data_df = pd.DataFrame([new_data])
            numeric_cols = new_data_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = new_data_df.select_dtypes(include=['object', 'bool']).columns.tolist()

            # Appliquer le scaler aux colonnes numériques
            new_data_numeric_scaled = self.scaler.transform(new_data_df[numeric_cols])
            new_data_numeric_scaled_df = pd.DataFrame(new_data_numeric_scaled, columns=numeric_cols)

            # Appliquer l'encoder aux colonnes catégorielles
            new_data_categorical_encoded = self.encoder.transform(new_data_df[categorical_cols])
            new_data_categorical_encoded_df = pd.DataFrame(
                new_data_categorical_encoded,
                columns=self.encoder.get_feature_names_out(categorical_cols)
            )

            # Combiner les données numériques et catégorielles transformées
            new_data_transformed_df = pd.concat([new_data_numeric_scaled_df, new_data_categorical_encoded_df], axis=1)

            # Appliquer PCA
            new_data_pca_transformed = self.pca.transform(new_data_transformed_df)

            # Sélectionner le modèle choisi
            selected_model_name = self.model_combo.currentText()
            model = self.models[selected_model_name]

            # Faire la prédiction
            prediction = model.predict(new_data_pca_transformed)
            # Interpréter le résultat
            if prediction[0] == 0:
                result_text = "Aucun risque de maladie cardiaque pour le moment."
            elif prediction[0] == 1:
                result_text = "Risque de maladie cardiaque élevé."
            else:
                result_text = f"Prédiction inconnue: {prediction[0]}"

            self.result_label.setText(result_text)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Erreur", f"Une erreur est survenue durant la prédiction:\n{e}")

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
