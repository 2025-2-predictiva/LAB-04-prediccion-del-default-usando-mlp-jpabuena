# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import gzip
import json
import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"default payment next month": "default"})
    df = df.drop(columns="ID")
    df = df.dropna()
    df = df[
        (df["SEX"] != 0) &
        (df["EDUCATION"] != 0) &
        (df["MARRIAGE"] != 0)
    ]
    df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4

    return df

def create_pipeline():
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    numerical_features = ["LIMIT_BAL", "AGE",
                          "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
                          "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4",
                          "BILL_AMT5", "BILL_AMT6",
                          "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
                          "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]


    transformer = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(dtype='int', handle_unknown='ignore'),
             categorical_features),
            ('num', StandardScaler(), numerical_features)
        ],
        remainder="passthrough"
    )


    pipeline = Pipeline([
        ('transformer', transformer),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('pca', PCA()),
        ('classifier', MLPClassifier(max_iter=15000, random_state=42))
    ])

    return pipeline


def make_grid_search(pipeline, cv=10):
    param_grid = {
        'pca__n_components': [None],
        'feature_selection__k': [20],
        'classifier__hidden_layer_sizes': [(50,30,40,60)],
        'classifier__alpha': [0.28],
        'classifier__learning_rate_init': [0.001]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='balanced_accuracy',
        cv=cv,
        refit='true'
    )

    return grid_search

def save_model(estimator):
    if not os.path.exists("../files/models"):
        os.makedirs("../files/models")
    with gzip.open("../files/models/model.pkl.gz", "wb") as f:
        pickle.dump(estimator, f)

def calc_metrics_precision(y_true, y_pred, dataset_type):
    return {
        "type": "metrics",
        "dataset": dataset_type,
        "precision": precision_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }

def calc_conf_matrix(y_true, y_pred, dataset_type):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "type": "cm_matrix",
        "dataset": dataset_type,
        "true_0": {"predicted_0": int(tn), "predicted_1": int(fp)},
        "true_1": {"predicted_0": int(fn), "predicted_1": int(tp)}
    }

def save_metrics(metrics):
    if not os.path.exists("../files/output"):
        os.makedirs("../files/output")

    with open("../files/output/metrics.json", "w") as f:
        for m in metrics:
            f.write(json.dumps(m) + "\n")

# load the data
train_data = pd.read_csv("../files/input/train_data.csv.zip", compression="zip")
test_data = pd.read_csv("../files/input/test_data.csv.zip", compression="zip")

# clean the data
train_data = clean_data(train_data)
test_data = clean_data(test_data)

# separate the data
(x_train, y_train) = train_data.drop(columns="default"), train_data["default"]
(x_test, y_test) = test_data.drop(columns="default"), test_data["default"]

# create the pipeline
pipeline = create_pipeline()
pipeline.fit(x_train, y_train)

# hyperparameter
estimator = make_grid_search(pipeline)
estimator.fit(x_train, y_train)

# save model
save_model(estimator)

# calculate metrics
y_train_pred = estimator.predict(x_train)
y_test_pred = estimator.predict(x_test)

train_metrics = calc_metrics_precision(y_train, y_train_pred, "train")
test_metrics = calc_metrics_precision(y_test, y_test_pred, "test")

train_cm = calc_conf_matrix(y_train, y_train_pred, "train")
test_cm = calc_conf_matrix(y_test, y_test_pred, "test")

metrics = [train_metrics, test_metrics, train_cm, test_cm]
save_metrics(metrics)