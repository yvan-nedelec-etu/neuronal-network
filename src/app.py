"""
Projet: Discrimination d'une image (Base Wang/Corel-1K)
Université de Poitiers — BUT Niort — Introduction aux réseaux de neurones

Ce script propose deux approches de classification:
  1) Approche basée descripteurs (FCTH) via un MLP (Keras/TensorFlow)
  2) Approche "Deep" bout-en-bout par CNN à partir des images

Prérequis fichiers (posez-les à la racine du projet):
  - wang.xlsx : matrice descripteurs FCTH, 1000 x D (une ligne par image i.jpg)
  - ./wang_images/ : dossier contenant les 1000 images nommées i.jpg (i de 0 à 999)

Hypothèses d'indexation (selon l'énoncé):
  - Classes regroupées par centaine de l'index i
      0–99   : Jungle
      100–199: Plage
      200–299: Monuments
      300–399: Bus
      400–499: Dinosaures
      500–599: Éléphants
      600–699: Fleurs
      700–799: Chevaux
      800–899: Montagne
      900–999: Plats

Utilisation:
  - Exécutez ce fichier tel quel (python wang_classification_python.py).
  - Les sections sont indépendantes : la partie CNN saute automatiquement si le dossier d'images est manquant.

Librairies nécessaires:
  pandas, numpy, scikit-learn, matplotlib, tensorflow>=2.10, pillow

"""

from __future__ import annotations
import os
import sys
import json
import time
import math
import random
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from PIL import Image

# ===========================
# Configuration globale
# ===========================
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

CLASS_NAMES = [
    "Jungle", "Plage", "Monuments", "Bus", "Dinosaures",
    "Éléphants", "Fleurs", "Chevaux", "Montagne", "Plats"
]
N_CLASSES = len(CLASS_NAMES)

IMG_DIR = "../data/Wang"  # répertoire des images i.jpg
EXCEL_PATH = "../data/WangSignatures.xls"  # descripteurs FCTH
IMG_SIZE = (256, 256)
BATCH_SIZE = 32

OUT_DIR = "../outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ===========================
# Utilitaires étiquettes
# ===========================

def id_to_class_index(i: int) -> int:
    """Retourne l'indice de classe (0..9) à partir de l'index d'image i (0..999)."""
    if i < 0 or i > 999:
        raise ValueError(f"Index image invalide: {i}")
    return i // 100


def class_index_to_name(idx: int) -> str:
    return CLASS_NAMES[idx]


# ===========================
# Visualisation
# ===========================

def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], title: str, save_as: Optional[str] = None):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=150)
    plt.show()


def plot_history(history: keras.callbacks.History, title: str, save_as: Optional[str] = None):
    hist = history.history
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(hist.get("loss", []), label="train loss")
    ax.plot(hist.get("val_loss", []), label="val loss")
    if "accuracy" in hist:
        ax.plot(hist.get("accuracy", []), label="train acc")
    if "val_accuracy" in hist:
        ax.plot(hist.get("val_accuracy", []), label="val acc")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.legend()
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=150)
    plt.show()


# ===========================
# 1) APPROCHE DESCRIPTEURS (FCTH) + MLP
# ===========================

def load_fcth_features(excel_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Charge la matrice FCTH (wang.xlsx) et génère les labels selon l'index ligne.
    Retourne X (float32) et y (int64).
    """
    if not os.path.exists(excel_path):
        raise FileNotFoundError(
            f"Fichier descripteurs introuvable: {excel_path}. Placez wang.xlsx à la racine."
        )
    df = pd.read_excel(excel_path, header=None)
    # Suppose: 1000 lignes (0..999), D colonnes de descripteurs
    if df.shape[0] != 1000:
        print(f"[Avertissement] Nombre de lignes inattendu dans {excel_path}: {df.shape[0]} (attendu 1000)")
    X = df.values.astype(np.float32)
    # Label par index de ligne = id image
    ids = np.arange(df.shape[0], dtype=int)
    y = np.array([id_to_class_index(i) for i in ids], dtype=np.int64)
    return X, y


def build_mlp(input_dim: int, hidden_units: List[int] = [256, 128], dropout: float = 0.2) -> keras.Model:
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for hu in hidden_units:
        model.add(layers.Dense(hu, activation="relu"))
        if dropout and dropout > 0.0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(N_CLASSES, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_mlp_pipeline():
    print("\n=== APPROCHE FCTH + MLP ===")
    X, y = load_fcth_features(EXCEL_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_mlp(X_train.shape[1], hidden_units=[512, 256], dropout=0.3)
    cb = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    ]
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=32,
        callbacks=cb,
        verbose=0,
    )

    plot_history(history, title="MLP FCTH — courbes d'apprentissage", save_as=os.path.join(OUT_DIR, "mlp_fcth_history.png"))

    # Évaluation
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"MLP — Test accuracy: {test_acc:.4f} — Test loss: {test_loss:.4f}")

    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    print("\nRapport de classification (MLP FCTH):")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

    plot_confusion(y_test, y_pred, CLASS_NAMES, title="Matrice de confusion — MLP FCTH",
                   save_as=os.path.join(OUT_DIR, "mlp_fcth_confusion.png"))

    # Comparaison rapide hyperparamètres
    print("\n=== Exploration hyperparamètres MLP (rapide) ===")
    grids = [
        dict(hidden=[256, 128], dropout=0.2),
        dict(hidden=[512, 256], dropout=0.3),
        dict(hidden=[1024, 256], dropout=0.4),
    ]
    for g in grids:
        m = build_mlp(X_train.shape[1], hidden_units=g["hidden"], dropout=g["dropout"])
        h = m.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)])
        _, acc = m.evaluate(X_test, y_test, verbose=0)
        print(f"Config {g} => Test acc: {acc:.4f}")


# ===========================
# 2) APPROCHE DEEP (CNN) SUR IMAGES BRUTES
# ===========================

def list_image_paths(img_dir: str) -> List[str]:
    if not os.path.isdir(img_dir):
        return []
    paths = []
    for name in os.listdir(img_dir):
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            paths.append(os.path.join(img_dir, name))
    # On attend i.jpg => trions par index i
    def key_fn(p: str) -> int:
        try:
            stem = os.path.splitext(os.path.basename(p))[0]
            return int(stem)
        except Exception:
            return 10**9
    paths.sort(key=key_fn)
    return paths


def build_cnn(input_shape=(256, 256, 3), base_filters: int = 32, dense_units: int = 256, dropout: float = 0.3) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        # Extraction de descripteurs par convolution
        layers.Conv2D(base_filters, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(base_filters * 2, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(base_filters * 4, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(dense_units, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(N_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def prepare_image_dataset(paths: List[str], img_size: Tuple[int, int] = IMG_SIZE, batch_size: int = BATCH_SIZE):
    """Crée un tf.data à partir des chemins, infère le label via i//100 (i depuis le nom i.jpg)."""
    def parse_label(path: tf.Tensor) -> tf.Tensor:
        # extraire l'index i depuis le nom de fichier
        fname = tf.strings.split(path, os.sep)[-1]
        stem = tf.strings.regex_replace(fname, "\\.jpg$|\\.jpeg$|\\.png$", "", replace_global=True)
        i = tf.strings.to_number(stem, out_type=tf.int32)
        return i // 100

    def load_and_preprocess(path: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        label = parse_label(path)
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, img_size)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # Split train/test de manière déterministe
    n = len(paths)
    idx = np.arange(n)
    # On fabrique y pour stratifier approximativement (par i//100)
    y = (idx // 100).astype(int)
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
    train_paths = [paths[i] for i in train_idx]
    test_paths = [paths[i] for i in test_idx]

    def ds_from_paths(ps: List[str], augment: bool) -> tf.data.Dataset:
        ds_local = tf.data.Dataset.from_tensor_slices(ps)
        ds_local = ds_local.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            aug = keras.Sequential([
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.05),
                layers.RandomZoom(0.1),
            ])
            ds_local = ds_local.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        return ds_local.shuffle(1024, seed=SEED).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = ds_from_paths(train_paths, augment=True)
    test_ds = ds_from_paths(test_paths, augment=False)
    return train_ds, test_ds


def run_cnn_pipeline():
    print("\n=== APPROCHE CNN (Deep) SUR IMAGES ===")
    paths = list_image_paths(IMG_DIR)
    if len(paths) == 0:
        print(f"[Info] Dossier images introuvable ou vide: {IMG_DIR}. Approche CNN ignorée.")
        return

    train_ds, test_ds = prepare_image_dataset(paths, img_size=IMG_SIZE, batch_size=BATCH_SIZE)

    model = build_cnn(input_shape=(*IMG_SIZE, 3), base_filters=32, dense_units=256, dropout=0.4)
    cb = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    ]
    history = model.fit(
        train_ds,
        validation_data=test_ds,  # petite base => on valide sur test pour la démo; idéalement séparer val/test
        epochs=50,
        callbacks=cb,
        verbose=1,
    )

    plot_history(history, title="CNN — courbes d'apprentissage", save_as=os.path.join(OUT_DIR, "cnn_history.png"))

    # Évaluation + matrice de confusion
    y_true, y_pred = [], []
    for xb, yb in test_ds:
        probs = model.predict(xb, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1))
        y_true.extend(yb.numpy())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\nRapport de classification (CNN):")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    plot_confusion(y_true, y_pred, CLASS_NAMES, title="Matrice de confusion — CNN",
                   save_as=os.path.join(OUT_DIR, "cnn_confusion.png"))

    # Mini exploration hyperparamètres
    print("\n=== Exploration hyperparamètres CNN (rapide) ===")
    for bf in [16, 32, 48]:
        m = build_cnn(input_shape=(*IMG_SIZE, 3), base_filters=bf, dense_units=256, dropout=0.4)
        h = m.fit(train_ds, validation_data=test_ds, epochs=30, verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])
        _, acc = m.evaluate(test_ds, verbose=0)
        print(f"base_filters={bf} => Test acc: {acc:.4f}")


# ===========================
# MAIN
# ===========================
if __name__ == "__main__":
    # 1) Approche descripteurs
    try:
        run_mlp_pipeline()
    except FileNotFoundError as e:
        print(str(e))
    except Exception as e:
        print("[ERREUR MLP]", e)

    # 2) Approche CNN
    try:
        run_cnn_pipeline()
    except Exception as e:
        print("[ERREUR CNN]", e)

    print("\nTerminé. Les figures sont enregistrées dans ./outputs/ si affichage non possible.")
