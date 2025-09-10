import numpy as np
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical
import keras
import os
from pathlib import Path
from tqdm import tqdm



def build_model(input_shape : int, n_seas : int) -> Sequential:
    model = Sequential([Input(shape=(input_shape,))])
    model.add(Dense(n_seas, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_perceptron(data : np.array, n_features : int, n_year_training : int = 50, epochs : int = 50, batch_size : int = 52) -> tuple:

    x = data[:, 0:n_features]
    y = data[:, n_features]

    y = to_categorical(y)

    x_train, x_test = x[:365*n_year_training], x[365*n_year_training:]
    y_train, y_test = y[:365*n_year_training], y[365*n_year_training:]

    mod = build_model(n_features, y.shape[1])

    history = mod.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, verbose = False)

    mse = mean_squared_error(y_test, mod.predict(x_test, verbose=False))
    r2 = r2_score(y_test, mod.predict(x_test, verbose=False))
    accuracy = mod.evaluate(x_test, y_test, verbose=False)[1]

    metrics_dict = {'mse': mse, 'r2': r2, 'accuracy': accuracy}

    return mod, history, metrics_dict


def train_perceptron_spatial(
    data: np.ndarray,
    n_features: int,
    n_year_training: int,
    epochs: int,
    batch_size: int,
    models_dir: str | Path,
    min_years_test: int = 1,
    verbose: bool = False
) -> tuple:
    """Addestra un modello perceptron per ciascun punto griglia.

    Parameters
    ----------
    data : np.ndarray
        Array (lat, lon, time, n_features+1) con ultima feature = etichetta intera.
    n_features : int
        Numero di feature climatiche.
    n_year_training : int
        Anni da usare per training (365 giorni per anno). Se eccede disponibilità viene ridotto.
    epochs : int
        Epoche di training.
    batch_size : int
        Batch size.
    models_dir : str | Path
        Directory dove salvare i modelli (sottocartella 'weights').
    min_years_test : int
        Anni minimi richiesti per test set.
    verbose : bool
        Se True stampa progressi.

    Returns
    -------
    mse, r2, accuracy, models, histories
        Metriche (lat, lon) e lista storici accuracy (lat, lon, epochs) opzionale.
    """
    models_dir = Path(models_dir)
    weights_dir = models_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)

    lat_n, lon_n, time_n, total_feat = data.shape
    assert total_feat == n_features + 1, "L'array deve contenere le label come ultima colonna"

    # Calcola anni disponibili
    available_years = time_n // 365
    if available_years < (n_year_training + min_years_test):
        # Riduci training per lasciare almeno un anno test se possibile
        n_year_training = max(1, available_years - min_years_test)
    train_days = 365 * n_year_training

    mse = np.full((lat_n, lon_n), np.nan, dtype=float)
    r2 = np.full((lat_n, lon_n), np.nan, dtype=float)
    accuracy = np.full((lat_n, lon_n), np.nan, dtype=float)
    models = np.empty((lat_n, lon_n), dtype=object)
    histories = np.empty((lat_n, lon_n), dtype=object)

    for j in range(lat_n):
        print(f"Training row {j+1}/{lat_n}")
        for i in tqdm(range(lon_n)):
            series = data[j, i]
            x = series[:, :n_features]
            y = series[:, n_features]

            # Verifica dati sufficienti
            if np.isnan(x).all() or np.isnan(y).all():
                continue
            if train_days >= len(x) - 365:  # lascia almeno un anno test se possibile
                continue

            try:
                y_cat = to_categorical(y)
                n_classes = y_cat.shape[1]
                x_train, x_test = x[:train_days], x[train_days:]
                y_train, y_test = y_cat[:train_days], y_cat[train_days:]

                mod = build_model(n_features, n_classes)
                history = mod.fit(
                    x_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=False,
                )

                y_pred_soft = mod.predict(x_test, verbose=False)
                mse[j, i] = mean_squared_error(y_test, y_pred_soft)
                r2[j, i] = r2_score(y_test, y_pred_soft)
                accuracy[j, i] = mod.evaluate(x_test, y_test, verbose=False)[1]

                # Salva modello
                mod.save(weights_dir / f"model_{j}_{i}.keras")
                models[j, i] = mod
                histories[j, i] = history.history.get('accuracy')
            except Exception as e:
                if verbose:
                    print(f"[WARN] pixel ({j},{i}) skipped: {e}")
                continue

    return mse, r2, accuracy, models, histories

# def train_perceptron(data, n_features, models_dir, n_year_training=50, epochs=50):

#     r2 = np.zeros((np.shape(data)[0], np.shape(data)[1]))
#     mse = np.zeros((np.shape(data)[0], np.shape(data)[1]))
#     accuracy = np.zeros((np.shape(data)[0], np.shape(data)[1]))
#     models = np.zeros((np.shape(data)[0], np.shape(data)[1]), dtype=object)
#     histories = np.zeros((np.shape(data)[0], np.shape(data)[1], epochs), dtype=object)

#     for j in tqdm(range(np.shape(data)[0])):
#         for i in range(np.shape(data)[1]):

#             try:

#                 x = data[j,i,:,0:n_features]
#                 y = data[j,i,:,n_features]

#                 y = to_categorical(y)

#                 x_train, x_test = x[:365*n_year_training], x[365*n_year_training:]
#                 y_train, y_test = y[:365*n_year_training], y[365*n_year_training:]

#                 mod = build_model(n_features, y.shape[1])

#                 history = mod.fit(x_train, y_train, epochs=epochs, batch_size=52, verbose=False)
#                 histories[j,i,:] = history.history['accuracy']

#                 models[j,i] = mod

#                 if not os.path.exists(os.path.join(models_dir, 'weights')):
#                     os.makedirs(os.path.join(models_dir, 'weights'))

#                 mse[j,i] = mean_squared_error(y_test, mod.predict(x_test, verbose=False))
#                 r2[j,i] = r2_score(y_test, mod.predict(x_test, verbose=False))
#                 accuracy[j,i] = mod.evaluate(x_test, y_test, verbose=False)[1]



#                 model_filename = os.path.join(models_dir, 'weights' ,f"model_{j}_{i}.keras")
#                 mod.save(model_filename)
            
#             except:
                
#                 mse[j,i] = np.nan
#                 r2[j,i] = np.nan
#                 accuracy[j,i] = np.nan
#                 models[j,i] = None
        
#     return mse, r2, models, histories, accuracy



def predict_custom(array_res, datarray_model, n_features):

    predictions = np.zeros((np.shape(array_res)[0], np.shape(array_res)[1], np.shape(array_res)[2]))

    models = datarray_model

    for j in tqdm(range(np.shape(array_res)[0])):
        for i in range(np.shape(array_res)[1]):
            
            model = models[j,i]
            x = array_res[j,i,:,0:n_features]

            try:
                predictions[j,i,:] = model.predict(x, verbose=False).argmax(axis=1)
            except:
                predictions[j,i,:] = np.nan

    return predictions


def evaluate_custom(array_res, datarray_model, n_features):
    """
    Valuta i modelli sui dati di input e restituisce l'accuracy, precision e recall per ogni classe.

    Args:
        array_res (numpy.ndarray): Dati di input, inclusi i valori target.
        datarray_model (numpy.ndarray): Matrice contenente i modelli allenati.
        n_features (int): Numero di feature di input.

    Returns:
        tuple: 
            - numpy.ndarray con l'accuracy per ogni classe
            - numpy.ndarray con la precision per ogni classe
            - numpy.ndarray con la recall per ogni classe
    """

    num_classes = None
    # Determiniamo il numero di classi in modo dinamico
    for j in range(np.shape(array_res)[0]):
        for i in range(np.shape(array_res)[1]):
            y = array_res[j, i, :, n_features]
            y_categorical = to_categorical(y)
            if y_categorical.shape[1] > 1:
                num_classes = y_categorical.shape[1]
                break
        if num_classes:
            break

    if num_classes is None:
        raise ValueError("Impossibile determinare il numero di classi.")

    # Inizializziamo la matrice per contenere l'accuracy, precision e recall per classe
    evaluations_accuracy = np.full((np.shape(array_res)[0], np.shape(array_res)[1], num_classes), np.nan)
    evaluations_precision = np.full((np.shape(array_res)[0], np.shape(array_res)[1], num_classes), np.nan)
    evaluations_recall = np.full((np.shape(array_res)[0], np.shape(array_res)[1], num_classes), np.nan)

    models = datarray_model

    for j in tqdm(range(np.shape(array_res)[0])):
        for i in range(np.shape(array_res)[1]):
            x = array_res[j, i, :, 0:n_features]
            y = array_res[j, i, :, n_features]
            y_categorical = to_categorical(y, num_classes=num_classes)

            model = models[j, i]

            if model is not None:
                try:
                    predictions = model.predict(x, verbose=False)  # Output softmax
                    predicted_classes = np.argmax(predictions, axis=1)  # Classe predetta
                    true_classes = np.argmax(y_categorical, axis=1)  # Classe reale

                    # Calcolare l'accuracy per ciascuna classe
                    for c in range(num_classes):
                        mask = (true_classes == c)  # Seleziona solo i campioni della classe c
                        if np.sum(mask) > 0:  # Evita errori se una classe è assente
                            evaluations_accuracy[j, i, c] = np.mean(predicted_classes[mask] == c)
                        else:
                            evaluations_accuracy[j, i, c] = np.nan  # Nessun campione per questa classe

                        # Calcolare la precisione per ciascuna classe
                        true_positives = np.sum((predicted_classes == c) & (true_classes == c))  # TP per la classe c
                        false_positives = np.sum((predicted_classes == c) & (true_classes != c))  # FP per la classe c
                        if true_positives + false_positives > 0:
                            evaluations_precision[j, i, c] = true_positives / (true_positives + false_positives)
                        else:
                            evaluations_precision[j, i, c] = np.nan  # Nessuna occorrenza della classe c

                        # Calcolare la recall per ciascuna classe
                        false_negatives = np.sum((predicted_classes != c) & (true_classes == c))  # FN per la classe c
                        if true_positives + false_negatives > 0:
                            evaluations_recall[j, i, c] = true_positives / (true_positives + false_negatives)
                        else:
                            evaluations_recall[j, i, c] = np.nan  # Nessuna occorrenza della classe c

                except Exception as e:
                    pass
                    #print(f"Errore nella valutazione del modello ({j}, {i}): {e}")

    return evaluations_accuracy, evaluations_precision, evaluations_recall




def load_models(models_dir):
    """
    Carica i modelli salvati dalla directory specificata e li restituisce in una matrice.

    Args:
        models_dir (str): La directory in cui sono salvati i modelli.

    Returns:
        numpy.ndarray: Una matrice di modelli caricati.
    """

    models = []
    #Trova il numero di modelli salvati.
    models_dir = os.path.join(models_dir, 'weights')
    num_models = 0
    for filename in os.listdir(models_dir):
      if filename.endswith(".keras"):
        num_models +=1

    #Trova le dimensioni della matrice.
    max_j = 0
    max_i = 0

    for filename in os.listdir(models_dir):
      if filename.endswith(".keras"):
        parts = filename.split("_")
        j = int(parts[1])
        i = int(parts[2].split(".")[0])
        if j > max_j:
          max_j = j
        if i > max_i:
          max_i = i

    max_j +=1
    max_i +=1
    models = np.zeros((max_j, max_i), dtype=object)

    for filename in os.listdir(models_dir):
        if filename.endswith(".keras"):
            parts = filename.split("_")
            j = int(parts[1])
            i = int(parts[2].split(".")[0])
            model_path = os.path.join(models_dir, filename)
            try:
                model = keras.models.load_model(model_path)
                models[j, i] = model
            except Exception as e:
                print(f"Errore nel caricamento del modello {filename}: {e}")
                models[j, i] = None

    return models
