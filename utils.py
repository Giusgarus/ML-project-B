import numpy as np
from sklearn.preprocessing import StandardScaler
import itertools

# Funzione per normalizzare i dati
def normalize(data):
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data


def split_data(data, labels, k):
    """
    Divide i dati in k fold.
    
    Args:
        data (np.ndarray | pd.DataFrame): Dati di input.
        labels (np.ndarray | pd.Series): Etichette.
        k (int): Numero di fold.
    
    Returns:
        list: Lista di tuple (fold_data, fold_labels).
    """
    if not isinstance(data, np.ndarray):
        data = data.to_numpy()  # Converte DataFrame in NumPy array
    if not isinstance(labels, np.ndarray):
        labels = labels.to_numpy()  # Converte Series in NumPy array
    
    data = normalize(data)

    fold_size = len(data) // k
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    folds = []

    for i in range(k):
        fold_indices = indices[i * fold_size: (i + 1) * fold_size]
        fold_data = data[fold_indices]
        fold_labels = labels[fold_indices]
        folds.append((fold_data, fold_labels))
    
    return folds


def evaluate_model(model, x_test, y_test):
    """
    Valuta le prestazioni di un modello sui dati di test.
    
    Args:
        model: Modello addestrato.
        x_test (np.ndarray): Dati di test.
        y_test (np.ndarray): Etichette di test.
    
    Returns:
        float: Accuratezza del modello.
    """
    y_pred = model.predict(x_test)
    print(y_test)
    print(y_pred)
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    return accuracy



def generate_hyperparameter_combinations(param_ranges):
    """
    Genera tutte le combinazioni di iperparametri basate su range e step specificati.

    :param param_ranges: Dizionario con i nomi degli iperparametri come chiavi.
                         Ogni valore è una tupla (start, stop, step).
    :return: Lista di dizionari con tutte le combinazioni possibili.
    """
    param_values = {
        key: np.arange(start, stop + step, step)
        for key, (start, stop, step) in param_ranges.items()
    }
    
    param_combinations = list(itertools.product(*param_values.values()))
    return [
        dict(zip(param_values.keys(), combination))
        for combination in param_combinations
    ]
