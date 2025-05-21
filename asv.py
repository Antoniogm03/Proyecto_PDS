import librosa
from scipy.spatial.distance import euclidean, cosine
import numpy as np
from scipy.stats import skew, kurtosis


def compute_vocal_fingerprint(audio_path, sr=16000, num_parameters=13):
    """
    Extrae la huella vocal de un audio mediante los coeficientes MFCC.

    Args:
        audio_path (str): Ruta del archivo de audio.
        sr (int, opcional): Frecuencia de muestreo a la que se cargará el audio. 
                            Por defecto es 16 kHz, una frecuencia común en biometría de voz.
        num_parameters (int, opcional): Número de coeficientes MFCC a extraer. 
                                        Por defecto es 13, un valor estándar en procesamiento de voz.

    Returns:
        np.ndarray: Vector de características obtenido al calcular la media de cada coeficiente MFCC.

    Nota:
        - Se pueden experimentar distintos valores de `num_parameters` para mejorar la discriminación entre usuarios.
        - Se podrían usar estadísticas adicionales como la varianza en lugar de solo la media.
    """
    # Cargar audio con la frecuencia de muestreo especificada
    y, sr = librosa.load(audio_path, sr=sr)  
    
    # Extraer los coeficientes MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_parameters)  
    
    # Calcular la media de cada coeficiente para obtener un vector representativo
    return np.mean(mfccs, axis=1)  

# Metodo para calcular usando desviación estándar, asimetría y kurtosis
def compute_vocal_fingerprint_desviacion_estandar(audio_path, sr=16000, num_parameters=13):
    """
    Extrae una huella vocal enriquecida con estadísticas de los coeficientes MFCC:
    media, desviación estándar, skewness y kurtosis.
    skewness y kurtosis son medidas de asimetría y apuntamiento de la distribución.
    skewness mide la "asimetría" de la distribución.
    kurtosis mide la "altura" de la distribución.

    Returns:
        np.ndarray: Vector de características enriquecido (4 × num_parameters).
    """
    # Cargar audio
    y, sr = librosa.load(audio_path, sr=sr)

    # Extraer MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_parameters)

    # Calcular estadísticas
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    mfcc_skew = skew(mfccs, axis=1)
    mfcc_kurt = kurtosis(mfccs, axis=1)

    # Concatenar todas en un solo vector
    fingerprint = np.concatenate([mfcc_mean, mfcc_std, mfcc_skew, mfcc_kurt])

    return fingerprint
# Metodo para calcular usando derivadas, velocidad y aceleración
def compute_vocal_fingerprint_deltas(audio_path, sr=16000, n_mfcc=13):
    """
    Extrae una huella vocal basada en MFCC + delta + delta-delta (media de cada uno).
    Utiliza la primera derivada para calcular la velocidad del que habla y la seguda derivada para ver la aceleración vocal.
    Args:
        audio_path (str): Ruta al archivo de audio.
        sr (int): Frecuencia de muestreo.
        n_mfcc (int): Número de coeficientes MFCC.

    Returns:
        np.ndarray: Vector de características concatenadas.
    """
    y, sr = librosa.load(audio_path, sr=sr)
    
    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Delta y Delta-Delta
    delta = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Calcular la media
    mfcc_media = np.mean(mfcc, axis=1)
    delta_media = np.mean(delta, axis=1)
    delta2_media = np.mean(delta2, axis=1)

    # Concatenar todo en un solo vector
    fingerprint = np.concatenate([mfcc_media, delta_media, delta2_media])
    return fingerprint


def compute_vocal_fingerprint_nota_musical(audio_path, sr=16000):
    """
    Extrae una huella vocal simple basada en el promedio por nota musical.

    Args:
        audio_path (str): Ruta al archivo de audio.
        sr (int): Frecuencia de muestreo.

    Returns:
        np.ndarray: Vector de 12 dimensiones (una por nota musical).
    """
    y, sr = librosa.load(audio_path, sr=sr)

    # Extraer nota musical 
    nota = librosa.feature.chroma_stft(y=y, sr=sr)

    # Calcular promedio por semitono
    notas_media = np.mean(nota, axis=1)

    return chroma_mean


def compare_vocal_fingerprints(x, y, threshold=100):
    """
    Compara dos huellas vocales utilizando la distancia euclídea.

    Args:
        x (array-like): Primera huella vocal.
        y (array-like): Segunda huella vocal.
        threshold (float, opcional): Umbral de decisión para determinar si pertenecen al mismo usuario. 
                                     Valores más bajos hacen la verificación más estricta. 
                                     Por defecto es 50.

    Returns:
        tuple: 
            - bool: True si la distancia es menor que el umbral (misma persona), False en caso contrario.
            - float: Valor de la distancia euclídea calculada.
    
    Nota:
        - Se puede experimentar con distintos valores del umbral.
        - También es posible probar otras métricas de distancia como la coseno o Manhattan.
    """
    distance = euclidean(x, y)
    return distance < threshold, distance
