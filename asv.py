import librosa
from scipy.spatial.distance import euclidean, cosine
import numpy as np
import soundfile as sf
from scipy.stats import pearsonr


def spectral_subtraction(y, sr, n_fft=2048, hop_length=512, noise_frames=6):
    """
    Realiza una reducción básica del ruido mediante sustracción espectral.

    Args:
        y (np.array): Señal de audio.
        sr (int): Frecuencia de muestreo.
        n_fft (int): Tamaño de la ventana FFT.
        hop_length (int): Longitud de salto entre ventanas.
        noise_frames (int): Número inicial de frames considerados ruido.

    Returns:
        np.array: Señal filtrada.
    """
    # Espectrograma del audio
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(D), np.angle(D)

    # Estimar espectro del ruido a partir de los primeros frames
    noise_mag = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

    # Sustracción espectral (asegura valores positivos)
    magnitude_denoised = np.maximum(magnitude - noise_mag, 0.0)

    # Reconstruir señal filtrada
    D_denoised = magnitude_denoised * np.exp(1j * phase)
    y_denoised = librosa.istft(D_denoised, hop_length=hop_length)

    return y_denoised


def cepstral_mean_variance_normalization(mfccs):
    """
    Normalización cepstral media-varianza manual (CMVN).

    Args:
        mfccs (np.array): Matriz de coeficientes MFCC.

    Returns:
        np.array: MFCC normalizados.
    """
    mean = np.mean(mfccs, axis=1, keepdims=True)
    std_dev = np.std(mfccs, axis=1, keepdims=True)
    normalized_mfccs = (mfccs - mean) / (std_dev + 1e-8)

    return normalized_mfccs


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

def compute_vocal_fingerprint_vector(audio_path, sr=16000, n_mfcc=13):
    """
    Extrae vector de características vocales (MFCC estáticos, delta y delta-delta)
    y devuelve la concatenación de medias y desviaciones típicas de cada coeficiente.
    """
    # 1. Leer y re-muestrear
    y, orig_sr = sf.read(audio_path)
    if orig_sr != sr:
        y = librosa.resample(y.astype(float), orig_sr, sr)

    # 2. Convertir a mono
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # 3. Reducción de ruido (asumiendo que spectral_subtraction está vectorizada)
    y = spectral_subtraction(y, sr)

    # 4. Calcular MFCC + delta + delta-delta
    #    Resultado shape = (3*n_mfcc, n_frames)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feats = np.vstack((mfcc, delta, delta2))

    # 5. Calcular medias y desviaciones por coeficiente (vectorizado)
    mean = feats.mean(axis=1)
    std  = feats.std(axis=1)

    # 6. Devolver vector final [means | stds]
    return np.concatenate((mean, std))


def compare_vocal_fingerprints(x, y, threshold=100):
    """
    Compara dos huellas vocales utilizando la distancia euclídea.

    Args:
        x: Primera huella vocal.
        y: Segunda huella vocal.
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

def compare_vocal_fingerprints_vector(fp1, fp2, threshold=0.9):
    """
    Compara dos vectores de huella vocal y devuelve:
      sim la similitud de coseno entre fp1 y fp2 (float en [0,1])
      is_same bool, True si sim >= threshold (mismo hablante)

    Parámetros:
      fp1, fp2 : array-like de forma idéntica (p.ej. salida de compute_vocal_fingerprint_vector)
      threshold: umbral de similitud (por defecto 0.9)
    Retorna:
      sim, is_same
    """
    # Convertir a array float64
    v1 = np.asarray(fp1, dtype=np.float64)
    v2 = np.asarray(fp2, dtype=np.float64)

    # Comprobar dimensiones
    if v1.shape != v2.shape:
        raise ValueError("Los vectores deben tener la misma forma, "
                         f"pero recibí {v1.shape} y {v2.shape}")

    # Calcular norma y proteger contra división por cero
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        raise ValueError("Norma de alguno de los vectores casi cero; huella inválida")

    # Similitud de coseno
    sim = np.dot(v1, v2) / (norm1 * norm2)

    # Clasificación
    is_same = (sim >= threshold)
    return sim, is_same

def compare_vocal_fingerprints_coseno(x, y, threshold=0.2):
    """
    Compara dos huellas vocales utilizando la distancia del coseno.

    Args:
        x: Primera huella vocal.
        y: Segunda huella vocal.
        threshold (float, opcional): Umbral de decisión para determinar si pertenecen al mismo usuario. 
        Valores más bajos hacen la verificación más estricta. 
        Por defecto es 0.2.

    Returns:
        tuple: 
            - bool: True si la distancia es menor que el umbral (misma persona), False en caso contrario.
            - float: Valor de la distancia del coseno calculada.
    """
    
    distance = cosine(x, y)
    return distance < threshold, distance

def compare_vocal_fingerprints_manhattan(x, y, threshold=10):
    """
    Compara dos huellas vocales usando la distancia Manhattan (L1).

    Args:
        x: Primera huella vocal.
        y: Segunda huella vocal.
        threshold: Umbral de decisión. 
        Por defecto es 10.

    Returns:
        tuple:
            - bool: True si la distancia es menor que el umbral, False en caso contrario.
            - float: Distancia Manhattan calculada.
    """
    distance = np.sum(np.abs(np.array(x) - np.array(y)))
    return distance < threshold, distance

def compare_vocal_fingerprints_pearson(x, y, threshold=0.85):
    """
    Compara dos huellas vocales usando la correlación de Pearson.

    Args:
        x: Primera huella vocal.
        y: Segunda huella vocal.
        threshold: Umbral de correlación. Si la correlación es mayor que el umbral,
        se considera la misma persona. Por defecto es 0.85.

    Returns:
        tuple:
            - bool: True si la correlación es mayor que el umbral, False en caso contrario.
            - float: Valor de la correlación de Pearson.
    """
    
    correlation, _ = pearsonr(x, y)
    return correlation > threshold, correlation