import librosa
from scipy.spatial.distance import euclidean, cosine
import numpy as np
import soundfile as sf


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

def compute_vocal_fingerprint2(audio_path, sr=16000, num_parameters=13):
    """
    Extrae la huella vocal (MFCC) aplicando reducción de ruido y normalización CMVN.

    Args:
        audio_path (str): Ruta del archivo de audio.
        sr (int): Frecuencia de muestreo.
        num_parameters (int): Número de coeficientes MFCC.

    Returns:
        np.ndarray: Vector de características normalizado.
    """
    print(f'Leyendo archivo wav: {audio_path}')
    # Cargar audio
    # y, sr = librosa.load(audio_path, sr=sr)
    
    x, fs = sf.read(audio_path)
    if fs != sr:
        x = librosa.resample(x, fs, sr)

    # Aplicar reducción básica de ruido (sustracción espectral)
    y_denoised = spectral_subtraction(y, sr)

    # Extraer MFCC del audio filtrado
    mfccs = librosa.feature.mfcc(y=y_denoised, sr=sr, n_mfcc=num_parameters)

    # Aplicar normalización cepstral manual (CMVN)
    mfccs_normalized = cepstral_mean_variance_normalization(mfccs)

    # Retornar la media de los MFCC normalizados
    return np.mean(mfccs_normalized, axis=1)

def compute_vocal_fingerprint_vector_caract(audio_path, sr=16000, num_mfcc=13):
    """
    Extrae un vector de características vocales:
      - MFCC estáticos, delta y delta-delta
      - CMVN
      - Media y desviación típica de cada coeficiente
    """
    # 1. Lectura y remuestreo
    x, fs = sf.read(audio_path)
    if fs != sr:
        x = librosa.resample(x.astype(float), orig_sr=fs, target_sr=sr)
    
    # 2. Reducción del ruido
    x_dn = spectral_subtraction(x, sr)
    
    # 3. MFCC + delta + delta-delta
    mfcc = librosa.feature.mfcc(y=x_dn, sr=sr, n_mfcc=num_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    # 4. Normalización CMVN por tipo
    mfcc  = cepstral_mean_variance_normalization(mfcc)
    delta = cepstral_mean_variance_normalization(delta)
    delta2= cepstral_mean_variance_normalization(delta2)
    
    # 5. Media y desviación típica a lo largo de los frames
    def stats(mfcc):
        media = np.mean(mfcc, axis=1)
        desviacion = np.std(mfcc, axis=1)
        return np.hstack([media, desviacion])
    
    s_mfcc = stats(mfcc)
    s_delta = stats(delta)
    s_delta2 = stats(delta2)
    
    # 6. Construir y devolver vector final
    return np.concatenate([s_mfcc, s_delta, s_delta2])


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


def compare_vocal_fingerprints2(x, y, threshold=50):
    """
    Compara dos huellas vocales usando distancia euclídea.

    Args:
        x (array-like): Primera huella vocal.
        y (array-like): Segunda huella vocal.
        threshold (float): Umbral para determinar coincidencia.

    Returns:
        tuple: (bool, float) indicando coincidencia y distancia.
    """
    distance = np.linalg.norm(x - y)
    return distance < threshold, distance

