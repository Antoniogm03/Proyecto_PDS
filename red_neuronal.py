import os
import librosa
import numpy as np
from scipy.spatial.distance import euclidean
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# -------------------- Preprocesado original --------------------

def spectral_subtraction(y, sr, n_fft=2048, hop_length=512, noise_frames=6):
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(D), np.angle(D)
    noise_mag = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
    magnitude_denoised = np.maximum(magnitude - noise_mag, 0.0)
    D_denoised = magnitude_denoised * np.exp(1j * phase)
    return librosa.istft(D_denoised, hop_length=hop_length)

def cepstral_mean_variance_normalization(mfccs):
    mean = np.mean(mfccs, axis=1, keepdims=True)
    std  = np.std(mfccs, axis=1, keepdims=True)
    return (mfccs - mean) / (std + 1e-8)

def compute_vocal_fingerprint(audio_path, sr=16000, num_parameters=13):
    y, sr = librosa.load(audio_path, sr=sr)
    y = spectral_subtraction(y, sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_parameters)
    mfccs = cepstral_mean_variance_normalization(mfccs)
    return np.mean(mfccs, axis=1)

def compare_vocal_fingerprints(x, y, threshold=50):
    distance = np.linalg.norm(x - y)
    return distance < threshold, distance

# -------------------- Dataset y Modelo PyTorch --------------------

class VoiceDataset(Dataset):
    def __init__(self, file_paths, labels, sr=16000, n_mfcc=40, max_duration=10.0):
        self.file_paths = file_paths
        self.labels     = labels
        self.sr         = sr
        self.n_mfcc     = n_mfcc
        self.max_len    = int(sr * max_duration)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path  = self.file_paths[idx]
        label = self.labels[idx]
        y, _  = librosa.load(path, sr=self.sr)
        if len(y) < self.max_len:
            y = np.pad(y, (0, self.max_len - len(y)))
        else:
            y = y[:self.max_len]
        mfcc = librosa.feature.mfcc(y, sr=self.sr, n_mfcc=self.n_mfcc)
        mfcc = cepstral_mean_variance_normalization(mfcc)
        mfcc = torch.tensor(mfcc).unsqueeze(0)  # (1, n_mfcc, T)
        return mfcc.float(), torch.tensor(label, dtype=torch.long)

class SpeakerNet(nn.Module):
    def __init__(self, num_speakers):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(64, num_speakers)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# -------------------- Funciones de entrenamiento e inferencia --------------------

def train_model(data_dir, model_path='speaker_net.pth',
                batch_size=32, lr=1e-3, epochs=20, test_size=0.2):
    # Preparar lista de archivos y etiquetas
    speakers = sorted(os.listdir(data_dir))
    paths, labels = [], []
    for idx, spk in enumerate(speakers):
        for fn in os.listdir(os.path.join(data_dir, spk)):
            if fn.lower().endswith('.wav'):
                paths.append(os.path.join(data_dir, spk, fn))
                labels.append(idx)

    tr_p, vl_p, tr_l, vl_l = train_test_split(
        paths, labels, test_size=test_size, stratify=labels, random_state=42
    )

    train_ds = VoiceDataset(tr_p, tr_l)
    val_ds   = VoiceDataset(vl_p, vl_l)
    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    vl_loader = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = SpeakerNet(len(speakers)).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    crit   = nn.CrossEntropyLoss()

    for epoch in range(1, epochs+1):
        model.train()
        loss_acc = 0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            out  = model(x)
            loss = crit(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_acc += loss.item()
        print(f"Epoch {epoch}/{epochs} — Loss: {loss_acc/len(tr_loader):.4f}")

        # Validación
        model.eval()
        corr, tot = 0, 0
        with torch.no_grad():
            for x, y in vl_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                corr += (preds==y).sum().item()
                tot += y.size(0)
        print(f"  ➔ Val Acc: {corr/tot:.2%}")

    torch.save(model.state_dict(), model_path)
    print(f"Modelo guardado en {model_path}")

def verify_speaker_nn(audio_path, model_path, speakers_list, sr=16000, n_mfcc=40, max_duration=3.0):
    """
    Carga un modelo entrenado y devuelve el locutor más probable para el audio dado.
    speakers_list: lista de nombres de carpetas en el mismo orden usado en entrenamiento.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = SpeakerNet(len(speakers_list)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Procesar audio
    y, _ = librosa.load(audio_path, sr=sr)
    if len(y) < sr * max_duration:
        y = np.pad(y, (0, int(sr*max_duration)-len(y)))
    else:
        y = y[:int(sr*max_duration)]
    mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=n_mfcc)
    mfcc = cepstral_mean_variance_normalization(mfcc)
    mfcc = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        out = model(mfcc)
        pred = out.argmax(dim=1).item()
        probs = torch.softmax(out, dim=1)[0, pred].item()

    return speakers_list[pred], probs

# -------------------- Uso desde línea de comandos --------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Entrena o verifica con ASV NN")
    parser.add_argument('--mode', choices=['train','verify'], required=True)
    parser.add_argument('--data_dir', help="Carpeta raíz con subcarpetas de cada locutor")
    parser.add_argument('--model_path', default='speaker_net.pth')
    parser.add_argument('--audio', help="Ruta .wav a verificar")
    args = parser.parse_args()

    if args.mode == 'train':
        if not args.data_dir:
            raise ValueError("Para entrenar necesitas --data_dir")
        train_model(args.data_dir, model_path=args.model_path)

    elif args.mode == 'verify':
        if not args.audio or not args.data_dir:
            raise ValueError("Para verificar necesitas --audio y --data_dir")
        speakers = sorted(os.listdir(args.data_dir))
        locutor, prob = verify_speaker_nn(args.audio, args.model_path, speakers)
        print(f"Predicción: {locutor} (confianza {prob:.2%})")
