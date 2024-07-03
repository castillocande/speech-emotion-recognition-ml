import os
import librosa
import numpy as np
import matplotlib.pyplot as plt


emotion_labels = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


def extract_spectrogram(audio_path, sr=22050, n_mels=224, fmax=8000):
    y, sr = librosa.load(audio_path, sr=sr)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return log_spectrogram


def save_spectrogram(spectrogram, save_path):
    plt.figure(figsize=(5, 5))
    librosa.display.specshow(spectrogram, sr=22050, x_axis="time", y_axis="mel", fmax=8000)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def process_files(files, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for audio_path in files:
        filename = os.path.basename(audio_path)
        emotion_code = filename.split('-')[2] 
        emotion = emotion_labels.get(emotion_code, "unknown")
        class_dir = os.path.join(output_path, emotion)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        spectrogram = extract_spectrogram(audio_path)
        save_path = os.path.join(class_dir, filename.replace(".wav", ".png"))
        save_spectrogram(spectrogram, save_path)


if __name__ == "__main__":
    pass