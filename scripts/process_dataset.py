from src.data_loader import DataLoader

speech1_dataset_dev = "./data/1frame_Speech_dataset_dev.npy"
speech2_dataset_dev = "./data/2frame_Speech_dataset_dev.npy"
speech3_dataset_dev = "./data/3frame_Speech_dataset_dev.npy"
speech4_dataset_dev = "./data/4frame_Speech_dataset_dev.npy"
speech8_dataset_dev = "./data/8frame_Speech_dataset_dev.npy"
speech10_dataset_dev = "./data/10frame_Speech_dataset_dev.npy"
speech16_dataset_dev = "./data/16frame_Speech_dataset_dev.npy"
song1_dataset_dev = "./data/1frame_Song_dataset_dev.npy"
song2_dataset_dev = "./data/2frame_Song_dataset_dev.npy"
song3_dataset_dev = "./data/3frame_Song_dataset_dev.npy"
song4_dataset_dev = "./data/4frame_Song_dataset_dev.npy"
song8_dataset_dev = "./data/8frame_Song_dataset_dev.npy"
song10_dataset_dev = "./data/10frame_Song_dataset_dev.npy"
song16_dataset_dev = "./data/16frame_Song_dataset_dev.npy"

speech1_dataset_test = "./data/1frame_Speech_dataset_test.npy"
speech2_dataset_test = "./data/2frame_Speech_dataset_test.npy"
speech3_dataset_test = "./data/3frame_Speech_dataset_test.npy"
speech4_dataset_test = "./data/4frame_Speech_dataset_test.npy"
speech8_dataset_test = "./data/8frame_Speech_dataset_test.npy"
speech10_dataset_test = "./data/10frame_Speech_dataset_test.npy"
speech16_dataset_test = "./data/16frame_Speech_dataset_test.npy"
song1_dataset_test = "./data/1frame_Song_dataset_test.npy"
song2_dataset_test = "./data/2frame_Song_dataset_test.npy"
song3_dataset_test = "./data/3frame_Song_dataset_test.npy"
song4_dataset_test = "./data/4frame_Song_dataset_test.npy"
song8_dataset_test = "./data/8frame_Song_dataset_test.npy"
song10_dataset_test = "./data/10frame_Song_dataset_test.npy"
song16_dataset_test = "./data/16frame_Song_dataset_test.npy"

path_speech_dev = r".\data\data_dev\speech\*\*"
path_song_dev = r".\data\data_dev\song\*\*"
path_speech_test = r"./data/data_test/speech/*/*"
path_song_test = r"./data/data_test/song/*/*"


def main():
    """Procesa los datasets para distinta cantidad de frames, crea un archivo con el contenido y lo guarda en la carpeta data"""
    DL = DataLoader()

    DL.process_dataset(path_song_dev, song1_dataset_dev, 1)
    DL.process_dataset(path_speech_dev, speech1_dataset_dev, 1)
    DL.process_dataset(path_song_test, song1_dataset_test, 1)
    DL.process_dataset(path_speech_test, speech1_dataset_test, 1)

    DL.process_dataset(path_song_dev, song2_dataset_dev, 2)
    DL.process_dataset(path_speech_dev, speech2_dataset_dev, 2)
    DL.process_dataset(path_song_test, song2_dataset_test, 2)
    DL.process_dataset(path_speech_test, speech2_dataset_test, 2)

    DL.process_dataset(path_song_dev, song3_dataset_dev, 3)
    DL.process_dataset(path_speech_dev, speech3_dataset_dev, 3)
    DL.process_dataset(path_song_test, song3_dataset_test, 3)
    DL.process_dataset(path_speech_test, speech3_dataset_test, 3)

    DL.process_dataset(path_song_dev, song4_dataset_dev, 4)
    DL.process_dataset(path_speech_dev, speech4_dataset_dev, 4)
    DL.process_dataset(path_song_test, song4_dataset_test, 4)
    DL.process_dataset(path_speech_test, speech4_dataset_test, 4)

    DL.process_dataset(path_song_dev, song8_dataset_dev, 8)
    DL.process_dataset(path_speech_dev, speech8_dataset_dev, 8)
    DL.process_dataset(path_song_test, song8_dataset_test, 8)
    DL.process_dataset(path_speech_test, speech8_dataset_test, 8)

    DL.process_dataset(path_song_dev, song10_dataset_dev, 10)
    DL.process_dataset(path_speech_dev, speech10_dataset_dev, 10)
    DL.process_dataset(path_song_test, song10_dataset_test, 10)
    DL.process_dataset(path_speech_test, speech10_dataset_test, 10)

    DL.process_dataset(path_song_dev, song16_dataset_dev, 16)
    DL.process_dataset(path_speech_dev, speech16_dataset_dev, 16)
    DL.process_dataset(path_song_test, song16_dataset_test, 16)
    DL.process_dataset(path_speech_test, speech16_dataset_test, 16)



