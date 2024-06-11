import opensmile
from glob import glob


def load_dataset(config_file):
    files = glob(r'data/Audio_Speech_Actors_01-24/Actor_01/*.wav')
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02, # en el config, aclaramos qué va acá
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    return smile.process_files(files)

