from src.pipeline import run_experiment
# from src.data_loader import load_data
from src.files import read_configs

def main(): # recibe los configs como parámetro
    # tenemos que hacer que, cada vez que agregamos una config, 
    # se nos debería agregar sola una carpeta con los resultados. 
    # esto lo deberíamos hacer e el main. 

    # recibe los nombres de los archivos?
    # dataset = load_data(config_file=r"configs\data\[el nombre del .py]")
    # features = get_features(dataset, config_file=r"configs\features") 
    # con open smile, obtenemos directamente los features, no solo el dataset. cómo hacemos eso?
    model, data, features = read_configs()
    # al medio defino el logger
    run_experiment(model, data, features) #paths, configuraciones del modelo y de las features que quiero extraer

if __name__ == "__main__":
    main()