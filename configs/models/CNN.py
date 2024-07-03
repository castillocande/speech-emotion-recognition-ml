name = "CNN"
type = "NN"
device = "cuda"
input_dim = (224, 224, 3)
learning_rate = 0.0001 
momentum = 0.8
epochs = 50
# dataloader_params = {
#         "batch_size" : 32,
#         "num_workers": 6,
#         "shuffle": True}
batch_size = 32