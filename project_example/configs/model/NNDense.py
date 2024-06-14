name = 'Dense'
device='cuda'
input_dim = None
output_dim = 7
learning_rate = 1e-3
early_stop_patience = 4
max_epochs = 100
tboard = True
dataloader_params ={
        'batch_size' : 32,#'persistent_workers' : True,
        'num_workers': 6,
        'shuffle': True}

