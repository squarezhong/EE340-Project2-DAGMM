import os
import torch
from train import Trainer

hyp={
	'input_dim':33,
	'hidden_dim1':21, # 32
	'hidden_dim2':10, # 16
	'hidden_dim3':8, # None
	'zc_dim':1,
	'n_gmm':2,
	'dropout':0.5,
	'lr':1e-4,
	'batch_size':128,
	'epochs': 30,
    'decay_rate': 0.96,
    'is_threshold': False,
    'threshold': 8.8982,
	}

if __name__ == "__main__":
	torch.manual_seed(11)
	os.mkdir('./data/') if not os.path.isdir('./data/') else None
	os.mkdir('./results/') if not os.path.isdir('./results/') else None
	trainer = Trainer(hyp)
	#trainer.train(hyp)
	trainer.test(hyp)

	
        