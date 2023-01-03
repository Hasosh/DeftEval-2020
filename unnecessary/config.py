#this class contains all relevant configuration parameters for training the underlying BERT models
class Config(object):
	dev = True

	data_path = "test.csv"
	use_crf = True

	# model / optimizer parameters
	bert_model: str = 'bert-base-cased' #all considered models: #bert-base-multilingual-cased #allenai/scibert_scivocab_cased #bert-base-cased #bert-base-german-cased 
	full_finetuning: bool = True # toggle if all hyperparameters are fine-tuned. if False only parameters of classifier are fine-tuned
	epochs: int = 4
	sequence_length: int = 128 # sequence length of the bert model
	batch_size: int = 4 # batch size used during training / testing
	hidden_dropout_prob: float = 0.1 # hidden dropout probability, BERT default is 0.1
	lr: float = 5e-5 # lerning rate, best feeling: 4.5e-5 or 5e-5, best values were between 1e-5 and 6e-5
	eps: float = 1e-8 # Adams epsilon parameter for numerical stability
	warmup_steps: int = 0 # warm up steps (not investigated any numbers but 0)
	max_grad_norm: float = 1.0 # clip the norm of the gradient

	update_step = 100

if __name__ == "__main__":
	print("call location:", __name__)
	print("this is just the config file, nothing to execute. Only specify parameters for all models (e.g. Extraction_model.py)") #TODO adapt