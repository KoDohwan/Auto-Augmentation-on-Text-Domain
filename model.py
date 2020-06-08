from methods import *
from config import *
from numpy.random import seed
seed(0)


def run_model(train_file, test_file, num_classes, input_size, data_ratio, glove):

	#initialize model
	if model_type == 'cnn':
		model = cnn(input_size, glove_len, num_classes)
	elif model_type == 'rnn':
		model = rnn(input_size, glove_len, num_classes)
	else:
		print('Model Error!!!')

	#load data
	train_x, train_y = get_x_y(train_file, num_classes, glove_len, input_size, glove, data_ratio)
	test_x, test_y = get_x_y(test_file, num_classes, glove_len, input_size, glove, 1)


	#train model
	model.fit(	train_x, 
				train_y, 
				epochs=1, 
				validation_split=0.1, 
				batch_size=1024, 
				shuffle=True, 
				verbose=1)
	#model.save('checkpoints/lol')
	#model = load_model('checkpoints/lol')

	
	res = model.evaluate(test_x, test_y)


	train_x, train_y, model = None, None, None
	gc.collect()

	return res

def compute_baselines():
	glove = load_pickle(glove_pickle)
	res = run_model(train_path, test_path, num_classes, input_size, data_ratio, glove)
	return res


if __name__ == "__main__":
	# gen_vocab_dicts('./data/trec', glove_pickle, huge_glove)
	
	print(compute_baselines())
