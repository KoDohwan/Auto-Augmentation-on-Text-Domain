from os.path import join

# config data
data = 'cr'                         # Change the data what will be used. (cr, mr, sst1, sst2, subj, trec)
raw_data = 'raw_' + data + '.txt'
all_data = data + '.txt'
train_data = data + '_train.txt'
tag_data = data + '_tag.txt'
test_data = data + '_test.txt'
input_size = 50                     # Change if you need
data_ratio = 1.0                    # Change the ratio of data
num_classes = 0
if data == 'cr' or data == 'mr' or data == 'sst2' or data == 'subj':
    num_classes = 2
elif data == 'trec':
    num_classes = 6
elif data == 'sst1':
    num_classes = 5
else:
    print('Data Error!!!')


# config glove
glove_len = 300 
glove_pickle = join('./data', data, 'glove.p')
huge_glove = './glove/glove.840B.300d.txt'

# config model training
model_type = 'cnn'                  # Change the model what will be used. (cnn, rnn) 


# config data path
raw_path = join('./data', data, raw_data)
all_path = join('./data', data, all_data)
train_path = join('./data', data, train_data)
test_path = join('./data', data, test_data)
datafolder = join('./data', data)

# config augmentation
orig_path = join('./data', data, train_data)
tag_path = join('./data', data, tag_data)
