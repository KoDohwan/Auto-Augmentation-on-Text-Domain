#!/usr/bin/env python
# coding: utf-8

# In[1]:
import tensorflow as tf
from keras import models, layers, datasets, utils, optimizers, initializers
from keras import backend as K
from transformations import get_transformations
import time
from config import *
from methods import *
from ast import literal_eval


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


# In[2]:


glove = load_pickle(glove_pickle)
tag_list, sentences, train_y = get_tag(tag_path, data_ratio)
test_x, test_y = get_x_y(test_path, num_classes, glove_len, input_size, glove, 1)
transformations = get_transformations()


# In[3]:


LSTM_UNITS = 100

SUBPOLICIES = 4
SUBPOLICY_OPS = 2

OP_TYPES = 20
OP_PROBS = 11

CONTROLLER_EPOCHS = 200
LEARNING_RATE = 1e+5


# In[4]:
def autoaugment(subpolicies, tag_list, X):
    subpolicy = np.random.choice(subpolicies)
    _X = subpolicy(tag_list, X)
    _X = get_x(_X, input_size, glove_len, glove)
    return _X          


# In[5]:


class Operation:
    def __init__(self, types_softmax, probs_softmax, argmax=False):
        if argmax:
            self.type = types_softmax.argmax()
            t = transformations[self.type]
            self.prob = probs_softmax.argmax() / (OP_PROBS-1)
        else:
            print(types_softmax)
            print(probs_softmax)
            self.type = np.random.choice(OP_TYPES, p=types_softmax)
            t = transformations[self.type]
            self.prob = np.random.choice(np.linspace(0, 1, OP_PROBS), p=probs_softmax)
        self.transformation = t

    def __call__(self, tag_list, X):
        _X = []
        for i, x in enumerate(X):
            if np.random.rand() < self.prob:
                x = self.transformation((x, tag_list[i]))
            _X.append(x)
        return _X

    def __str__(self):
        return 'Operation %2d (P=%.3f)' % (self.type, self.prob)


# In[6]:


class Subpolicy:
    def __init__(self, *operations):
        self.operations = operations

    def __call__(self, tag_list, X):
        for op in self.operations:
            X = op(tag_list, X)
        return X

    def __str__(self):
        ret = ''
        for i, op in enumerate(self.operations):
            ret += str(op)
            if i < len(self.operations)-1:
                ret += '\n'
        return ret


# In[7]:


class Controller:
    def __init__(self):
        self.model = self.create_model()
        self.scale = tf.placeholder(tf.float32, ())
        self.grads = tf.gradients(self.model.outputs, self.model.trainable_weights)

        self.grads = [g * (self.scale) for g in self.grads]
        self.grads = zip(self.grads, self.model.trainable_weights)
        self.optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).apply_gradients(self.grads)
 

    def create_model(self):
        input_layer = layers.Input(shape=(SUBPOLICIES, 1))
        init = initializers.RandomUniform(-0.1, 0.1)
        lstm_layer = layers.LSTM(
            LSTM_UNITS, recurrent_initializer=init, return_sequences=True,
            name='controller')(input_layer)
        outputs = []
        for i in range(SUBPOLICY_OPS):
            name = 'op%d-' % (i+1)
            outputs += [
                layers.Dense(OP_TYPES, activation='softmax', name=name + 't')(lstm_layer),
                layers.Dense(OP_PROBS, activation='softmax', name=name + 'p')(lstm_layer),
            ]
        return models.Model(input_layer, outputs)

    def fit(self, mem_softmaxes, mem_accuracies):
        session = K.get_session()
        min_acc = np.min(mem_accuracies)
        max_acc = np.max(mem_accuracies)
        dummy_input = np.ones((1, SUBPOLICIES, 1))
        dict_input = {self.model.input: dummy_input}
        
        for softmaxes, acc in zip(mem_softmaxes, mem_accuracies):
            scale = (acc-min_acc) / (max_acc-min_acc)
            dict_outputs = {_output: s for _output, s in zip(self.model.outputs, softmaxes)}
            dict_scales = {self.scale: scale}
            session.run(self.optimizer, feed_dict={**dict_outputs, **dict_scales, **dict_input})
        return self

    def predict(self, size, argmax=False):
        dummy_input = np.zeros((1, size, 1))
        softmaxes = self.model.predict(dummy_input)
        # convert softmaxes into subpolicies
        subpolicies = []
        for i in range(SUBPOLICIES):
            operations = []
            for j in range(SUBPOLICY_OPS):
                op = softmaxes[j*2:(j+1)*2]
                op = [o[0, i, :] for o in op]
                operations.append(Operation(*op, argmax))
            subpolicies.append(Subpolicy(*operations))
        return softmaxes, subpolicies

    def save(self):
        self.model.save('./model/model.h5')

    def load(self, load_path):
        self.model = models.load_model(load_path)
        self.scale = tf.placeholder(tf.float32, ())
        self.grads = tf.gradients(self.model.outputs, self.model.trainable_weights)
        self.grads = [g * (self.scale) for g in self.grads]
        self.grads = zip(self.grads, self.model.trainable_weights)
        self.optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).apply_gradients(self.grads)


# In[8]:


class Child:
    def __init__(self, input_size, glove_len, num_classes):
        self.model = self.create_model(input_size, glove_len, num_classes)

    def create_model(self, sentence_length, glove_len, num_classes):
        if model_type == 'cnn':
            adam = Adam(lr = 0.001)
            model = None
            model = Sequential()
            model.add(layers.Conv1D(128, 5, activation='relu', input_shape=(sentence_length, glove_len)))
            model.add(layers.GlobalMaxPooling1D())
            model.add(Dense(20, activation='relu'))
            model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
            return model
        elif model_type == 'rnn':
            model = None
            model = Sequential()
            model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(sentence_length, glove_len)))
            model.add(Dropout(0.5))
            model.add(Bidirectional(LSTM(32, return_sequences=False)))
            model.add(Dropout(0.5))
            model.add(Dense(20, activation='relu'))
            model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model 
        else:
            print('model error!!!')
            return None

    def fit(self, subpolicies, tag_list, X, y):
        _X = autoaugment(subpolicies, tag_list, X)
        self.model.fit(_X, y, epochs=1, batch_size=1024, shuffle=True, verbose=1)
        return self

    def evaluate(self, X, y):
        return self.model.evaluate(X, y, verbose=0)


# In[9]:


def find_best_policy(controller, best_results):
    print()
    print('Best policies found:')
    print()
    best_policies = []
    _, subpolicies = controller.predict(SUBPOLICIES, argmax=False)
    best_policies += subpolicies
    for i, subpolicy in enumerate(subpolicies):
        print('# Subpolicy %d' % (i+1))
        print(subpolicy)

    best_model = Child(input_size, glove_len, num_classes)
    best_model.fit(best_policies, tag_list, sentences, train_y)
    best_result = best_model.evaluate(test_x, test_y)
    best_results.append(best_result)
    
    with open('./model/best.txt', 'w') as f:
        for res in best_results:
            f.write(str(res[0]) + ', ' + str(res[1]) + '\n')
    print('Best loss: %.5f, Best accuacy: %.3f' % (best_result[0], best_result[1]))


# In[ ]:


# mem_softmaxes = []
# mem_accuracies = []
# mem_results = []
# best_results = []

# controller = Controller()

# for epoch in range(CONTROLLER_EPOCHS):
#     print('Controller: Epoch %d / %d' % (epoch+1, CONTROLLER_EPOCHS))

#     softmaxes, subpolicies = controller.predict(SUBPOLICIES)
#     for i, subpolicy in enumerate(subpolicies):
#         print('# Sub-policy %d' % (i+1))
#         print(subpolicy)
#     mem_softmaxes.append(softmaxes)

#     print('\nChild: ', end='')
#     child = Child(input_size, glove_len, num_classes)
#     start = time.time()
#     child.fit(subpolicies, tag_list, sentences, train_y)
#     end = time.time()
#     result = child.evaluate(test_x, test_y)
#     print('-> Child loss: %.5f, Child accuracy: %.3f (elaspsed time: %ds)' % (result[0], result[1], (end - start)))
#     mem_accuracies.append(result[0])
#     mem_results.append(result)

#     if len(mem_softmaxes) > 1:
#         controller.fit(mem_softmaxes, mem_accuracies)

#     controller.save()
#     with open('./model/results.txt', 'w') as f:
#         for res in mem_results:
#             f.write(str(res[0]) + ', ' + str(res[1]) + '\n')
    
    
            
#     if (epoch + 1) % 10 == 0:
#         find_best_policy(controller, best_results)
#     print()

controller = Controller()
controller.load('./results/trec_rnn/model.h5')
find_best_policy(controller, [])