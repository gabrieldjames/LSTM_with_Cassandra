import pandas as pd
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks.callbacks import ModelCheckpoint
import random
import sys
import numpy as np



def series_to_words(s):
	text = ''
	for i in range(len(s)):
		text = text + ' ' + s.iloc[i]

	text_words = [w for w in text.split(' ') if w.strip()!='']
	return text_words


def drop_low_freq_words(text_words, min_word_freq, verbose = False):
	word_freq = {}
	for word in text_words:
		word_freq[word] = word_freq.get(word, 0) + 1

	ignored_words = set()
	for i, j in word_freq.items():
		if word_freq[i] < min_word_freq:
			ignored_words.add(i)

	keep_words = set(text_words)
	og_word_length = len(keep_words)

	keep_words = sorted(keep_words - ignored_words)

	if(verbose):
		print()
		print('Unique words before ignoring:', og_word_length)
		print('Ignoring words with occurences <', min_word_freq)
		print('Unique words after ignoring:', len(keep_words))


	return keep_words, ignored_words





def sample(preds, temperature=1.0):
	# helper function to sample an index from a probability array
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)


def generate_comment():

	print()

	index = random.randint(0, len(sentences)-1)
	
	sentence = sentences[index]
	generated = ' '.join(sentence) + ' '

	print('----- Generating with seed:"' + generated + '"')
	sys.stdout.write(generated)

	for i in range(100):
		x_pred = np.zeros((1, maxlen, len(keep_words)))
		for t, word in enumerate(sentence):
			x_pred[0, t, word_indices[word]] = 1

		preds = model.predict(x_pred, verbose=0)[0]
		next_index = sample(preds, 0.5)
		next_word = indices_word[next_index]

		sentence = sentence[1:]
		sentence.append(next_word)

		sys.stdout.write(next_word+' ')
		sys.stdout.flush()
	print()





min_word_freq = 5
maxlen = 6
step = 2
num_units = 256
activ_func = 'softmax'
learn_rate = 0.01
epoch_num = 5




df = pd.read_csv('init_data.csv')
text_words = series_to_words(df['comment'])
keep_words, ignored_words = drop_low_freq_words(text_words, min_word_freq, True)


sentences = []
next_words = []
ignored = 0

word_indices = dict((c, i) for i, c in enumerate(keep_words))
indices_word = dict((i, c) for i, c in enumerate(keep_words))

for i in range(0, len(text_words) - maxlen, step):
	if len(set(text_words[i: i + maxlen+1]).intersection(ignored_words))==0:
		sentences.append(text_words[i: i + maxlen])
		next_words.append(text_words[i + maxlen])
	else:
		ignored = ignored + 1

print()
print('Ignored sequences:', ignored)
print('Remaining sequences', len(sentences))


x = np.zeros((len(sentences), maxlen, len(keep_words)), dtype=np.bool)
y = np.zeros((len(sentences), len(keep_words)), dtype=np.bool)
for i, sentence in enumerate(sentences):
	for t, keep_word in enumerate(sentence):
		x[i, t, word_indices[keep_word]] = 1
	y[i, word_indices[next_words[i]]] = 1



model = Sequential()
model.add(LSTM(num_units, input_shape=(maxlen, len(keep_words))))
model.add(Dense(len(keep_words), activation= activ_func))

optimizer = RMSprop(learning_rate = learn_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

mc = ModelCheckpoint('bot.h5', monitor='val_accuracy', mode = 'min')

callbacks_list = [mc]


model.fit(x, y, batch_size = 128, epochs = epoch_num, callbacks=callbacks_list)

generate_comment()
