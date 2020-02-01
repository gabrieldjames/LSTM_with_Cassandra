#format text data, vectorize, train or test and save or don't save LSTM
import pandas as pd
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks.callbacks import ModelCheckpoint
from keras.models import load_model
import random
import sys
import numpy as np



def series_to_words(s):
	text = ''
	for i in range(len(s)):
		text = text + ' ' + s.iloc[i]

	text_words = [w for w in text.split(' ') if w.strip()!='']
	return text_words


def drop_low_freq_words(text_words, min_word_freq):
	word_freq = {}
	for word in text_words:
		word_freq[word] = word_freq.get(word, 0) + 1

	ignored_words = set()
	for i, j in word_freq.items():
		if word_freq[i] < min_word_freq:
			ignored_words.add(i)

	keep_words = set(text_words)
	keep_words = sorted(keep_words - ignored_words)

	return keep_words, ignored_words


def vectorize_sentence(text_words, keep_words, ignored_words, maxlen, step):
	sentences = []
	next_words = []

	word_indices = dict((c, i) for i, c in enumerate(keep_words))
	indices_word = dict((i, c) for i, c in enumerate(keep_words))

	for i in range(0, len(text_words) - maxlen, step):
		if len(set(text_words[i: i + maxlen+1]).intersection(ignored_words))==0:
			sentences.append(text_words[i: i + maxlen])
			next_words.append(text_words[i + maxlen])


	x = np.zeros((len(sentences), maxlen, len(keep_words)), dtype=np.bool)
	y = np.zeros((len(sentences), len(keep_words)), dtype=np.bool)
	for i, sentence in enumerate(sentences):
		for t, keep_word in enumerate(sentence):
			x[i, t, word_indices[keep_word]] = 1
		y[i, word_indices[next_words[i]]] = 1

	return x, y, sentences, word_indices, indices_word


def train_model(x, y, epoch_num, save):
	model = load_model('bot.h5')

	if(save):
		mc = ModelCheckpoint('bot.h5', monitor='val_accuracy', mode = 'min')
		callbacks_list = [mc]

		model.fit(x, y, batch_size = 128, epochs = epoch_num, callbacks=callbacks_list)
	else:
		model.fit(x, y, batch_size = 128, epochs = epoch_num)

	return model


def sample(preds, temperature=1.0):
	# helper function to sample an index from a probability array
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)


def generate_comment(model, sentences, keep_words, maxlen, word_indices, indices_word):

	new_comment = ''

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

		new_comment += next_word + ' '
		sys.stdout.write(next_word+' ')
		sys.stdout.flush()
	print()

	return new_comment, generated


def run_lstm(df, min_word_freq, maxlen, step, epoch_num, train, save = False):
	text_words = series_to_words(df['comment'])
	keep_words, ignored_words = drop_low_freq_words(text_words, min_word_freq)

	x, y, sentences, word_indices, indices_word = vectorize_sentence(text_words, keep_words, ignored_words, maxlen, step)
	if(train):
		model = train_model(x, y, epoch_num, save)
	else:
		model = load_model('bot.h5')

	new_comment, generated = generate_comment(model, sentences, keep_words, maxlen, word_indices, indices_word)

	return new_comment, generated
