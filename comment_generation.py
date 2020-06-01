#format text data, vectorize, train or test and save or don't save LSTM
import pandas as pd
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks.callbacks import ModelCheckpoint
from keras.models import load_model
import random
import sys
import numpy as np



#convert string to list of words by splitting on ' ', used with .apply()
#s: type String
#returns: list of Strings (words)

def series_to_chars(s):
	text = ''
	for i in range(len(s)):
		text = text + ' ' + s.iloc[i]

	text_chars = [w for w in text]

	return text, text_chars



#represent characters of text as 1 or 0 in numpy arrays
#text_chars: type list, all characters to be modeled on
#maxlen: type int, max length of comment
#step: type int, when training model, how many words skipped over to limit repeat of sequences
#returns:
	#x: type numpy array
	#y: type numpy array
	#chars: type list
	#sentences: type list
	#char_indices: type dictionary
	#indices_char: type dictionary

def vectorize_sentence(text_chars, maxlen, step):
	chars = sorted(list(set(text_chars)))
	sentences = []
	next_chars = []

	df_dict = pd.read_csv('reddit_chars.csv')
	df_dict['characters'] = df_dict['characters'].astype(str)
	df_dict['encoding'] = df_dict['encoding'].astype(int)

	char_indices = dict(zip(df_dict['characters'], df_dict['encoding']))
	indices_char = dict(zip(df_dict['encoding'], df_dict['characters']))

	for i in range(0, len(text_chars) - maxlen, step):
		sentences.append(text_chars[i: i + maxlen])
		next_chars.append(text_chars[i + maxlen])


	x = np.zeros((len(sentences), maxlen, len(char_indices)), dtype=np.bool)
	y = np.zeros((len(sentences), len(char_indices)), dtype=np.bool)
	for i, sentence in enumerate(sentences):
		for t, char in enumerate(sentence):
			if(char in char_indices):
				x[i, t, char_indices[char]] = 1
			else:
				x[i, t, char_indices['unknown']] = 1
		if(next_chars[i] in char_indices):
			y[i, char_indices[next_chars[i]]] = 1
		else:
			y[i, char_indices['unknown']] = 1

	return x, y, chars, sentences, char_indices, indices_char


#trains LSTM
#x: type numpy array, X featureset
#y: type numpy array, y
#epoch_num: type int, number of epochs to train
#save: type boolean, whether or not to save model
#maxlen: type int, max length of comment
#char_indices: type dictionary, integer encoding of characters
#returns: type Keras model (LSTM)

def train_model(x, y, epoch_num, save, maxlen, char_indices):
	#model = load_model('bot.h5')

	model = Sequential()
	model.add(LSTM(256, input_shape=(maxlen, len(char_indices)), return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(128, return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(128, return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(64))
	model.add(Dropout(0.2))
	model.add(Dense(len(char_indices), activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer = 'adam')

	if(save):
		#save model
		mc = ModelCheckpoint('bot.h5', monitor='val_accuracy', mode = 'min')
		callbacks_list = [mc]

		model.fit(x, y, batch_size = 32, epochs = epoch_num, callbacks=callbacks_list, verbose = 1)
	else:
		model.fit(x, y, batch_size = 32, epochs = epoch_num, verbose = 1)

	return model



# helper function to sample an index from a probability array

def sample(preds, temperature=1.0):
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)



#generates new comment
#text: type String
#maxlen: type int, max length of comment
#chars: type list, all characters modeled on
#model: Keras model
#char_indices: type dictionary, integer encoding of characters (char -> int)
#indices_char: type dictionary, character encoding of integers (int -> char)
#returns:
	#new_comment: type String
	#generated: type String, seed for comment generation

def generate_comment(text, maxlen, chars, model, sentences, char_indices, indices_char):

	new_comment = ''
	generated = ''

	start_index = random.randint(0, len(text)-maxlen-1)
	sentence = text[start_index: start_index+maxlen]
	generated += sentence
	print('----- Generating with seed:"' + generated + '"')
	sys.stdout.write(generated)

	for i in range(600):
		x_pred = np.zeros((1, maxlen, len(char_indices)))
		for t, char in enumerate(sentence):
			x_pred[0, t, char_indices[char]] = 1

		preds = model.predict(x_pred, verbose=0)[0]
		next_index = sample(preds, 0.5)
		next_char = indices_char[next_index]

		sentence = sentence[1:] + next_char

		new_comment += next_char
		sys.stdout.write(next_char)
		sys.stdout.flush()
	print()

	return new_comment, generated



#acts as main method, to be called in another script
#df: type DataFrame
#maxlen: type int, max length of comment
#step: type int, number of words skipped over when adding a new set of words for training
#epoch_num: number of epochs to train model with
#save: type boolean, whether or not model is saved as .h5
#returns:
	#new_comment: String, new comment generated
	#generated: String, seed for generated comment

def run_lstm(df, maxlen, step, epoch_num, train, save = False):
	text, text_chars = series_to_chars(df['comment'])

	x, y, chars, sentences, char_indices, indices_char = vectorize_sentence(text_chars, maxlen, step)
	if(train):
		model = train_model(x, y, epoch_num, save, maxlen, char_indices)
	else:
		model = load_model('bot.h5')

	new_comment, generated = generate_comment(text, maxlen, chars, model, sentences, char_indices, indices_char)

	return new_comment, generated