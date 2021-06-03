# lemmatize all tokens -- remove ending from words
from nltk.corpus import wordnet as wnet

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'): return wnet.ADJ
    elif treebank_tag.startswith('V'): return wnet.VERB
    elif treebank_tag.startswith('N'): return wnet.NOUN
    elif treebank_tag.startswith('R'): return wnet.ADV
    else: return nltk.wordnet.NOUN
    
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
lemmatized_tokens = { p : [lemmatizer.lemmatize(t, pos = get_wordnet_pos(t))
                      for t in tokensBySpeaker[p]] 
                      for p in tokensBySpeaker }

tokenCounts = { p : {t.title() : lemmatized_tokens[p].count(t)}
               for t in list(set(tokensBySpeaker[p]))
               for p in tokensBySpeaker }




# split tokens by POS
frequencyByPOS = { pos : [] for pos in [wnet.ADJ, wnet.VERB, wnet.NOUN, wnet.ADV] }

print(nltk.pos_tag('vehicle'))

for token in lemmatized_tokens['all']:
    pos = get_wordnet_pos(nltk.pos_tag(token)[0][1])
    frequencyByPOS[pos].append(token)

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding

model = Sequential()

num_words = len(tokens['Brian Francis'])

glove_vectors = '/Users/brianfrancis/Downloads/glove.6B.100d.txt'
glove = np.loadtxt(glove_vectors, dtype = 'str', comments = None)
vectors = glove[:, 1:].astype('float')
words = glove[:, 0]

word_lookup = { word : vector for word, vector in zip(words, vectors) }
embedding_matrix = np.zeros((num_words, vectors.shape[1]))

# embedding layer
model.add(
    Embedding(input_dim = num_words,
              input_length = 50,
              output_dim = 100,
              weights = [embedding_matrix],
              trainable = False,
              mask_zero = True)
)

# recurrent layer
model.add(LSTM(64, return_sequences = False, dropout = 0.1, recurrent_dropout = 0.1))

# fully connected layer
model.add(Dense(64, activation = 'relu'))

# dropout for regularization
model.add(Dropout(0.5))

# output layer
model.add(Dense(num_words, activation = 'softmax'))

# compile the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.callbacks import EarlyStopping, ModelCheckpoint

callback = [EarlyStopping]
