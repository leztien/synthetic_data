
"""
synthetic corpus with synthetic labeling for neural network classifier
(the follwing NN is overfitting on the training data. can you improve the validation rate?)
"""

def make_corpus(m=100, binary=True):
    """
    makes a fake synthetic corpus with binary or multiclass (4 classes) labels
    the labeling represents arbitrary text complexity
    """
    import re
    from random import randint
    from textwrap import TextWrapper
    s = TextWrapper.__doc__.lower() + re.__doc__.lower()
    
    reg = re.compile('[a-z]{2,}')
    words = sorted(tuple(set(re.findall(reg, s)))[:200])
    
    mn,mx = (10,100)  # random length of texts
    corpus = [str.join(" ", (words[randint(0,len(words)-1)] for _ in range(randint(mn,mx)))) for _ in range(m)]
    
    def label(text, corpus, words=None, binary=True):
        """immitates defining the complexity of a text"""
        text = text if isinstance(text, list) else text.split(" ")
        
        middle = len(words)/2
        b1 = (sum(words.index(word)+1 for word in text) / len(text)) > middle
        
        avg = sum(len(word) for word in words) / len(words)   # average length of words
        b2 = (sum(len(word) for word in text) / len(text)) > avg
        
        p = len(re.findall("\w+ing|\w+ed|\w+abl[ye]", str.join(" ", words))) / len(words)
        b3 = len(re.findall("\w+ing|\w+ed|\w+abl[ye]", str.join(" ", text))) / len(text) >= p
        
        if binary:
            b2 = b2 or b3
            return int(b1 and b2)
        else:
            b2 = b2 and b3
            label = int(str(int(b1))+str(int(b2)), base=2)
            return label
        return#
    
    y = [label(text, corpus, words, binary) for text in corpus]
    assert len(y)==len(corpus),"error"
    return(corpus,y)


def encode_corpus(corpus):
    sets = [frozenset(text.split(" ")) for text in corpus]
    from operator import or_
    from functools import reduce
    words = sorted(reduce(or_, sets))
    d = {v:k for k,v in enumerate(words)}
    data = [tuple(d[k] for k in s.split(" ")) for s in corpus]
    return data


def vectorize(data:'list of lists of different sizes', n=None):
    n = int(n) if not(n is None) else max(max(l) for l in data)+1
    from numpy import zeros
    X = zeros(shape=(len(data),n), dtype='uint8')
    for i,l in enumerate(data):
        X[i,l] = 1
    return X


def onehotize(labels:'list of integers', n=None):  #categorical encoding
    n = n or max(labels)+1
    from numpy import zeros
    Y = zeros(shape=(len(labels), n), dtype='uint8')
    for i,ix in enumerate(labels):
        Y[i,ix] = 1
    return Y

#==============================================================================================   
    
data, y = make_corpus(3000, binary=True)
data = encode_corpus(data)
X = vectorize(data)
ytrue = y

#build a model (binary)
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers

penalty = regularizers.l2(0.00001)

md = Sequential()
md.add(Dense(64, input_shape=(X.shape[1],), activation="relu", kernel_regularizer=penalty))
md.add(Dense(64, activation="relu", kernel_regularizer=penalty))
md.add(Dense(1, activation="sigmoid", kernel_regularizer=penalty))

md.summary()

md.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
md.fit(X,y, batch_size=128, epochs=20, validation_split=0.2)   


#build a model (multiclass)
data, y = make_corpus(3000, binary=False)
data = encode_corpus(data)
X = vectorize(data)
Y = onehotize(y)

penalty = regularizers.l2(0.01)

md = Sequential()
md.add(Dense(64, input_shape=(X.shape[1],), activation="relu", kernel_regularizer=penalty))
md.add(Dense(64, activation="relu", kernel_regularizer=penalty))
md.add(Dense(4, activation="softmax", kernel_regularizer=penalty))

md.summary()

md.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
md.fit(X,Y, batch_size=128, epochs=40, validation_split=0.2)   
    
