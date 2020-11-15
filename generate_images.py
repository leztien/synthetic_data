

import numpy as np

def make_images(n_images, side_length):
    """
    generates 5 classes of images
    """
    m = n_images
    h = w = side_length
    d = 3
    
    classes = (0, 1, 2, 3, 4)
    k = len(classes)
    m_per_class = m//k
    y = sum([(k,)*m_per_class for k in range(k)], ())
    X = np.empty(shape=(m_per_class*k, h,w,d))
    
    for (ix,c) in enumerate(y):
        pp = np.random.rand(w*2)
        pp = pp / pp.sum()
        row = np.random.multinomial(n=255*w, pvals=pp, size=3)
        for i in range(h):
            for j in range(3):
                if   c==3: X[ix,i,:,j] = row[j, :w]
                elif c==4: X[ix, :,i,j] = row[j, :w]
                else:      X[ix, i,:,j] = row[j,i:i+w][::(1 if np.random.rand()>(c/2) else -1)]
        X[ix, :, :, np.random.randint(0,3)] *= np.linspace(0,1, num=w)
        X[ix, :, :, np.random.randint(0,3)] *= np.linspace(1,0, num=w).reshape(-1,1)
    #add noise, clip
    X += np.random.randn(*X.shape)*30
    X = np.clip(X, 0, 255).astype('uint8')
    return(X, np.array(y, dtype='uint8'))

####################################################################################################

#DATA
X,y = make_images(n_images=1000, side_length=70)
k = len(set(y))
m,h,w,d = X.shape


#MODEL
from keras import Sequential
from keras.layers import SeparableConv2D, MaxPooling2D, Conv2D, GlobalAveragePooling2D, Dense
from keras.losses import sparse_categorical_crossentropy
from keras.metrics import sparse_categorical_accuracy, categorical_accuracy

layers = [
SeparableConv2D(filters=32, kernel_size=3, strides=1, padding='valid', activation='relu', input_shape=(h,w,d)),
SeparableConv2D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu'),
MaxPooling2D(pool_size=2),

SeparableConv2D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu'),
SeparableConv2D(filters=128, kernel_size=3, strides=1, padding='valid', activation='relu'),
MaxPooling2D(pool_size=2),
#                  1x1 convolution
Conv2D(filters=32, kernel_size=1, strides=1, padding='valid', activation='relu'),
GlobalAveragePooling2D(),

Dense(units=16, activation='relu'),
Dense(units=k, activation='softmax')
    ]

md = Sequential(layers)
md.summary()
md.compile(optimizer='adam', loss=sparse_categorical_crossentropy, metrics=[sparse_categorical_accuracy])
md.fit(X,y, validation_split=0.2, batch_size=256, epochs=3)

#PREDICT
ypred = md.predict(X).argmax(axis=1)
acc = (y==ypred).mean().round(3)
print("accuracy =", acc)
