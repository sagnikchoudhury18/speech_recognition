import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation,LSTM,GRU,Dropout

test_feature=np.load('array/test_feature.npy')
train_feature=np.load('array/train_feature.npy')
test_label=np.load('array/test_label.npy')
train_label=np.load('array/train_label.npy')
num_classes=10

print(train_feature.shape)
print(train_label.shape)
#train_label=train_label.reshape(1920,10,1)
model = Sequential()
model.add(GRU(64, return_sequences=True,input_shape=(train_feature.shape[1],train_feature.shape[2])))  # returns a sequence of vectors of dimension 32
model.add(GRU(128, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(GRU(256))  # return a single vector of dimension 32
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model.fit(train_feature,train_label,epochs=500)
score = model.evaluate(test_feature, test_label, batch_size=15)
print(score)