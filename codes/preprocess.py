import librosa
from tqdm import tqdm
from keras.utils import np_utils
import numpy as np
import os
dest=os.listdir('dataset')
label=[]
feature=[]
#print(label)

for i in (range(len(dest))):
    try:
        fi='dataset/'+dest[i]
        print(fi)
        data,rate=librosa.load(fi)
        data=data[len(data)-5000:len(data)]
        #print(rate)
        mfcc=librosa.feature.mfcc(y=data,sr=rate,n_mfcc=5)
        print(mfcc.shape)
        if(mfcc.shape==(5,10)):
            l=dest[i]
            print(1)
            label.append(int(l[0]))
            print(2)
            feature.append(mfcc)
            print(3)
        #print(label)    
    except:
        print('File not found')

#print(len(train))
#print(len(label))
#label=np.array(label)
label=np_utils.to_categorical(label,10)
feature=np.array(feature)
train_label,test_label=label[0:1920],label[1920:2400]
train_feature,test_feature=feature[0:1920],feature[1920:2400]

np.save('array/train_label',train_label)
np.save('array/test_label',test_label)
np.save('array/train_feature',train_feature)
np.save('array/test_feature',test_feature)

