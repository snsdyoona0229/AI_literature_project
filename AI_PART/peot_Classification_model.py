import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False 
from tensorflow import keras
from keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
#from keras.utils.np_utils import *
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers import Conv2D
import jieba as jb
import re
from keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

df = pd.read_excel("/literature_project/AI_MODEL/peot_Classification_model/all_peot_0206.xlsx")
df=df[['athor','Name','class_id','poet']]
#print("總數: %d ." % len(df))
#df.sample(10)
#總數2400

df['cat_id'] = df['class_id'].factorize()[0]
cat_id_df = df[['class_id', 'cat_id']].drop_duplicates().sort_values('cat_id').reset_index(drop=True)
cat_to_id = dict(cat_id_df.values)
id_to_cat = dict(cat_id_df[['cat_id', 'class_id']].values)
#df.sample(10)
#文本列表pandas


#定義删除除字母,数字，文字以外的所有符號的函数
def remove_punctuation(line):
    line = str(line)
    if line.strip()=='':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('',line)
    return line
 
def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords

#定義删除除字母,数字，文字以外的所有符號的函数
df['clean_review'] = df['poet'].apply(remove_punctuation)
#df.sample(10)

#删除除字母,数字，文字以外的所有符號
df['cut_review'] = df['clean_review'].apply(lambda x: " ".join([w for w in list(jb.cut(x))]))
#df.head(10)

# 設置最頻繁使用的50000個詞(在texts_to_matrix是會取前MAX_NB_WORDS,會取前MAX_NB_WORDS列)
MAX_NB_WORDS = 5000
# 每條cut_review最大的長度
MAX_SEQUENCE_LENGTH =50
# 設置Embeddingceng層的维度
EMBEDDING_DIM = 50

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['cut_review'].values)
word_index = tokenizer.word_index
#print('共有 %s 個不相同的詞語.' % len(word_index))

X = tokenizer.texts_to_sequences(df['cut_review'].values)
#填充X,X的各個列的長度统一
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

#多類別標籤的onehot展開
Y = pd.get_dummies(df['cat_id']).values

#print(X.shape)(2400, 50)
#print(Y.shape)(2400, 8)

#拆分訓練集和測試集
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.30, random_state = 60)
#print(X_train.shape,Y_train.shape)(1680, 50) (1680, 8)
#print(X_test.shape,Y_test.shape)(720, 50) (720, 8)

#########################################################################################################

#定義模型.
def definition_model():
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.1))
    #model.add(LSTM(30, dropout=0.2, recurrent_dropout=0.2))
    model.add(Bidirectional(LSTM(8, return_sequences=True), input_shape=(8, 10)))
    #model.add(LSTM(32, return_sequences=True))  # 返回维度为 32 的向量序列
    #model.add(LSTM(32)) 
#model.add(GlobalAveragePooling1D())
    #model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(30)))

    model.add(Dense(8, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
#########################################################################################################
#訓練
def train_model():
    epochs = 25
    batch_size = 10

    #history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.13)
    #callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.30)
    #callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

#########################################################################################################
model = tf.keras.models.load_model('/literature_project/AI_PART/peot_Classification_model/peot_Classification_model_02.h5')
#accr = model.evaluate(X_test,Y_test)#準確率
#print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

def predict(text):
    txt = remove_punctuation(text)
    txt = [" ".join([w for w in list(jb.cut(txt))])]
    seq = tokenizer.texts_to_sequences(txt)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)
    cat_id= pred.argmax(axis=1)[0]
   #print(txt)
   # print(padded)
    style=["%f"%pred[0][0],"%f"%pred[0][1],"%f"%pred[0][2],"%f"%pred[0][3],"%f"%pred[0][4],"%f"%pred[0][5],"%f"%pred[0][6],"%f"%pred[0][7]]
    class_style=["創世紀","原住民","客家","新月","新詩","現代詩","笠","藍星"]
    #print(style)
    myvar = pd.Series(style,class_style)
    return myvar,cat_id_df[cat_id_df.cat_id==cat_id]['class_id'].values[0],cat_id


#reference
#https://github.com/tongzm/ml-python/blob/master/%E5%9F%BA%E4%BA%8ELSTM%E7%9A%84%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%A4%9A%E5%88%86%E7%B1%BB%E5%AE%9E%E6%88%98.ipynb
#https://github.com/649453932/Chinese-Text-Classification-Pytorch
#https://github.com/Ailln/text-classification
#https://github.com/wavewangyue/text-classification
