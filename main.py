from FileHandling import HandlingClass
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import tensorflow
from tensorflow.keras.layers import LSTM,Embedding,Dense,TimeDistributed,Dropout,Bidirectional
from tensorflow.keras.utils import plot_model
from numpy.random import seed
class Main:
    def printiningFunction(self):
        print('I am Function')
        data = HandlingClass.dataloader(self)
        tok2idx,idx2token = HandlingClass.get_dict_data(self,data,'token')
        tag2idx,idx2tag  = HandlingClass.get_dict_data(self, data, 'tag')
        data['Word_idx'] = data['Word'].map(tok2idx)
        data['Tag_idx'] = data['Tag'].map(tag2idx)
        data_fillNa = data.fillna(method='ffill',axis=0)
        data_group = data_fillNa.groupby(['Sentence #'],as_index=False)['Word', 'POS', 'Tag', 'Word_idx', 'Tag_idx'].agg(lambda  x:list(x))
        print(data_group)

    def get_pad_train_test(self,group_data,data):
        n_token = len(list(set(data['Word'].to_list())))
        n_tag = len(list(set(data['Tag'].to_list())))
        token = group_data['Word_idx'].tolist()
        maxlen = max([len(s) for s in token])
        pad_token = pad_sequences(token,maxlen=maxlen,dtype='int32',padding='post',value=n_token - 1)
        tags = group_data['Tag_idx'].tolist()
        pad_tags = pad_sequences(tags,maxlen=maxlen,dtype='int32',padding='post',value=tag2idx["0"])
        n_tag = len(tag2idx)
        pad = [to_categorical(i,num_classes = n_tags) for i in pad_tags]
        token_,test_tokens,tag_,test_tags = train_test_split(pad_token,pad_tags,test_size=0.1,train_size=0.9,random_state=2020)
        train_tokens,val_tokens,train_tags,val_tags = train_test_split(token_,tag_,test_size=0.25,train_size=0.75,random_state=2020)
        return train_tokens,val_tokens,test_tokens,train_tags,val_tags,test_tags

    train_tokens, val_tokens, test_tokens, train_tags, val_tags, test_tags = get_pad_train_test(data_group,data

    # Training Neural Network For NER
    seed(1)
    tensorflow.random.set_seed(2)
    input_dim = len(list(set(data['Word'])))+1
    output_dim = 64
    input_length = max(len(s) for in data_group['Word_idx'].toList()))

    def get_bilstm_lstm_model():
        model = Sequential()

        # Add Embedding layer
        model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))

        # Add bidirectional LSTM
        model.add(Bidirectional(LSTM(units=output_dim, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
                                merge_mode='concat'))

        # Add LSTM
        model.add(LSTM(units=output_dim, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))

        # Add timeDistributed Layer
        model.add(TimeDistributed(Dense(n_tags, activation="relu")))

        # Optimiser
        # adam = k.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)

        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        return model

    def train_model(X, y, model):
        loss = list()
        for i in range(25):
            # fit model for one epoch on this sequence
            hist = model.fit(X, y, batch_size=1000, verbose=1, epochs=1, validation_split=0.2)
            loss.append(hist.history['loss'][0])
        return loss

    results = pd.DataFrame()
    model_bilstm_lstm = get_bilstm_lstm_model()
    plot_model(model_bilstm_lstm)
    results['with_add_lstm'] = train_model(train_tokens, np.array(train_tags), model_bilstm_lstm)
    import spacy
    from spacy import displacy
    nlp = spacy.load('en_core_web_sm')
    text = nlp(
        'Hi, My name is Aman Kharwal \n I am from India \n I want to work with Google \n Steve Jobs is My Inspiration')
    displacy.render(text, style='ent', jupyter=True)






main = Main()
main.printiningFunction()
