from flask import Flask, render_template
from flask.globals import request
import os
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)
model=load_model('model\model_lstm.h5')
seq_len=3
tokenizer = pickle.load(open('model/model_tokenizer.pkl', 'rb'))



@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/predict',methods=['POST'])
def predict():
    # print(request.files)
    input_text = request.form.get('message').strip().lower()
    encoded_text = tokenizer.texts_to_sequences([input_text])[0]
    pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
    #print(encoded_text, pad_encoded)
    l=[]
    for i in (model.predict(pad_encoded)[0]).argsort()[-3:][::-1]:
        pred_word = tokenizer.index_word[i]
        l.append(pred_word)
    return render_template('form.html',one="Next word suggestion:"+l[0],two="Next word suggestion:"+l[1],three="Next word suggestion:"+l[2])
        
if __name__ == '__main__':
    app.run()
