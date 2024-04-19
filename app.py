from flask import Flask, render_template, request, url_for, jsonify
from SMSSpamModel import SMSSpamModel
import json

app = Flask(__name__, template_folder='templates')

model = SMSSpamModel('./sms_spam_collection/SMSSpamCollection')
sms = model.create_dataframe()
model.preprocess(sms)
X_train, X_test, y_train, y_test = model.train_split(sms)
model.train_model(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.post('/predict')
def predict():
    message = request.form.get('message')
    lang = request.form.get('lang')

    print(message)
    
    result = model.translate_and_predict(message)
    
    result = bool(result)
    
    return jsonify(result=result)

if __name__ == '__main__':  
    app.run(debug = True)

