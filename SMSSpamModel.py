import pandas as pd
import os, re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from googletrans import Translator


class SMSSpamModel:
    def __init__(self, filename) -> None:
        self.corpus = []
        self.spam_detect_model = None
        self.filename = filename

        self.ps = PorterStemmer()
        self.cv = CountVectorizer()

    def __read_examples(self, filename):
        label = []
        message = []
        with open(filename, mode = 'r', encoding = 'utf-8') as file:
            for line in file:
                #Spam or Ham
                label.append(line[:4].strip())

                #Preprocess the message
                message.append(line[4:].strip())

        return label, message
    
    def create_dataframe(self):
        label, message = self.__read_examples(self.filename)
        sms = pd.DataFrame({'Label': label, 'Message': message})
        return sms
    
    def preprocess(self, sms):
        for i in range(0, sms.shape[0]):
            message = re.sub('[^a-zA-Z]', ' ', sms['Message'][i])
            message = message.lower()
            message = message.split()
            words = [self.ps.stem(word) for word in message if not word in set(stopwords.words('english'))]
            message = ' '.join(words)
            self.corpus.append(message)

    def train_split(self, sms):
        X = self.cv.fit_transform(self.corpus).toarray()

        y = pd.get_dummies(sms['Label'])

        y=y.iloc[:,1].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        self.spam_detect_model = MultinomialNB().fit(X_train, y_train)

    
    def model_accuracy(self, X_test, y_test):
        y_pred = self.spam_detect_model.predict(X_test)

        accuracy = (y_pred == y_test).mean()
        print("Accuracy:", accuracy)

    def predict(self, message):
        message = re.sub('[^a-zA-Z]', ' ', message)
        message = message.lower()
        message = message.split()
        words = [self.ps.stem(word) for word in message if not word in set(stopwords.words('english'))]
        message = ' '.join(words)
        X = self.cv.transform([message]).toarray()
        prediction = self.spam_detect_model.predict(X)
        return prediction[0] == 1
    
    def translate_and_predict(self, msg, msg_lang = None):
        translator = Translator()

        if msg_lang == None:
            msg_lang = translator.detect(msg).lang

        text_to_translate = translator.translate(msg, src=msg_lang, dest='en').text
        
        return self.predict(text_to_translate)

    