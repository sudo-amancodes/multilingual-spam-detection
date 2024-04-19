import pandas as pd
import os, re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import googletrans
from sklearn.metrics import classification_report, confusion_matrix
  
translator = googletrans.Translator()  
print(googletrans.LANGUAGES)  

text = "Hello, how are you?"

text2 = "Hola, ¿cómo estás?"

text3 = "Bonjour, comment vas-tu?"

print(translator.detect(text))

datapath = "./sms_spam_collection"

corpus = []
ps = PorterStemmer()
cv = CountVectorizer()

def read_examples(filename):
    label = []
    message = []
    with open(filename, mode = 'r', encoding = 'utf-8') as file:
        for line in file:
            #Spam or Ham
            label.append(line[:4].strip())

            #Preprocess the message
            message.append(line[4:].strip())

    return label, message

train_file = os.path.join(datapath, 'SMSSpamCollection')
label, message = read_examples(train_file)

sms = pd.DataFrame({'Label': label, 'Message': message})

for i in range(0, sms.shape[0]):
    message = re.sub('[^a-zA-Z]', ' ', sms['Message'][i])
    message = message.lower()
    message = message.split()
    words = [ps.stem(word) for word in message if not word in set(stopwords.words('english'))]
    message = ' '.join(words)
    corpus.append(message)

X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(sms['Label'])

y=y.iloc[:,1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred = spam_detect_model.predict(X_test)

accuracy = (y_pred == y_test).mean()
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# def language_converter(msg, msg_lang):
#     translator = Translator()
#     text_to_translate = translator.translate(msg, src=msg_lang, dest='en')
#     text = text_to_translate.text
#     return text