import fasttext
import pandas as pd
from pymorphy2 import MorphAnalyzer
from sklearn.metrics import classification_report

TEST_DATA = pd.read_excel('test.xlsx')
MORPH = MorphAnalyzer()


def train():
    classifier = fasttext.train_supervised(input="data_for_fasttext_train.txt", epoch=20)
    classifier.save_model('fasttext_classifier.bin')
    return classifier


def test(data=TEST_DATA):
    predicted_labels = []
    classifier = train()
    data['pleonasm'] = data['pleonasm'].fillna('')

    for item in data['pleonasm']:
        item = item.replace('ё', 'е')
        item = ' '.join([MORPH.parse(word)[0].normal_form for word in item.split()])
        label = classifier.predict(item)[0][0]
        if label == '__label__same':
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)

    print(classification_report(data['class'], predicted_labels))