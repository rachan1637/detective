from preprocess import preprocessing
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
import re
import pandas as pd
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from csv import DictWriter

def clean(text):
# Removes all special characters and numericals leaving the alphabets
    text = re.sub('[^A-Za-z]+', ' ', text)
    return text


def lemmatize(pos_data):
    wordnet_lemmatizer = WordNetLemmatizer()
    lemma_rew = " "
    for word, pos in pos_data:
        if not pos:
            lemma = word
            lemma_rew = lemma_rew + " " + lemma
        else:
            lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
            lemma_rew = lemma_rew + " " + lemma
    return lemma_rew


def vadersentimentanalysis(review):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(review)
    return vs['compound']


def vader_analysis(compound):
    if compound >= 0.5:
        return 'Positive'
    elif compound <= -0.5 :
        return 'Negative'
    else:
        return 'Neutral'


def create_code_to_partition_senti():
    code_to_partitions, code_to_plain_text = preprocessing()

    pos_dict = {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}

    senti_list = []
    for code, partitions in tqdm(code_to_partitions.items()):
        first = partitions['first']
        second = partitions['second']
        
        results = []
        for section in [first, second]:
            sents = sent_tokenize(section)
            lemma_list = []
            for sent in sents:
                cleaned_sent = clean(sent)
                tokens = word_tokenize(cleaned_sent)

                tags = pos_tag(tokens)

                new_list = []
                for word, tag in tags:
                    if word.lower() not in set(stopwords.words('english')):
                        new_list.append(tuple([word, pos_dict.get(tag[0])]))

                lemma = lemmatize(new_list)
                lemma_list.append(lemma)

            df = pd.DataFrame(data = {"sents": sents, "lemma": lemma_list})
            df['sentiment'] = df['lemma'].apply(vadersentimentanalysis)
            df['analysis'] = df['sentiment'].apply(vader_analysis)

            analysis_result = df['analysis'].value_counts()

            result_dict = {}
            result_dict['n_sents'] = len(df)
            for senti in ['Negative', 'Neutral', 'Positive']:
                if senti not in analysis_result:
                    result_dict[senti.lower()] = 0
                else:
                    result_dict[senti.lower()] = round(analysis_result[senti] / len(df), 2)
            results.append(result_dict)

        code_dict = dict()
        code_dict['code'] = code
        for key, value in results[0].items():
            code_dict[f'first_{key}'] = value
        
        for key, value in results[1].items():
            code_dict[f'second_{key}'] = value

        senti_list.append(code_dict)

    return senti_list

if __name__ == "__main__":
    senti_list = create_code_to_partition_senti()

    with open('sentiment.csv', 'w') as fout:
        writer = DictWriter(fout, fieldnames = ["code", "first_n_sents", "first_negative", "first_neutral", "first_positive", "second_n_sents", "second_negative", "second_neutral", "second_positive"])
        writer.writeheader()
        for row in senti_list:
            writer.writerow(row)