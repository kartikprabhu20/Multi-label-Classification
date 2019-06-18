import errno
import os
import re

import nltk
import pandas as pd
from IPython.display import clear_output
from bs4 import BeautifulSoup
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import ToktokTokenizer, word_tokenize
from pathlib import Path


nltk.data.path.append('/Users/kartikprabhu/Desktop/AML/nltk_data')

docs = []
lemma = WordNetLemmatizer()
token = ToktokTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
# removing law related stopwords
eurLex_en_stopwords = ["gt",
                       "notext",
                       "p",
                       "lt",
                       "aka",
                       "oj",
                       "n",
                       "a",
                       "eec",
                       "article",
                       "directive",
                       "follow",
                       "accordance",
                       "chairman",
                       "necessary",
                       "comply",
                       "reference",
                       "commission",
                       "opinion",
                       "decision",
                       "annex",
                       "refer",
                       "member",
                       "european",
                       "treaty",
                       "throughout",
                       "regulation",
                       "particular",
                       "thereof",
                       "community",
                       "committee",
                       "measure",
                       "parliament",
                       "regard",
                       "amend",
                       "procedure",
                       "administrative",
                       "procedure",
                       "publication",
                       "month",
                       "date",
                       "year",
                       "enter",
                       "force",
                       "ensure",
                       "authority",
                       "take",
                       "council",
                       "act",
                       "within",
                       "national",
                       "law",
                       "main",
                       "provision",
                       "mention",
                       "approve",
                       "certain",
                       "whereas",
                       "eea",
                       "also",
                       "apply",
                       "may",
                       "can",
                       "will",
                       "shall",
                       "require",
                       "paragraph",
                       "subparagraph",
                       "official",
                       "journal",
                       "ec",
                       "b",
                       "s",
                       "c",
                       "e",
                       "na"]
stop_words.extend(eurLex_en_stopwords)

# remove html tags
def removeTags(data):
    soup = BeautifulSoup(data, 'html.parser')
    text = soup.get_text()
    return text


def removeAlphaNumerics(content):
    cleaned_text = [w for w in content if w.isalpha()]
    return cleaned_text

def lemmatizeWords(text):
    words = token.tokenize(text)
    listLemma = []
    for w in words:
        x = lemma.lemmatize(w, 'v')
        listLemma.append(x)
    return text

def cleanData(content):
    # word_tokens = nltk.RegexpTokenizer(r'\w+').tokenize(content)
    word_tokens = word_tokenize(content)

    filtered_content = [t.lower() for t in word_tokens if t.lower() not in stop_words]

    # only allow words with characters
    filtered_content = [t for t in filtered_content if re.match(r"[a-zA-Z]+", t)]

    # remove alphanueric characters and numbers
    filtered_content = removeAlphaNumerics(filtered_content)

    # removeAccentedChars(filtered_content) #
    filtered_content = lemmatizeWords(filtered_content)

    porter = nltk.PorterStemmer()
    stemmed_content = [porter.stem(t) for t in filtered_content]
    filtered_content = [t.lower() for t in stemmed_content if len(t) >= 3]

    return filtered_content

def getClassification(content):
    soup = BeautifulSoup(content, 'html.parser')
    # [s.extract() for s in soup(['style', 'script', '[document]', 'head', 'title'])]
    # visible_text = soup.getText()
    # soup.body.findAll(text='Python')
    labelData = soup.find(text='Classification')
    if labelData:
        return 1
    else:
        return 0

def setupData(sourcePath, destinationPath):
    dir_entries = Path(sourcePath)
    filenames = []
    whole_labels = []
    doc_label_dict = {}
    doc_content_dict = {}

    for count, f_name in enumerate(dir_entries.iterdir()):
        print(f_name)

        print("===================")

        clear_output(wait=True)
        print('Progress: {:.2f}'.format(count / len(list(dir_entries.iterdir())) * 100), '%')
        documentArr = []

        if os.stat(f_name).st_size > 0:
            try:
                with open(f_name, 'r', encoding='utf-8') as file:
                    data = file.read()
                    filenames.append(Path(sourcePath).name)
                    soup = BeautifulSoup(data, 'html.parser')
                    for txt in soup.findAll(attrs={'class': 'texte'}):
                        content = txt.text.strip()

                        formatted_content = removeTags(content)
                        cleanContent = cleanData(formatted_content)
                        # vectorizer = TfidfVectorizer()
                        cleanContent1 = ' '.join(cleanContent)
                        documentArr.append(cleanContent1)
                        # print(documentArr)

                    # Code to print all Labels
                    dc_label = ''
                    labelArr = []
                    for link in soup.find_all("a", href=True):
                        linkurl = link.get('href')
                        linktext = link.text
                        if linkurl.find('EUROVOC') > 0:
                            labelArr.append((linktext).strip().lower())

                    print("Labels : ", labelArr)
                    filename = Path(f_name).stem
                    doc_label_dict[filename] = labelArr
                    doc_content_dict[filename] = documentArr
                    whole_labels.extend(labelArr)
                    docs.append([filename, documentArr, labelArr])

            except IOError as exc:
                if exc.errno != errno.EISDIR:
                    raise

    columnNames = ['index', 'content']
    unique_labels = list(set(whole_labels))
    columnNames.extend(unique_labels)

    print("Creating csv")
    localDF = pd.DataFrame(columns=columnNames)

    foldSize = len(doc_label_dict.items()) // 10;
    print("foldSize :" + str(foldSize))

    i = 0;
    for key, val in doc_label_dict.items():
        i = i + 1
        foldNumber = i // foldSize
        someDict = {}
        someDict['index'] = key

        docContent = doc_content_dict.get(key, "")

        if not docContent:
            continue

        someDict['content'] = docContent
        for d in unique_labels:
            if d in val:
                someDict[d] = 1
            else:
                someDict[d] = 0

        print("key:" + key)
        localDF = localDF.append(someDict, ignore_index=True)

        if (i % foldSize == 0 and foldNumber < 10):
            localDF.to_csv(destinationPath+'/output' + str(foldNumber) + '.csv',
                           header=columnNames, index=False)
            localDF = pd.DataFrame(columns=columnNames)

    localDF.to_csv(destinationPath+'/output' + str(10) + '.csv', header=columnNames,
                   index=False)


