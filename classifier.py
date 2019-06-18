import numpy as np
import pandas as pd
import statistics
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from skmultilearn.problem_transform import LabelPowerset
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model

import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Taking one label at a time

OVR_SVC_UNIGRAM_1= 1 #{For OneVsRest LSVM Unigram (single label at a time)}
OVR_SVC_BiGRAM_1= 2 #{For OneVsRest LSVM Bigram (single label at a time)}
OVR_SVC_UNIGRAM_2 = 3 #{For OneVsRest LSVM Unigram with tfidf parameter min_df=0.01, max_df=0.8 (single label at a time)}
OVR_SVC_BiGRAM_2 = 4 #{For OneVsRest LSVM Bigram with tfidf parameter min_df=0.01, max_df=0.8 (single label at a time)}

# Taking whole dataset
OVR_SVC_UNIGRAM_3 = 5 #{For OneVsRest LSVM Unigram with tfidf parameter min_df=0.01, max_df=0.8 }
OVR_SVC_BiGRAM_3 = 6 #{For OneVsRest LSVM bigram with tfidf parameter min_df=0.01, max_df=0.8 }
OVR_MNB_UNIGRAM = 7 #{For OneVsRest MultinomialNB Unigram with tfidf parameter min_df=0.01, max_df=0.8 }
OVR_MNB_BiGRAM = 8 #{For OneVsRest MultinomialNB bigram with tfidf parameter min_df=0.01, max_df=0.8 }
OVR_SGD_UNIGRAM = 9 #{For OneVsRest SGDClassifier Unigram tfidf parameter min_df=0.01, max_df=0.8 }
OVR_SGD_BiGRAM = 10 #{For OneVsRest SGDClassifier Bigram tfidf parameter min_df=0.01, max_df=0.8 }
LP_SVC_UNIGRAM = 11 #{For LabelPowerset LSVM Unigram with tfidf parameter min_df=0.01, max_df=0.8 }
LP_SVC_BIGRAM = 12 #{For LabelPowerset LSVM bigram with tfidf parameter min_df=0.01, max_df=0.8 }
LP_MNB_UNIGRAM = 13 #{For LabelPowerset MultinomialNB Unigram with tfidf parameter min_df=0.01, max_df=0.8 }
LP_MNB_BIGRAM = 14 #{For LabelPowerset MultinomialNB bigram with tfidf parameter min_df=0.01, max_df=0.8 }
LP_SGD_UNIGRAM = 15 #{For LabelPowerset SGDClassifier Unigram tfidf parameter min_df=0.01, max_df=0.8 }
LP_SGD_BiGRAM = 16 #{For LabelPowerset SGDClassifier Bigram tfidf parameter min_df=0.01, max_df=0.8 }


def getPipline(classifierType):
    if(classifierType == OVR_SVC_UNIGRAM_1):
        return Pipeline([('TFidf', TfidfVectorizer()), ("multilabel", OneVsRestClassifier(LinearSVC(random_state=0)))])
    elif(classifierType == OVR_SVC_BiGRAM_1):
        return Pipeline([('TFidf', TfidfVectorizer(ngram_range=(2, 2))), ("multilabel", OneVsRestClassifier(LinearSVC(random_state=0)))])
    elif(classifierType == OVR_SVC_UNIGRAM_2):
        return Pipeline([('TFidf', TfidfVectorizer(min_df=0.01, max_df=0.8)), ("multilabel", OneVsRestClassifier(LinearSVC(random_state=0)))])
    elif (classifierType == OVR_SVC_BiGRAM_2):
        return Pipeline([('TFidf', TfidfVectorizer(ngram_range=(2, 2), min_df=0.01, max_df=0.8)), ("multilabel", OneVsRestClassifier(LinearSVC(random_state=0)))])


def getClassifier(classifierType):
    if (classifierType == OVR_SVC_UNIGRAM_3 or classifierType == OVR_SVC_BiGRAM_3):
        return  OneVsRestClassifier(LinearSVC())
    elif (classifierType == OVR_MNB_UNIGRAM or classifierType == OVR_MNB_BiGRAM):
        return  OneVsRestClassifier(MultinomialNB(alpha=0.7))
    elif (classifierType == OVR_SGD_UNIGRAM or classifierType == OVR_SGD_BiGRAM):
        return  OneVsRestClassifier(linear_model.SGDClassifier())
    elif (classifierType == LP_SVC_UNIGRAM or classifierType == LP_SVC_BIGRAM):
        return  LabelPowerset(LinearSVC())
    elif (classifierType == LP_MNB_UNIGRAM or classifierType == LP_MNB_BIGRAM):
        return  LabelPowerset(MultinomialNB(alpha=0.7))
    elif (classifierType == LP_SGD_UNIGRAM or classifierType == LP_SGD_BiGRAM):
        return  LabelPowerset(linear_model.SGDClassifier())


def getVectoriser(classifierType):
    if (classifierType == OVR_SVC_BiGRAM_3 or classifierType == OVR_MNB_BiGRAM or classifierType == OVR_SGD_BiGRAM
            or classifierType == LP_SVC_BIGRAM or classifierType == LP_MNB_BIGRAM or classifierType == LP_SGD_BiGRAM):
        return TfidfVectorizer(ngram_range=(2, 2),min_df=0.01, max_df=0.8)
    else:
        return TfidfVectorizer(min_df=0.01, max_df=0.8)



def trainClassifier(destinationPath, classifierType):
    df1 = pd.read_csv(destinationPath+'/output1.csv')
    df2 = pd.read_csv(destinationPath+'/output2.csv')
    df3 = pd.read_csv(destinationPath+'/output3.csv')

    df4 = pd.read_csv(destinationPath+'/output4.csv')
    df5 = pd.read_csv(destinationPath+'/output5.csv')
    df6 = pd.read_csv(destinationPath+'/output6.csv')

    df7 = pd.read_csv(destinationPath+'/output7.csv')
    df8 = pd.read_csv(destinationPath+'/output8.csv')
    df9 = pd.read_csv(destinationPath+'/output9.csv')

    trainDF = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9]);
    testDF = pd.read_csv(destinationPath+'/output10.csv')

    totalDF = pd.concat([trainDF, testDF])

    trainDF, testDF = train_test_split(totalDF, test_size=0.3)

    print("Training trainDF.shape:" + str(trainDF.shape))
    print("Training testDF.shape:" + str(testDF.shape))

    # wordcloud after preprocessing
    # words = []
    # for t in totalDF['content']:
    #     words.append(t)
    # words[:4]
    # # print(words.shape)
    # word_text = pd.Series(words).str.cat(sep=' ')
    # w = WordCloud(width=1000, height=800, mode='RGBA', background_color='white',max_words=2000, collocations=False).generate(word_text)
    # plt.imshow(w)
    # plt.show()
    # plt.savefig('/Users/kartikprabhu/Desktop/wordCloud.png')


    print("Removing labels with frequency less than 5...")
    y_train = totalDF.iloc[:, 2:]
    constrainLabels = []
    for label in y_train:
        if np.count_nonzero(totalDF[label])<= 5:
            constrainLabels.append(label)

    print(y_train.shape)
    print("Number of labels with frequeny less than 5:" + str(len(constrainLabels)))
    trainDF = trainDF.drop(constrainLabels, axis=1)
    testDF = testDF.drop(constrainLabels, axis=1)

    print("Final TrainDF " + str(trainDF.shape))
    print("Final TestDF " + str(testDF.shape))

    if(classifierType < 5):
        pipe = getPipline(classifierType)

        aveg_ovr_uni = []
        avghloss_ovr_uni = []
        avgBalAcc_ovr_uni = []
        avgPrecision_ovr_uni = []
        avgFscore_ovr_uni = []
        avgFscore_micro_ovr_uni = []

        i = 0

        temp = trainDF.drop(['index', 'content'], axis=1)
        for label in temp.columns:
            i = i + 1
            print('... Processing ' + str(i) + ' {}'.format(label))

            pipe.fit(trainDF['content'], trainDF[label])
            predicted = pipe.predict(testDF['content'])
            print("Predicting")
            # print(predicted)

            acc = accuracy_score(testDF[label], predicted)
            print('Test accuracy is {}'.format(acc))
            aveg_ovr_uni.append(acc)
            print('Total accuracy is {}'.format(statistics.mean(aveg_ovr_uni)))

            hloss = hamming_loss(testDF[label], predicted)
            print('Test hamloss is {}'.format(hloss))
            avghloss_ovr_uni.append(hloss)
            print('Total hamloss is {}'.format(statistics.mean(avghloss_ovr_uni)))

            balancedAccuracyScore = balanced_accuracy_score(testDF[label], predicted)
            print('Test balancedAccuracyScore is {}'.format(balancedAccuracyScore))
            avgBalAcc_ovr_uni.append(balancedAccuracyScore)
            print('Total avgBalAcc is {}'.format(statistics.mean(avgBalAcc_ovr_uni)))

            precis = precision_score(testDF[label], predicted)
            print('Test precision is {}'.format(precis))
            avgPrecision_ovr_uni.append(precis)
            print('Total avgPrecision is {}'.format(statistics.mean(avgPrecision_ovr_uni)))

            f1score = f1_score(testDF[label], predicted, average='macro')
            print('Test f1 is {}'.format(f1score))
            avgFscore_ovr_uni.append(f1score)
            print('Total avgFscore is {}'.format(statistics.mean(avgFscore_ovr_uni)))

            f1score_micro = f1_score(testDF[label], predicted, average='micro')
            print('Test f1(micro) is {}'.format(f1score_micro))
            avgFscore_micro_ovr_uni.append(f1score_micro)
            print('Total avgFscore(micro) is {}'.format(statistics.mean(avgFscore_micro_ovr_uni)))

        totalAvg = statistics.mean(aveg_ovr_uni)
        print('Final Total accuracy is {}'.format(totalAvg))

        totalHLoss = statistics.mean(avghloss_ovr_uni)
        print('Final Total hloss is {}'.format(totalHLoss))

        totalBalanceAcc = statistics.mean(avgBalAcc_ovr_uni)
        print('Final Total balancedAccuracyScore is {}'.format(totalBalanceAcc))

        totalPrecision = statistics.mean(avgPrecision_ovr_uni)
        print('Final Total precision is {}'.format(totalPrecision))

        totalFscore = statistics.mean(avgFscore_ovr_uni)
        print('Final Total f1score is {}'.format(totalFscore))

    else:
        vectorizer = getVectoriser(classifierType)
        dataset_tfidf = vectorizer.fit_transform(totalDF['content'])
        X_train, X_test, y_train, y_test = train_test_split(dataset_tfidf, totalDF, test_size=0.33,random_state=42)

        classifier = getClassifier(classifierType)

        start_time = time.time()
        print('start time: %s' % (time.ctime(start_time)))
        classifier.fit(X_train, y_train)
        print("--- %s seconds ---" % (time.time() - start_time))

        print(classifier)
        # X_test = vectorizer.transform(X_test)
        predicted = classifier.predict(X_test)

        print("The Hamming Loss is: %.3f" % (hamming_loss(y_test, predicted)))
        print("The macro averaged F1-score is: %.3f" % (f1_score(y_test, predicted, average='macro')))






