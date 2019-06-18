

import preprocessing as preprocessing
import classifier as customclassifier
import sys
import os

PREPROCESSING = 1
TRAINING = 2

PREPROCESSING_SOURCE_PATH = ""
PREPROCESSING_DESTINATION_PATH = ""


def printOptions():
    print('For Executing: python executor.py <RunType> <Source> <Destination> <ClassifierType>')
    print('RunType = 1 {For preprocessing}, 2 {For training}')
    print('Source = Folder path of training files (only for preprocessing)')
    print('Destination = Folder path for preprocessed files')
    print('ClassifierType = 1{For OneVsRest LSVM Unigram (single label at a time)}')
    print('               = 2{For OneVsRest LSVM Bigram (single label at a time)}')
    print('               = 3{For OneVsRest LSVM Unigram with tfidf parameter min_df=0.01, max_df=0.8 (single label at a time)}')
    print('               = 4{For OneVsRest LSVM Bigram with tfidf parameter min_df=0.01, max_df=0.8 (single label at a time)}')
    print('               = 5{For OneVsRest LSVM Unigram with tfidf parameter min_df=0.01, max_df=0.8 }')
    print('               = 6{For OneVsRest LSVM bigram with tfidf parameter min_df=0.01, max_df=0.8 }')
    print('               = 7{For OneVsRest MultinomialNB Unigram with tfidf parameter min_df=0.01, max_df=0.8 }')
    print('               = 8{For OneVsRest MultinomialNB bigram with tfidf parameter min_df=0.01, max_df=0.8 }')
    print('               = 9{For OneVsRest SGDClassifier Unigram tfidf parameter min_df=0.01, max_df=0.8 }')
    print('               = 10{For OneVsRest SGDClassifier Bigram tfidf parameter min_df=0.01, max_df=0.8 }')
    print('               = 11{For LabelPowerset LSVM Unigram with tfidf parameter min_df=0.01, max_df=0.8 }')
    print('               = 12{For LabelPowerset LSVM bigram with tfidf parameter min_df=0.01, max_df=0.8 }')
    print('               = 13{For LabelPowerset MultinomialNB Unigram with tfidf parameter min_df=0.01, max_df=0.8 }')
    print('               = 14{For LabelPowerset MultinomialNB bigram with tfidf parameter min_df=0.01, max_df=0.8 }')
    print('               = 15{For LabelPowerset SGDClassifier Unigram tfidf parameter min_df=0.01, max_df=0.8 }')
    print('               = 16{For LabelPowerset SGDClassifier Bigram tfidf parameter min_df=0.01, max_df=0.8 }')

if __name__ == '__main__':

    try:
        if (PREPROCESSING == int(sys.argv[1])):
            if len(sys.argv) > 3:
                print('PREPROCESSING...')
                PREPROCESSING_SOURCE_PATH = sys.argv[2]
                PREPROCESSING_DESTINATION_PATH = sys.argv[3]

                if not os.path.exists(PREPROCESSING_SOURCE_PATH) or not os.path.isdir(PREPROCESSING_SOURCE_PATH):
                    print("Something is wrong with the path specified")
                    printOptions()
                    sys.exit()

                preprocessing.setupData(PREPROCESSING_SOURCE_PATH, PREPROCESSING_DESTINATION_PATH)
            else:
                printOptions()

        elif (TRAINING == int(sys.argv[1])):
            print('TRAINING...')
            PREPROCESSING_DESTINATION_PATH = sys.argv[2]

            if not os.path.isdir(PREPROCESSING_DESTINATION_PATH):
                print("Something is wrong with the path specified")
                printOptions()
                sys.exit()

            customclassifier.trainClassifier(PREPROCESSING_DESTINATION_PATH, int(sys.argv[3]))
        else:
            printOptions()

    except ValueError:
        print("Something went wrong with the input parameters...")
        printOptions()
