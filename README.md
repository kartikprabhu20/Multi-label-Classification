# Multi-label-Classification

This project is part of Advanced Machine learning course at Otto Von Guericke University, Magdeburg - 2019 ( http://www.dke-research.de/en/Studies/Courses/Summer+Term+2019/Advanced+Topics+in+Machine+Learning.html)

For Executing: python executor.py <RunType> <Source> <Destination> <ClassifierType> <br/> 
RunType = 1 {For preprocessing}, 2 {For training}<br/>
Source = Folder path of training files (only for preprocessing)<br/>
Destination = Folder path for preprocessed files<br/><br/>
ClassifierType 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp= 1{For OneVsRest LSVM Unigram (single label at a time)}<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp= 2{For OneVsRest LSVM Bigram (single label at a time)}<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp= 3{For OneVsRest LSVM Unigram with tfidf parameter min_df=0.01, max_df=0.8 (single label at a time)}<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp= 4{For OneVsRest LSVM Bigram with tfidf parameter min_df=0.01, max_df=0.8 (single label at a time)}<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp= 5{For OneVsRest LSVM Unigram with tfidf parameter min_df=0.01, max_df=0.8 }<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp= 6{For OneVsRest LSVM bigram with tfidf parameter min_df=0.01, max_df=0.8 }<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp= 7{For OneVsRest MultinomialNB Unigram with tfidf parameter min_df=0.01, max_df=0.8 }<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp= 8{For OneVsRest MultinomialNB bigram with tfidf parameter min_df=0.01, max_df=0.8 }<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp= 9{For OneVsRest SGDClassifier Unigram tfidf parameter min_df=0.01, max_df=0.8 }<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp= 10{For OneVsRest SGDClassifier Bigram tfidf parameter min_df=0.01, max_df=0.8 }<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp= 11{For LabelPowerset LSVM Unigram with tfidf parameter min_df=0.01, max_df=0.8 }<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp= 12{For LabelPowerset LSVM bigram with tfidf parameter min_df=0.01, max_df=0.8 }<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp= 13{For LabelPowerset MultinomialNB Unigram with tfidf parameter min_df=0.01, max_df=0.8 }<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp= 14{For LabelPowerset MultinomialNB bigram with tfidf parameter min_df=0.01, max_df=0.8 }<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp= 15{For LabelPowerset SGDClassifier Unigram tfidf parameter min_df=0.01, max_df=0.8 }<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp= 16{For LabelPowerset SGDClassifier Bigram tfidf parameter min_df=0.01, max_df=0.8 }<br/>
