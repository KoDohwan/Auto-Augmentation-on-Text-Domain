# Datasets

| Dataset | Number of Classes | Size  |
| ------- | ----------------- | ----- |
| CR      | 2                 | 3771  |
| MR      | 2                 | 10662 |
| SST1    | 5                 | 11855 |
| SST2    | 2                 | 9613  |
| SUBJ    | 2                 | 10000 |
| TREC    | 6                 | 5952  |

The following datasets are included in this directory:

* **CR**: Customer reviews of various products (cameras, MP3s etc.). Task is to predict positive/negative reviews (Hu and Liu, 2004).

* **MR**: Movie reviews with one sentence per review. Classification involves detecting positive/negative reviews (Pang and Lee, 2005).

* **SST1**: Stanford Sentiment Treebank - an extension of MR but with train/dev/test splits provided and fine-grained labels (very positive, positive, neutral, negative, very negative), re-labeled by Socher et al. (2013).

  Note that data is actually provided at the phrase-level and hence we train the model on both phrases and sentences but only score on sentences at test time, as in Socher et al. (2013), Kalchbrenner et al. (2014), and Le and Mikolov (2014). Thus the training set is an order of magnitude larger than listed in the above table.

* **SST2** Same as SST-1 but with neutral reviews removed and binary labels.

* **Subj**: Subjectivity dataset where the task is to classify a sentence as being subjective or objective (Pang and Lee, 2004).

* **TREC**: TREC question dataset - task involves classifying a question into 6 question types (whether the question is about person, location, numeric information, etc.) (Li and Roth, 2002).