## News headline Category prediction

News is published at a very fast pace these days. With these large amounts of news published daily, it is important we distinguish the news categories from the headlines itself so we could read the news that we are most interested about. Using Natural Language Processing Techniques, we hope to generate classification labels for the news accurately.

The dataset we are working on is the [News Category Dataset](https://www.kaggle.com/rmisra/news-category-dataset). We extract headlines of 10 categories (CRIME, ENTERTAINMENT, FOOD & DRINK, MONEY, RELIGION, SPORTS, STYLE, TECH, TRAVEL and WOMEN) and try to classify the headlines into their corresponding categories.

### Approach

We explored the classification of headlines using logistic regression, feed-forward neural network, LSTM, LSTM + Attention, and transformer with various text preprocessing techniques applied. We found out that the transformer model (distilBert) with POS tagging to the word tokens on top of 50d Glove vector embeddings gives the best classification performance of 0.82 in F1 score, while the human classification benchmark is 0.84 in F1 score.

The report for this project can be found [here](https://github.com/fangpinsern/CS4248-Project/blob/master/CS4248_Group_Report.pdf)

### Set Up

- Download glove 6B 50d embedding vectors from: https://drive.google.com/file/d/1C5htxKX_Nk9OuyiQWhRjyD1GX2LMWAM2/view?usp=sharing , and place it into the folder `/data/embeddings`
- Download the dataset from: https://www.kaggle.com/rmisra/news-category-dataset , and place it into the folder `/data`
- run `pip install -r requirements.txt`
- run `python -m proj.models.utils` to generate the embeddings and tokenizers for varied vocabulary (e.g. unknown tokens, POS tags)

### User Guide

#### Data Anlysis and General Text Preprocessing Techniques

Notebook: [data_util_methods_and_analysis.ipynb](https://github.com/fangpinsern/CS4248-Project/blob/master/data_util_methods_and_analysis.ipynb)

This notebook is self-contained. Follow the instructions in the notebook to run.

#### Logistic Regression

Notebook: [Full_Logistic_Regression_Code.ipynb](https://github.com/fangpinsern/CS4248-Project/blob/master/Full_Logistic_Regression_Code.ipynb)

This notebook contains the code to run a logistic regression model on a dataset given. In our use case it is used to classify headlines into different categories. Details on the results and methods can be found in Section III-B and V-B of the [project report](https://github.com/fangpinsern/CS4248-Project/blob/master/CS4248_Group_Report.pdf).

With the dataset in the correct directory, change the datapath variable to point to the dataset. Run the notebook as usual and it will give an F1-score as well as an accuracy score.

The output of the predicted categories will be stored in the file "RESULT.csv".

#### Feed-forward Neural Network

Notebook: [simpleNNnews.ipynb](https://github.com/fangpinsern/CS4248-Project/blob/master/simpleNNnews.ipynb)

This notebook contains the code to run the model. Specific library requirements are given in the notebook itself.

The output of the predicted categories will be stored in the file "simpleNN.csv" and "simpleNN_balanced.csv".

- Download glove 840B 300d embedding vectors from https://nlp.stanford.edu/projects/glove/ and place it in the same directory as this README and simpleNNnews.ipynb

#### RNN + Transformers

For an introduction, do run `python -m proj.main` to run the sample training on a DistilBert with weighted sampler, Augmentation and POS tagging

For Hyperparameter tuning, edit the hp passed to trainer

Notebook: [RNN_Transformers.ipynb](https://github.com/fangpinsern/CS4248-Project/blob/master/RNN_Transformers.ipynb)

This notebook contains blocks that train variations of the LSTM and DistilBert.

Using variations would require changes to both the model and tokenizer loaded. Also changes would need to be made also with respect to the arguments given to the dataset to provide different experiment variations.

#### Visualizations for DistilBert

Notebook: [visualizeBert.ipynb](https://github.com/fangpinsern/CS4248-Project/blob/master/visualizeBert.ipynb)

> Note: this notebook can only be runned after we've trained some weights, you would also need to update the weight loading sections of the notebook to see some kind of reasonable result
