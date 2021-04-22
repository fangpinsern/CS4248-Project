## News headline Category prediction

News is published at a very fast pace these days. With these large amounts of news published daily, it is important we distinguish the news categories from the headlines itself so we could read the news that we are most interested about. Using Natural Language Processing Techniques, we hope to generate classification labels for the news accurately.

The dataset we are working on is the [News Category Dataset](https://www.kaggle.com/rmisra/news-category-dataset). We extract headlines of 10 categories (CRIME, ENTERTAINMENT, FOOD & DRINK, MONEY, RELIGION, SPORTS, STYLE, TECH, TRAVEL and WOMEN) and try to classify the headlines into their corresponding categories.

### Approach
We explored the classification of headlines using logistic regression, feed-forward neural network, LSTM, LSTM + Attention, and transformer with various text preprocessing techniques applied. We found out that the transformer model (distilBert) with POS tagging to the word tokens on top of 50d Glove vector embeddings gives the best classification performance of 0.82 in F1 score, while the human classification benchmark is 0.84 in F1 score. 

The report for this project can be found [here](https://github.com/fangpinsern/CS4248-Project/blob/master/CS4248_Group_Report.pdf)

Details of data analysis and derivation of the general text preprocessing techniques can be found [here](https://github.com/fangpinsern/CS4248-Project/blob/master/data_util_methods_and_analysis.ipynb)


### Set Up
+ Download glove 6B 50d embedding vectors from: https://drive.google.com/file/d/1C5htxKX_Nk9OuyiQWhRjyD1GX2LMWAM2/view?usp=sharing , and place it into the folder `/data/embeddings`
+ Download the dataset from: https://www.kaggle.com/rmisra/news-category-dataset , and place it into the folder `/data`
