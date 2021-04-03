## News headline Category prediction

### Motivation
Motivated by the large volume of news titles online, we seek if we are able to classify them accurately

### Statement of the Problem/Task
This task is a text classification task. When fed the headlines of 10 possible categories, is the model capable of predicting the right category?
There are a few key questions that we would like to ask through our project:
How can we do better than a Seq-to-Seq model for text classification?
Can we apply a better metric than accuracy for our task?
How should we deal with the imbalance of data that happends naturally? (some categories are more important than anything)

### General Approach
This project can be considered a text classification project. We will thus be planning our approach with this perspective.

### Set Up
+ Download glove 6B 50d embedding vectors from: https://drive.google.com/file/d/1C5htxKX_Nk9OuyiQWhRjyD1GX2LMWAM2/view?usp=sharing , and place it into the folder `/data/embeddings`
+ Download the dataset from: https://www.kaggle.com/rmisra/news-category-dataset , and place it into the folder `/data`
+ We are using this subset of categories {"CRIME", "RELIGION", "TECH", "MONEY", "FOOD & DRINK", "SPORTS", "TRAVEL", "WOMEN", "STYLE", "ENTERTAINMENT"}
