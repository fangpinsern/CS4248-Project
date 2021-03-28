## Name My Playlist: Playlist Generation using Transformers

### Motivation
This project is inspired from Sofia Samaniego’s project for the Natural Language Processing (NLP) course in Stanford, CS224N. It is entitled “Playlist Title Generation Using Sequence to Sequence”. As heavy users of Spotify, we are interested in building on her work, in improving the architecture of the model. 

### Statement of the Problem/Task
This task is a Natural Language Generation (NLG) task. When fed information about the tracklist, we want to build a robust model that can output a playlist name that is descriptive of the tracks. 
There are a few key questions that we would like to ask through our project:
How can we do better than a Seq-to-Seq model for natural language generation?
How do we know whether our model is close to human performance?
Can we apply a better metric than ROUGE-2 for our abstractive summarization task?.

### General Approach
This project can be considered a text generation project and more specifically a summarization project. We will thus be planning our approach with this perspective.

### Set Up
+ Download glove 6B 50d embedding vectors from: https://drive.google.com/file/d/1C5htxKX_Nk9OuyiQWhRjyD1GX2LMWAM2/view?usp=sharing , and place it into the folder `/embeddings`
