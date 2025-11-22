Natural Language Processing with Disaster Tweets

Description of the problem and data
Nowadays, people often use social media to get or share information. The purpose of this project is to use natural language processing skills to detect tweets about a real disaster. 
It is a challenging problem because people can discuss any topics online. Traditional methods, such as engineering features from the tweet texts, may not work very well because 
it is hard to develop features. Deep learning methods, including recurrent neural networks, long short-term memory, and gated recurrent unit, may better capture the sequence of 
the information, which may better help detect the information

Data information
The training dataset contains five variables: id, keyword, location, text and target. There is missing information in the variables of keyword and location, so we just focus 
on id, text and target. Id is the Id information of the tweet. Text is the tweets. Target is whether the tweet is about a disastrous event (1 for yes and 0 for no). In the training 
dataset, there are 7613 observations, and in the test dataset, there are 3263 observations. I will use the training set to develop and validate the model and then use the test set 
to predict the information.

Exploratory data analysis
Show a few visualizations like histograms. Describe any data cleaning procedures. Based on your EDA, what is your plan of analysis? I first showed the histogram of the outcome 
variable: 4342 tweets are not about a disastrous event but 3271 tweets are about a disastrous event. For the variable of "text", I engineered two new variables: length of the text 
and word count of the text, and also used histograms to visualize them. From the word count information, I know that most of the tweets are below 30. For the data cleaning process, 
I remove three variable: id, keyword and location. For the "text" variable, I tokenized them and added paddings to make each tweet the same length. My plan of analysis is to use 
deep learning models like RNN, LSTM, GRU and bi-LSTM because they are good at capturing the sequential information.

Model architecture
To convert the raw text into a numerical format suitable for a sequential neural network, I used a Word-Level tokenizer implemented with the tokenizers library. This tokenizer 
splits each sentence into words using simple whitespace rules, which ensures that each word is treated as an individual token. I trained a WordLevel vocabulary on the training 
texts, allowing the tokenizer to build a mapping from each unique word to an integer ID. Two special tokens were added: for sequence padding and for words not seen during training. 
After training, each piece of text is transformed into a list of token IDs that represent the words in the sentence. Because neural networks require inputs of uniform length, 
I applied padding: sequences longer than the maximum length are truncated, while shorter sequences are padded with the token’s ID. This ensures that all input sequences have the 
same length and can be batched efficiently during model training. This word-level tokenization and padding procedure provides a straightforward and interpretable numerical 
representation of text data that is compatible with RNN-based architectures such as LSTMs, GRUs, and Bidirectional RNNs.

Results and analysis
I estimated four deep learning models: RNN, LSTM, GRU, and bi-LSTM. First, I chose embedding dimension 100, hidden dimension 128, 5 epochs, 3 layers for each model, and no dropout.
I did the parameter tuning. I estimated the same four models but this time I added epoch to 20 to see whether there were steady results. I included 2 stacked layers and then had 
5 layers in total. I also chose dropout of 0.3.

From these results, we can see that GRU and BiLSTM achieved better results than RNN and LSTM. Adding layers and dropout helped improve RNN and LSTM performance a bit, but did not 
change GRU and BiLSTM performance much. GRU and BiLSTM outperform RNN and LSTM because they are more effective at capturing sequential patterns and long-range dependencies with 
fewer parameters and better bidirectional context. Additional layers and dropout improved weaker models (RNN, LSTM) by increasing capacity and reducing overfitting, but did not 
noticeably improve GRU or BiLSTM because these architectures were already expressive enough for the task.

Conclusion
Learnings and takeaways: you need to understand the unique features of different sequential neural models and then choose the appropriate model. Just adding new features, such as 
stacked layers or dropout, would help with some model but not others. For example, GRU and BiLSTM achieved better results than RNN and LSTM. Adding layers and dropout helped improve 
RNN and LSTM performance a bit, but did not change GRU and BiLSTM performance much. I created the submission file and submitted to Kaggle. I got 0.73337. In the future, I would try 
some transformer-based moels, such as RoBERTa and DistilBERT to see their performance.
