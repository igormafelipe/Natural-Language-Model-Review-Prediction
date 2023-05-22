# Natural-Language-Model-Review-Prediction
An exploration of different data transformations and models to classify customer reviews as good or bad.

# Overview
There are several thousand single-sentence reviews, collected from three domains: imdb.com, amazon.com, and yelp.com. Each review consists of a sentence, and has been assigned a
binary label indicating the sentiment (1 for positive and 0 for negative) of that sentence. The goal is to develop binary classifiers that can generate the sentiment-labels for new sentences, automating
the assessment process. While the reviews were collected from websites where much of the content is in English, the reviews may well contain slang, spelling errors, foreign characters and the like,
all of which make natural language data challenging, albeit fun, to try to classify like this.

# Data
The provided data consists of 2, 400 training examples in the usual CSV x and y format.∗Input data has two columns, for the source-website and review text; outputs are given as binary values,
where 1 indicates a positive review. There are also 600 testing inputs, for which no y-values are given; these will be used for validation against the Gradescope leaderboards.


## Examples of positive reviews include:
• (amazon) #1 It Works - #2 It is Comfortable.

• (imdb) "Gotta love those close-ups of slimy, drooling teeth! "

• (yelp) Food was so gooodd.

## Examples of negative reviews include:
• (amazon) DO NOT BUY DO NOT BUYIT SUCKS

• (imdb) This is not movie-making.

• (yelp) The service was poor and thats being nice.

# Project Analysys

**1. Bag of Words**

I started by analyzing the data with word counts, and looking at the frequency of certain words
across singular documents and across all documents. I realized it would be hard to use a simple
counter and have a precise metric of words to exclude, and what weight to give to each word.
Thus, I decided to use the inverse document frequency method.

I then looked at the kinds of problems words and sentences could have. It was quite obvious to
make it all lower case, and to remove special characters. However, some special characters are
still necessary, like for the word don't. I decided to remove all special characters besides ` and -
that are in between characters. I then stripped extra and trailing spaces.

I then noticed that many of the words were miss spelled. I decided to use nltk to correct the
works given by using Brown's dictionary of words. Lastly, I lemmatized the inputs, removing any
form of conjugation and other variances from words, also using nltk.

I also decided to use the stemming_tokenizer along with the TfidfVectorizer. For the vectorizer, I
decided to alter 3 parameters: min_df, max_df, and ngram_range. I decided to use a max_df of
.50, so words that appear in more than half of the documents are immediately discarded. I
decided to use a min_df of 0.001. I started by using 0.01, however that was including some
singular appearance words that I would like to avoid. Changing it to 0.001 seemed to take care
of that. Lastly, I used (1, 2) for the ngram_range. After analyzing some of the texts, I believe that
some words when combined give clear meaning, like “didn’t like” or “wanted more”.

I then decided to look for a list of stop_words to be used, and decided on the sklearn library list:
ENGLISH_STOP_WORDS

With this, I started part 2


**2. Logistic Regression**

The initial model built uses default values of the LogisticRegression. I decided to use this as a
benchmark for future optimizations. All the tests were done using and not using the Stop Words
list found

I decided to use a 10-fold-cross-validation method and measure the accuracy of the model. The
accuracy for the default model is

Without stop words -> accuracy: 0.
With stop words ->accuracy: 0.

I then decided to test the accuracy with the class_weight = ‘balanced’ parameter.

Without stop words -> accuracy: 0.
With stop words ->accuracy: 0.

It seems like there is a decrease in performance. I decided to not move forward with stop words, and
count only on the values passed as hyper-parameters to the Vectorizer.

Then, I decided to test a few hyper-parameters: C, Class_weight, and Solvers. I tested for 31
values of C, and the best result was:

accuracy: 0.83867, c: 3.

Lastly, I tested the many types of solvers, but not much changed despite altering the solvers:

accuracy: 0.83817, solver: newton-cg
accuracy: 0.83817, solver: lbfgs
accuracy: 0.83808, solver: liblinear
accuracy: 0.83817, solver: sag
accuracy: 0.83808, solver: saga

Since not much changed, I decided to stay with the default value.

Lastly, I tested different values of tolerance, and the best was:

accuracy: 0.83992, tol: 0.


These graphs show that there is a clear best value for both the Tolerance and the C values, and that
they “tank” at a certain value. There is no evidence of over-fitting at specific settings.


**3. Neural Network**

I started with a regular neural network model, which yielded accuracy:

```
Accuracy: 0.
```
I then decided to change the Hidden-layers. I did so in the following ways: I first tested the
accuracy for different numbers of neurons on a single layer network, and surprisingly, 10
neurons worked better than more neurons.

```
Accuracy:0.
Neurons: 10
Layers: 1
```
Due to my computer being quite slow to make runs, I decided to test a few combinations of
layers. Since a single 10 neurons layer worked so well, I wanted to test more layers. From 1 to
10, to be more precise. However, the number of neurons in each layer would be a random value
from 1 to 10 also. I would test 10 such networks, and choose the best. On each version of n
layers, a single network will be forced to have 10 neurons in every layer.

The best was using the minimum amount of layers, 2, and it only slightly out performed the
previous model:

```
Accuracy:0.
Neurons: 1, 6
Layers: 2
```
I then tested 2 layers with 10 neurons on each, but the result was worse:

```
Accuracy: 0.
Neurons: 10, 10
Layers: 2
```

I then analyzed the best performers of each number of layers, from 2 to 10, and noticed that 2 layers
is indeed the best:

I then decided to test logistic activation function (Relu is the default), but got a much worse
result:

```
Accuracy:0.
Neurons: 1, 6
Layers: 2
```
Then, I decided to vary the alphas with a logspace table from 10e-7 to 1 and 20 entries

```
Accuracy:0.
Alpha: 0.
Layers: (1, 6)
```

Here is a graph of the alphas tested. After analyzing the graph below, the values for alpha seem to
be quite varied, and maybe a smaller value could be better. However, due to the time it takes to run,
I settled down on the value found.

Lastly, I decided to test different learning rates. The default is the constant learning rate, and I
tested both invscaling and adaptive. I used the best values found so far for each, but kept the
power_t for invscaling default for now. The results were:

```
Accuracy:0.
Alpha: 0.
Layers: (1, 6)
Learning Rate: Invscaling
Power_t: 0.5 (Default)
```
```
Accuracy:0.0.
Alpha: 0.
Layers: (1, 6)
Learning Rate: adaptive
Power_t: 0.5 (Default)
```
Based on these results, I decided to move on with invscaling, but to test different values of Power_t.
I decided to test values in the range of 0 and 1 in increments of 0.05. The best result was:

```
accuracy: 0.
power_t: 0.
```

# 4. SVM optimized with SGD

I first tested the default parameters of the decision_tree_clasifier, and got an accuracy of0.8125,
which is the highest yet for default values.

I then decided to test squared_hinge instead of the default hinge, but it yielded worse results with an
accuracy of 0.8033.

I then decided to test different values for alpha and tolerance. The results for tolerance were:

```
accuracy: 0.
tol: 0.
```
And the results for alpha were:

```
accuracy: 0.82229,
alpha: 0.
```
The graph for tolerance was quite everywhere:


And the graph for alpha was as expected. As alpha gets bigger, we seem to overtrain the model a
bit:


**5. Best Model**

The best performing model was, in terms of raw accuracy, the logistic regression. Logistic
Regression is much faster to train, and much easier to optimize. I believe a better result could have
been achieved with the neural network, however a better computer capable of running tests at a
faster rate and simulating more layers with more neurons would be extremely helpful. Better results
could also have been achieved by boosting the model. I thought of using something like a stacking
classifier with two different models, the current logistic regression model, and one optimized to
classify based on context, which is the biggest mistake that is being made.

Another possible reason for Logistic Regression yielding better results is the fact that the
pre-processing was not very accurate, and the stop-words were also not optimized to its full extent.
There are ways of making this work better, however I could not achieve such a result.

The model accuracy is of 0.83959, with hyper parameters:

```
tol: 0.
c: 3.
class_weight = 'balanced'
```
To further analyze what could be improved and the mistakes it is making, I first outputted a confusion
matrix of the data:

```
True False
```
```
True 1163 37
```
```
False 58 1142
```
As we can see, there is quite an even distribution of False positives and False Negatives in the data.
It seems to get it wrong whenever the sentence has a word that, without context, would classify it as
“bad”, but in reality does not. It also gets it wrong if the word is similar to a more used “positive” or
“negative” word. For example, “displeased” seems to be a word that is constantly wrong.

```
● False Positive: “ very displeased”
● False Negative : “i did not have any problem with thisitem and would order it again if needed”
● False Positive: “ it failed to convey the broad sweepof landscapes that were a great part of
the original”
```
Whenever there are no words that make it easy to classify, it also gets it wrong. For example:

```
● False Negative: “ the things that the four kids getthemselves into is absolutely hilarious to
watch”
```

**6. Analyzing Results**

The performance is pretty much as expected. An errorrate of around 83% is found. This shows that
the cross validation method did a really good job in predicting real error, and that the model was not
over trained. This also demonstrates opportunities for improving the model, since its accuracy is not
even at 90%. The errors on the actual data are most likely from similar issues to the errors on
training data, which are mentioned above. In order to account for those, maybe using a boosting
method where the model looks a bit more closely to a bigger combination of words.


