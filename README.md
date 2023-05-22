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
