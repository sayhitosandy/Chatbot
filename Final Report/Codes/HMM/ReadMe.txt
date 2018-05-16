I have two  Dictionaries , tagAndNext , and tagToWordsDict, wordCount and tagCount which i populate as i go through the training data.

wordCount is a dictionary of words and their counts
tagCount is a  dictionary of tags and their counts

tagAndNext has keys as tag T and corresponding value is a list of tags T` that appear exactly after T.

tagToWordsDict has keys as tag T, and correspnding value as list of all words that have been tagged as T

Transition and observation likelihood matrices are calculated using the aboove 4 dictionaries, and are represented as 2D dictionaries.

Laplacian smoothing has been done to remove zero values from transition matrix.

Unknown words are assigned a low probability (1e-5).

I used 15 samples randomly selected from the database to test my system's accuracy, which were good enough.

Although my system is not very receptive to unknown terms, it does an okay job in the cases I've tested.
