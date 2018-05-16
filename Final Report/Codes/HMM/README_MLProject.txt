1.First run convertToConvPairs.py
2.Then run createEmmbeddingsOnConvPairs to create the embeddings of the corpus.Skip if already made so.
2.Then run dataPreprocessor.py and use pickle file from 1.
Data preprocessor will convert each word to word vector , and return a QAPairs_good with wordvectors instead of words.