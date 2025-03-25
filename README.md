# Extracting substances from free text fields
The input text might be "Abarelix (i.v.) weekly". The aim is extracting only the substance names which are provided by a reference list.
Not only additional information but also abbrevations, synonyms, and typos can make the extractions difficult.
## Using string and fuzzy matching
First, the free text field will be preprocessed (removing certain words, formatting...). Then, string and fuzzy matching will be used to extract substances from the free text field. A reference list will be used to find those valid substances.

## Using a machine Learning model
We train a model using SentenceTransformers. The training dataset is based on the reference list but typos are introduced randomly or words are added randomly. In this way, we derive a training dataset with synthetic free input texts and correct labels. After training, the model will be used to make predictions for those cases where string matching was not successful.

## Final output
A csv file with the original free text field in the first column and the extracted substance(s) in the second column.
A third column includes a dummy variable, indicating whether the extracted substance is based on string matching or the prediction of the ML model.
The forth column provides a similarity measure. This is the Levenshtein Distance in case the extracted substance wass found by string matchting or
the cosine similarity in case the subsance name was predicted by the ML model. The metric helps identifying those substances that should be evaluated by an expert to fix potential erroneous results. Cases with a similarity score < 90 are flagged and should be checked by an expert. The result can then be included in the training data set to make the ML model more reliable. 

## How to run the program?
After installing the packages from the requirements.txt, it should be possible to run the .ipynb file. All data files come from a gitlab repo and can be loaded by their URL.

## To do
Improve the model, test with different datasets, split the input text by ; and , before making predictions, deal with common input text that are not substances, e.g., Placebo or wait and see