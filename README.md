# Extracting substances from free text fields

## Using string and fuzzy matching
The free text field will be preprocessed. Then, string and fuzzy matching will be used to extract substances in the free text field.
A reference list will be used to determine which substances are valid and should be extracted from the free text field.

## Using a machine Learning model
We train a model using SentenceTransformers. This model will be used to make predictions in case string matching was not successful.

## Final output
A csv file with the original free text field in the first column and the extracted substance(s) in the second column.
A third column includes a dummy variable, indicating whether the extracted substance is based on string matching or the prediction of the ML model.
The forth column provides a similarity measure. This is the Levenshtein Distance in case the extracted substance wass found by string matchting or
the cosine similarity in case it was predicted by the ML model. The metric helps identifying those substances that should be evaluated by an expert to fix
potential erroneous results. 