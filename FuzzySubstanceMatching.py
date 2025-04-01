import re
import Levenshtein
from fuzzywuzzy import fuzz
import numpy as np
import Levenshtein
import pandas as pd


def remove_short_words(s):
    """removes words with less than 3 characters

    Args:
        s (string): string from free text field

    Returns:
        string: input string without short words
    """
    words = [word for word in s.split() if len(word) >= 3]
    out = " ".join(words)
    return out


def remove_unwanted_words(s):
    """removes common words that we dont want for string matching

    Args:
        s string: string from free text field

    Returns:
        string: input string without unwanted words
    """
    unwanted_words_pattern = (
        r"wöchentlich|weekly|woche|allgemein|entsprechend|beendet|zyklus|version|"
        r"bis|mg|kg|m2|bezeichnet|entfällt|watch & wait|watch and wait"
    )
    s = re.sub(unwanted_words_pattern, "", s, flags=re.IGNORECASE)
    return s


def find_5FU(s):
    """5FU is a common abbreviation for Fluorouracil. The functions finds it and replaces it with the full name.

    Args:
        s (string): input string from free text field

    Returns:
        string: Same string or string with 5-FU replaced by full name
    """
    fluorouracil_pattern = (
        r"5 fu|5fu|5-fu|5_fu|Fluoruracil|flourouracil|5-fluoruuracil|"
        r"5-fluoro-uracil|5-fluoruuracil|5-fluoruracil|floururacil|"
        r"5-fluorounacil|flourouraci|5-fluourouracil"
    )
    s = re.sub(fluorouracil_pattern, "fluorouracil", s, flags=re.IGNORECASE)
    return s


def find_gemcitabin(s):
    """To fix common typos for Gemcitabin

    Args:
        s (string): input string from free text field

    Returns:
        string: Same string or string with fixed typo
    """
    gemcitabin_pattern = r"Gemcibatin|Gemcibatine|Gemcibatine Mono|Gemcibatin Mono"
    s = re.sub(gemcitabin_pattern, "gemcitabin", s, flags=re.IGNORECASE)
    return s


def remove_special_symbols(s):
    """removes common symbols that hinder matching

    Args:
        s (string): input string from free text field

    Returns:
        string: Same string without symbols
    """
    special_symbols_pattern = r"[\u24C0-\u24FF\u2100-\u214F\u2200-\u22FF\u2300-\u23FF\u2600-\u26FF\u2700-\u27BF\u2B50\u2B06]|m²"

    return re.sub(special_symbols_pattern, "", s)


def preprocessing_func(input_col, split_string=True, split_pattern=r"[;,]"):
    """Applies all the little preprocessing functions from above to the the input column

    Args:
        input_col (PandaSeries): This is the PandasSeries input column with free text
        split_string (bool, optional): To split the string in case more substances are listed in one row. Defaults to True.
        split_pattern (regexp, optional): Put the regex for splitting. Defaults to r"[;,]" means split by ; and ,

    Raises:
        ValueError: In case user sets split_string to True but does not give a regex
        ValueError: In case user provides a wrong regex

    Returns:
        PandasDataFrame: Processed string, in case split_string = True than we add rows, i.e., split string from one row into multiple rows
    """
    if split_string:
        if split_pattern is None:
            raise ValueError(
                "Please provide a valid regex for string_split or set it to False"
            )
        try:
            re.compile(split_pattern)
        except re.error as e:
            raise ValueError(
                f"The provided regex {split_pattern} is not valid. Error: {e}"
            )

    input_col = input_col.fillna("")
    subs = input_col.astype(str).str.lower()
    subs = subs.apply(find_5FU)
    subs = subs.apply(find_gemcitabin)
    replace_slash = re.compile(r"[/]")
    subs = subs.apply(lambda x: replace_slash.sub(";", x))
    subs = subs.apply(remove_unwanted_words)
    unwanted_chars = re.compile(r"[()\[\]><:_/\.+]")
    subs = subs.apply(lambda x: unwanted_chars.sub(" ", x))
    remove_patterns = re.compile(
        r"\b(?:o\.n\.a\.|o\.n\.a|mg|kg|i\.v\.|i\.v)\b", re.IGNORECASE
    )
    subs = subs.apply(lambda x: remove_patterns.sub("", x))
    subs = subs.str.replace(r"\s+", " ", regex=True).str.strip()
    subs = subs.apply(remove_special_symbols)
    processed = subs.apply(remove_short_words)

    rearranged_df = pd.DataFrame(
        {
            "ID": range(1, len(input_col) + 1),
            "Original": input_col,
            "Processed": processed,
        }
    )

    if split_string:
        rearranged_df["Processed"] = rearranged_df["Processed"].str.split(split_pattern)
        rearranged_df = rearranged_df.explode("Processed", ignore_index=True)

    return rearranged_df


def find_matches(processed_df, reference_series, fuzzy_threshold=90):
    """Find exact and fuzzy matches

    Args:
        processed_df (PandasDataFrame): The output of preprocessing_func()
        reference_series (PandasSeries): The reference substances
        fuzzy_threshold (int, optional): The threshold parameter for fuzzy matching. Defaults to 90.

    Returns:
        PandasDataFrame: Dataset with all found matches in wide format
    """
    reference_words = reference_series

    exact_matches_list = []
    fuzzy_matches_list = []

    for text in processed_df["Processed"]:
        if not isinstance(text, str):
            text = ""

        # Exact matches
        exact_matches = [
            word for word in reference_words if word.lower() == text.lower()
        ]
        exact_matches_list.append(exact_matches)

        # Fuzzy matches, if no exact match is found
        if not exact_matches:
            fuzzy_matches = [
                word
                for word in reference_words
                if fuzz.partial_ratio(text, word) >= fuzzy_threshold
            ]
            fuzzy_matches_list.append(fuzzy_matches)
        else:
            fuzzy_matches_list.append([])

    # Add to data in wide format
    max_exact_matches = (
        max(len(matches) for matches in exact_matches_list) if exact_matches_list else 0
    )
    max_fuzzy_matches = (
        max(len(matches) for matches in fuzzy_matches_list) if fuzzy_matches_list else 0
    )

    # Use dictionaries to store columns before merging (avoid direct insertions)
    exact_match_columns = {}
    fuzzy_match_columns = {}

    for i in range(max_exact_matches):
        column_name = f"Match{i+1}"
        exact_match_columns[column_name] = [
            matches[i] if i < len(matches) else "" for matches in exact_matches_list
        ]

    for i in range(max_fuzzy_matches):
        column_name = f"FuzzyMatch{i+1}"
        fuzzy_match_columns[column_name] = [
            matches[i] if i < len(matches) else "" for matches in fuzzy_matches_list
        ]

    # Use pd.concat() to add all new columns at once, avoiding fragmentation
    processed_df = pd.concat([processed_df, pd.DataFrame(exact_match_columns)], axis=1)
    processed_df = pd.concat([processed_df, pd.DataFrame(fuzzy_match_columns)], axis=1)

    return processed_df


def calculate_similarity_percentage(original, match_found):
    """calculates relatative Levenshtein Distance

    Args:
        original (PandaSeries): The column with the original string from free text field
        best_match (PandaSeries): The column with a detected match

    Returns:
        The relative Levenshtein distance between original input and detected match
    """
    if not original or not match_found:
        return 0
    distance = Levenshtein.distance(original, match_found)
    max_len = max(len(original), len(match_found))
    return round((1 - distance / max_len) * 100, 2)


def calculate_best_match(processed_df, split_string=True):
    """calculates the best match from all matches found

    Args:
        processed_df (PandasDataFrame): The output from find_matches()
        split_string (bool, optional): The same setting as for the previous function. Defaults to True.

    Returns:
        PandasDataFrame: Adds the best match and the corresponding distance to orignal input string
    """
    best_matches = []
    lowest_distances = []

    # Best match is calculated as the lowest Levenshtein distance
    for _, row in processed_df.iterrows():
        processed_text = row["Processed"]
        match_columns = [
            col
            for col in processed_df.columns
            if col.startswith("Match") or col.startswith("FuzzyMatch")
        ]

        distances = {}
        for col in match_columns:
            match_word = row[col]
            if match_word:
                dist = calculate_similarity_percentage(
                    processed_text.lower(), match_word.lower()
                )
                distances[match_word] = dist

        if distances:
            best_match = max(distances, key=distances.get)
            lowest_distance = distances[best_match]
            best_matches.append(best_match)
            lowest_distances.append(lowest_distance)
        else:
            best_matches.append("")
            lowest_distances.append(np.nan)

    # Save the best match
    processed_df["Best_match"] = best_matches
    processed_df["LowestDistance"] = lowest_distances
    processed_df = processed_df[["ID", "Original", "Processed", "Best_match"]]

    # Collapse dataframe rows by ID in case it was splitted before
    if split_string:
        collapsed_df = (
            processed_df.groupby("ID")
            .agg(
                {
                    "Original": "first",
                    "Processed": lambda x: "; ".join(
                        map(str, filter(None, x.fillna("")))
                    ),
                    "Best_match": lambda x: "; ".join(
                        dict.fromkeys(map(str, filter(None, x.fillna(""))))
                    ),
                }
            )
            .reset_index()
        )
    else:
        collapsed_df = processed_df

    # Berechne die Levenshtein Distanz in Prozent
    collapsed_df["LevenshteinPercent"] = collapsed_df.apply(
        lambda row: calculate_similarity_percentage(row["Original"], row["Best_match"]),
        axis=1,
    )

    return collapsed_df


def get_substances(
    input_col,
    reference_series,
    split_string=True,
    split_pattern=r"[;,]",
    fuzzy_threshold=90,
):
    """Final function for (fuzzy) string matching

    Args:
        input_col (PandasSeries): The column of a PandasDataFrame with strings, i.e., the free text field for substances
        reference_series (PandasSeries): The column of a PandasDatFrame with reference substances, i.e., those substances that should be found in the free text field
        split_string (bool, optional): User can split the string to account for more substances in one field. Defaults to True.
        split_pattern (regexp, optional): User can decide when to split, e.g., comma, semicolon. Defaults to r"[;,]".
        fuzzy_threshold (int, optional): User can set the threshold for fuzzy matching. Defaults to 90.

    Raises:
        ValueError: The length of the input substance file must be equal to the length of the output substance file 

    Returns:
        PandasDataFrame: DataFrame with ID, original input string, best match, and corresponing similarity measure score
    """

    clean_data = preprocessing_func(
        input_col=input_col, split_string=split_string, split_pattern=split_pattern
    )

    matches_found = find_matches(
        processed_df=clean_data,
        reference_series=reference_series,
        fuzzy_threshold=fuzzy_threshold,
    )

    best_matches = calculate_best_match(
        processed_df=matches_found, split_string=split_string
    )

    out = best_matches.rename(
        columns={"Best_match": "Predicted", "LevenshteinPercent": "Similarity"}
    )
    
    if len(input_col) != len(out):
        raise ValueError(f"Something went wrong: Length of input is {len(input_col)} and length of output is {len(out)}")
    
    return out.drop(columns=["Processed"])
