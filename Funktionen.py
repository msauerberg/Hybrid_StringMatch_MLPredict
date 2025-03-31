import re
import Levenshtein
from fuzzywuzzy import fuzz
import numpy as np
import Levenshtein
import pandas as pd


# Alle Wörter im String, die weniger als 3 Buchstaben haben werden gelöscht
def remove_short_words(s):
    words = [word for word in s.split() if len(word) >= 3]
    out = " ".join(words)
    return out


# Diese Wörter werden ebenfalls entfernt
def remove_unwanted_words(s):
    unwanted_words_pattern = (
        r"wöchentlich|weekly|woche|allgemein|entsprechend|beendet|zyklus|version|"
        r"bis|mg|kg|m2|bezeichnet|entfällt|watch & wait|watch and wait"
    )
    s = re.sub(unwanted_words_pattern, "", s, flags=re.IGNORECASE)
    return s


# Da "5 FU" durch die Funktion "remove_short_words()" entfernt werden würde,
# habe ich diese strings vorher in die gemeinte Substanz übersetzt
def find_5FU(s):
    fluorouracil_pattern = (
        r"5 fu|5fu|5-fu|5_fu|Fluoruracil|flourouracil|5-fluoruuracil|"
        r"5-fluoro-uracil|5-fluoruuracil|5-fluoruracil|floururacil|"
        r"5-fluorounacil|flourouraci|5-fluourouracil"
    )
    s = re.sub(fluorouracil_pattern, "fluorouracil", s, flags=re.IGNORECASE)
    return s


def find_gemcitabin(s):
    gemcitabin_pattern = r"Gemcibatin|Gemcibatine|Gemcibatine Mono|Gemcibatin Mono"
    s = re.sub(gemcitabin_pattern, "gemcitabin", s, flags=re.IGNORECASE)
    return s


# Hier sollen Symbole wie das Copyright Symbol entfernt werden
def remove_special_symbols(s):
    special_symbols_pattern = r"[\u24C0-\u24FF\u2100-\u214F\u2200-\u22FF\u2300-\u23FF\u2600-\u26FF\u2700-\u27BF\u2B50\u2B06]|m²"

    return re.sub(special_symbols_pattern, "", s)


# Funktion zum Aufräumen der Input strings
def preprocessing_func(input_col, split_string=True, split_pattern=r"[;,]"):

    if split_string:
        if split_pattern is None:
            raise ValueError(
                "Bitte gibt ein gültiges split_pattern ein oder setze split_sting auf False"
            )
        try:
            re.compile(split_pattern)
        except re.error as e:
            raise ValueError(
                f"Das split_pattern {split_pattern} ist ungültig Error: {e}"
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
    # Wenn der string getrennt wird, gibt es mehrere Zeilen pro Meldung, sie sind durch die ID gruppierbar
    if split_string:
        rearranged_df["Processed"] = rearranged_df["Processed"].str.split(split_pattern)
        rearranged_df = rearranged_df.explode("Processed", ignore_index=True)

    return rearranged_df


def find_matches(processed_df, reference_df, fuzzy_threshold=90):
    reference_words = reference_df["Substanz"].tolist()

    exact_matches_list = []
    fuzzy_matches_list = []

    for text in processed_df["Processed"]:
        if not isinstance(text, str):
            text = ""

        # Exakte Treffer
        exact_matches = [
            word for word in reference_words if word.lower() == text.lower()
        ]
        exact_matches_list.append(exact_matches)

        # Fuzzy Treffer, wenn keine exakten Treffer gefunden
        if not exact_matches:
            fuzzy_matches = [
                word
                for word in reference_words
                if fuzz.partial_ratio(text, word) >= fuzzy_threshold
            ]
            fuzzy_matches_list.append(fuzzy_matches)
        else:
            fuzzy_matches_list.append([])

    # Alle Treffer werden in neuen Spalten gespeichert
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


# Funktion um Levenshtein in Prozent auszudrücken
def calculate_similarity_percentage(original, best_match):
    if not original or not best_match:
        return 0
    distance = Levenshtein.distance(original, best_match)
    max_len = max(len(original), len(best_match))
    return round((1 - distance / max_len) * 100, 2)


# Von allen Treffern soll nur der beste ausgewählt werden
def calculate_best_match(processed_df, reference_df, split_string=True):
    best_matches = []
    lowest_distances = []

    # Bester Treffer ist definiert als niedrigste Levenshein Distanz
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

    # Speichern vom besten Treffer
    processed_df["Best_match"] = best_matches
    processed_df["LowestDistance"] = lowest_distances

    # Der beste Treffer wird anhand der Referenztabelle mit dem dazugehörigen ATC verknüpft
    processed_df = processed_df.merge(
        reference_df[["Substanz"]],
        left_on="Best_match",
        right_on="Substanz",
        how="left",
    ).drop(columns=["Substanz"])

    # Wähle Spalten aus
    processed_df = processed_df[["ID", "Original", "Processed", "Best_match"]]

    # Hier wird wieder eine Zeile pro ID erstellt
    # Nur notwenig, wenn die strings vorher getrennt wurden

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
