import pandas as pd
import random
import string
    
def introduce_typos(text, typos_n=1):

    if len(text) > 4:   
        text_list = list(text)   
        for _ in range(typos_n):
            idx = random.randint(0, len(text_list) - 1)
            random_char = random.choice(string.ascii_lowercase + string.ascii_uppercase)
            text_list[idx] = random_char
        
        return ''.join(text_list)

def introduce_random_word(text, word_list):
    
    random_word = random.choice(word_list)
    text = text + ' ' + random_word
    
    return text

def create_labeled_train_data(train_data, word_list):

    typo_data = train_data.copy()
    typo_data['input_text'] = typo_data['input_text'].apply(introduce_typos)

    word_add_data = train_data.copy()
    word_add_data['input_text'] = word_add_data['input_text'].apply(introduce_random_word, word_list = word_list)

    labeled_data = pd.concat([train_data, typo_data, word_add_data], ignore_index=True)
    shuffled_df = labeled_data.sample(frac=1).reset_index(drop=True)

    return shuffled_df
