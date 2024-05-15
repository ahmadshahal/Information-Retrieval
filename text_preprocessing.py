import re
from datetime import datetime

import string
import pycountry
from dateutil import parser
from nltk.corpus import stopwords, words
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker

def get_preprocessed_text_terms(text: str) -> list:
    """
    Apply text processing steps of a given text

    Args:
        text: The text you want to process
        dataset_name: The name of the dataset to use. Can be either "lifestyle" or "antique".

    Returns:
        A list of cleansing tokens
    """
    text = _remove_urls(text)
    text = _remove_punctuations(text)
    # 1) Tokenizing: extract tokens from the text
    tokens = _get_words_tokenize(text)
    # 2) Lowerization: convert all tokens to lowercase
    lowercase_tokens = _lowercase_tokens(tokens)
    filtered_tokens = _remove_stop_words(lowercase_tokens)
    d = _normalize_dates(filtered_tokens)
    c = _normalize_country_names(d)
    # 5) Stemming: stemming the tokens
    stemmed_tokens = _stem_tokens(c)
    lemmitized_tokens = _lemmatize_tokens(stemmed_tokens)

    return lemmitized_tokens


def _get_words_tokenize(text: str) -> list:
    return word_tokenize(text)


def _remove_urls(text: str) -> str:
    return re.sub("http[^\s]*", "", text, flags=re.IGNORECASE)


def _spell_check_tokens(tokens: list, query: bool):
    if query:
        spell = SpellChecker()

        word_set = set(words.words())

        # Create a list to store the corrected tokens
        corrected_tokens = []

        # Spell check each token
        for token in tokens:
            if token in word_set:
                corrected_tokens.append(token)
            else:
                # Find the highest-ranked suggestion using the spell-checker
                suggestions = spell.candidates(token)
                if suggestions:
                    corrected_tokens.append(spell.correction(token))
                else:
                    corrected_tokens.append(token)

        return corrected_tokens
    else:
        return tokens


def _lowercase_tokens(tokens: list) -> list:
    return [token.lower() for token in tokens]


def _remove_punctuations(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def _remove_stop_words(tokens: list) -> list:
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    return filtered_tokens


def _stem_tokens(tokens: list) -> list:
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens


def _normalize_dates(tokens: list):
    normalized_tokens = []
    for word in tokens:
        new_text = word
        try:
            dt = parser.parse(word)
            if isinstance(dt, datetime):
                date_obj = parser.parse(word)
                formatted_date = date_obj.strftime("%Y %m %d")
                day_name = date_obj.strftime("%A")
                month_name = date_obj.strftime("%B")
                time_obj = date_obj.time().strftime("%I %M %p")
                new_formatted = f"{formatted_date} {day_name} {month_name} {time_obj}"
                new_text = new_text.replace(word, new_formatted)
        except (ValueError, OverflowError):
            pass
        normalized_tokens.append(new_text)
    return normalized_tokens


def _normalize_country_names(tokens: list) -> list:
    # Create a set of country names for faster lookup
    # ex. {'USA', 'KSA'}
    country_codes = set(country.alpha_3 for country in pycountry.countries)
    
    # Loop over the tokens and update country names if they match a country name
    for token in tokens.copy():
        if token.upper() in country_codes:
            try:
                country = pycountry.countries.lookup(token.upper())
                tokens.remove(token)
                tokens.append(country.name) # 'USA' -> 'United States of America'
            except LookupError:
                pass

    # Return the updated list of tokens
    return tokens


def _lemmatize_tokens(tokens: list) -> list:
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens


__all__ = ['get_preprocessed_text_terms']
