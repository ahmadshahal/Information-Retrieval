import nltk
from spellchecker import SpellChecker
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, words
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize


def process_query(query: str) -> str:
    tokens = word_tokenize(query)
    spell_checked = _spell_check_query(tokens)
    synonyms = _expand_query_synonyms(spell_checked)
    spell_checked.extend(synonyms)
    return ' '.join(spell_checked)


def _expand_query_synonyms(tokens) -> list:
    expanded_query = ' '.join(tokens)
    for word in tokens:
        synonyms = wordnet.synsets(word)
        for syn in synonyms:
            for lemma in syn.lemmas():
                expanded_query += ' ' + lemma.name()
    return list(set(expanded_query.split(' ')))


def _spell_check_query(tokens: list) -> list:
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