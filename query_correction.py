import enchant
import nltk
from spellchecker import SpellChecker
from text_preprocessing import _get_words_tokenize

def correct_query(query: str):
    tokens = _get_words_tokenize(query)
    spell = SpellChecker()
    wrong_words = spell.unknown(tokens)
    for i,token in enumerate(tokens):
        if token in wrong_words:
            correct_word = spell.correction(token)
            if(correct_word != None):
                tokens[i] = correct_word
    corrected_query = ' '.join(tokens)
    return corrected_query

