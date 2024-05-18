import nltk
from spellchecker import SpellChecker
from text_preprocessing import _get_words_tokenize
from nltk.tokenize import word_tokenize

def spell_check_query(query: str) -> str:
    spell = SpellChecker()

    tokens = word_tokenize(text)

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

    return ' '.join(corrected_tokens)


# def spell_check_query(query: str) -> str:
#     tokens = _get_words_tokenize(query)
#     spell = SpellChecker()
#     wrong_words = spell.unknown(tokens)
#     for i,token in enumerate(tokens):
#         if token in wrong_words:
#             correct_word = spell.correction(token)
#             if(correct_word != None):
#                 tokens[i] = correct_word
#     corrected_query = ' '.join(tokens)
#     return corrected_query
