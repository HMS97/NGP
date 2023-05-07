
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import RegexpTokenizer, word_tokenize
import spacy

def custom_sent_tokenize(text):
    # Use a regular expression to split sentences on '.', '?', '!', and ':'
    sentence_tokenizer = RegexpTokenizer('[^.!?:,]+')
    return sentence_tokenizer.tokenize(text)


def count_target_elements(text_list, target_elements, sentence_similarity_threshold=0.89):
    # Convert the target elements list to a set for faster lookup
    target_elements_set = set(target_elements)
    spacy.prefer_gpu()
    try:
        nlp = spacy.load("en_core_web_md")  # make sure to use larger package!
    except OSError:
        print('Downloading language model for the spaCy POS tagger\n'
            "(don't worry, this will only happen once)")
        from spacy.cli import download
        download('en_core_web_md')
        nlp = spacy.load('en_core_web_md')
        
    target_elements_count = 0
    target_elements = [nlp(str(i)) for i in target_elements]

    for part_text in text_list:
        status = False

        # Tokenize the text into words and sentences
        words = nltk.word_tokenize(part_text)
        sentences = custom_sent_tokenize(part_text)
        sentences = [nlp(i.lower()) for i in sentences if 'frame' not in i.lower() and 'says' not in i.lower()]

        # Count the occurrences of each target element
        for word in words:
            if word.lower() in target_elements_set:
                status = True
                target_elements_count += 1
                break

        if status:
            continue

        # Check for (partially) matching target sentences
        for sentence in sentences:
            if status:
                break
            for target_sentence in target_elements:
                try:
                    score = sentence.similarity(target_sentence)
                except:
                    print('error', target_sentence, sentence)
                if score >= sentence_similarity_threshold:
                    status = True
                    target_elements_count += 1
                    break

    return target_elements_count