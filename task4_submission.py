# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import sys, codecs, json, math, time, warnings, re, logging
warnings.simplefilter( action='ignore', category=FutureWarning )

import nltk, numpy, scipy, sklearn, sklearn_crfsuite, sklearn_crfsuite.metrics

LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
logger = logging.getLogger( __name__ )
logging.basicConfig( level=logging.INFO, format=LOG_FORMAT )
logger.info('logging started')


# Function to load the dataset
def load_training_data():
    # Specify the dataset file path and range of files to read
    dataset = ontonotes_file
    range_file = 25000

    # Read the JSON file
    read_handle = codecs.open(dataset, 'r', 'utf-8', errors='replace')
    # Read the content of the file as a string
    file_read = read_handle.read()
    # Close the file handle
    read_handle.close()

    # Parse the JSON string into a Python dictionary
    string_json = json.loads(file_read)
    # Extract the keys (file names) from the dictionary
    file_list = list(string_json.keys())
    # Limit the number of files to be used for training
    file_list = file_list[:range_file]
    # Select the files for training
    train_files_list = file_list
    
    # Initialize a list to store tokenized entries
    list_token = []
    # Iterate over each file for training
    for read_handle in train_files_list:
        # Iterate over each sentence in the file
        for sent_index in string_json[read_handle]:
            # Initialize a list to store token entries for the sentence and a variable to track the last named entity type
            entry_token_list = []
            last_ne_type = None

            # Iterate over each token in the sentence
            for id_token in range(len(string_json[read_handle][sent_index]["tokens"])):
                # Extract the token string
                token_str = string_json[read_handle][sent_index]["tokens"][id_token]
                # Extract the part-of-speech tag for the token
                pos_tag = string_json[read_handle][sent_index]["pos"][id_token]
                # Initialize a variable to store the named entity tag for the token
                token_ner = None

                # Check if named entities are present in the sentence
                for ne, ne_info in string_json[read_handle][sent_index].get("ne", {}).items():
                    if isinstance(ne_info, dict):  # Ensure ne_info is a dictionary
                        # Check if the token is part of the named entity
                        if id_token in ne_info.get("tokens", []):
                            # Assign the named entity type to the token
                            token_ner = ne_info.get("type", "")
                            break

                # Determine the IOB format for the named entity tag
                if token_ner is not None:
                    if token_ner == last_ne_type:
                        iob_label_str = "I-" + token_ner
                    else:
                        iob_label_str = "B-" + token_ner
                else:
                    iob_label_str = ""

                # Update the last named entity type
                last_ne_type = token_ner
                # Append the token entry to the list
                entry_token_list.append((token_str, pos_tag, iob_label_str))

            # Append the token entries for the sentence to the token list
            list_token.append(entry_token_list)

    # Return the list containing the tokenized training data
    return list_token

def extract_character_names_regex(text):
    # Define regex pattern for character names
    pattern = r'''
        \b                  # Word boundary
        (?:                 # Non-capturing group for optional titles
            (?:
                Judge|Mr|Mrs|Ms|Miss|Drs?|Profs?|Sens?|Reps?|Attys?|Lt|Col|Gen|Messrs|Govs?|Adm|Rev|Maj|Sgt|Cpl|Pvt|Capt|Ave|Pres|Lieut|Hon|Brig|Co?mdr|Pfc|Spc|Supts?|Det|Mt|Ft|Adj|Adv|Asst|Assoc|Ens|Insp|Mlle|Mme|Msgr|Sfc
            )\.?)\s*
            |(?:The\s+Honorable\s+) # Titles like Mr., Mrs., Miss, Ms., Dr., Prof., Rev, Honorable
        )?                  # Make the titles optional
        \s*                 # Match optional whitespace
        (                   # Capturing group for the name
            [A-Z][a-z']+(?:\s+[A-Z][a-z']+)*  # Capture capitalized words with optional apostrophes
            (?:                               # Non-capturing group for optional suffixes
                \s+(?:Jr\.|Sr\.|III|IV|V|VI|VII|VIII|IX|X)  # Suffixes like Jr., Sr., III, IV, V, VI etc
            )?                                # Make the suffixes optional
        )                   # End of name group
        \b                  # Word boundary
    '''
    compiled_pattern = re.compile(pattern, re.VERBOSE | re.IGNORECASE)
    
    # Find all matches of the pattern in the text
    matches = compiled_pattern.findall(text)
    
    # Clean and format the matches
    formatted_matches = []
    for match in matches:
        # Remove any leading or trailing whitespace
        formatted_name = match.strip()
        # Remove any trailing periods or apostrophes
        formatted_name = formatted_name.rstrip('.').rstrip("'")
        # Convert to lowercase and append the formatted name to the list
        formatted_matches.append(formatted_name.lower())
    
    return formatted_matches


# Generate a word shape representation for a given word.
def word_shape(word):
    shape = re.sub(r'[A-Z]', 'X', word)
    shape = re.sub(r'[a-z]', 'x', shape)
    for char in word:
        if char.isupper():
            # Uppercase letter
            shape += 'X'
        elif char.islower():
            # Lowercase letter
            shape += 'x'
        elif char.isdigit():
            # Digit
            shape += 'd'
        else:
            # Non-alphanumeric character
            shape += char
    return re.sub(r'\d', 'd', shape)

# def get_word_shape(word):
#     shape = ''
#     for char in word:
#         if char.islower():
#             shape += 'x'  
#         elif char.isupper():
#             shape += 'X'  
#         elif char.isdigit():
#             shape += 'd'  
#         else:
#             shape += char  
    
#     return shape

TITLE_RE_PAT = re.compile(r'(Judge|Mr|Mrs|Ms|Miss|Drs?|Profs?|Sens?|Reps?|Attys?|Lt|Col|Gen|Messrs|Govs?|Adm|Rev|Maj|Sgt|Cpl|Pvt|Capt|Ave|Pres|Lieut|Hon|Brig|Co?mdr|Pfc|Spc|Supts?|Det|Mt|Ft|Adj|Adv|Asst|Assoc|Ens|Insp|Mlle|Mme|Msgr|Sfc)\.?', re.IGNORECASE)
stopwords = nltk.corpus.stopwords.words('english')


# Function to extract features for a word
def features_extract(token, index):
    # Extract the token and POS tag at the given index
    word = token[index][0]
    ner = token[index][1]
    # Initialize a dictionary to store the features
    features = {
        "bias": 1.0,  
        "word": word,  # Token itself
        "pos_tag": ner,  # POS tag of the token
        "word.lower()": word.lower(),  
        "word.isupper()": word.isupper(),  
        "word.istitle()": word.istitle(), 
        "word.suffix": word.lower()[-3:],  
        "word.isdigit()": word.isdigit(),     
        "pos[:2]": ner[:2],
        "word_length": len(word),  # Word length feature
        "has_hyphen": '-' in word,  # Presence of hyphen
        'word:title': True if re.match(TITLE_RE_PAT, word) else False,
        "has_apostrophe": "'" in word,  # Presence of apostrophe
        "word_shape": word_shape(word),  # Word shape feature
        "has_numbers": any(char.isdigit() for char in word),  # Presence of numbers
        'word:stopword': word.lower() in stopwords,
    }
    # Check if there is a previous token available
    if index > 0:
        # Extract the previous token and POS tag
        word_prev = token[index - 1][0]
        tag_prev = token[index - 1][1]
        # Add features for the previous token to the dictionary
        features.update({
            "-1:word.lower()": word_prev.lower(),  # Lowercased previous token
            "-1:word.isupper()": word_prev.isupper(),
            "-1:word.istitle()": word_prev.istitle(),  
            "-1:word.suffix": word_prev.lower()[-3:], 
            '-1:word.isdigit()': word_prev.isdigit(),
            "-1:pos_tag": tag_prev,  # POS tag of the previous token
            "-1:pos_tag[:2]": tag_prev[:2],
            '-1:word:shape': word_shape(word_prev),
            '-1:word:title': True if re.match(TITLE_RE_PAT, word_prev) else False,
            '-1:word:stopword': word_prev.lower() in stopwords,
        })
    else:
        # If no previous token is available, mark the beginning of the sequence
        features['BOS'] = True 

    # Check if there is a next token available
    if index < len(token) - 1:
        # Extract the next token and POS tag
        word_next = token[index + 1][0]
        tag_next = token[index + 1][1]
        # Add features for the next token to the dictionary
        features.update({
            "+1:word.lower()": word_next.lower(), 
            "+1:word.isupper()": word_next.isupper(),  
            "+1:word.istitle()": word_next.istitle(), 
            "+1:word.suffix": word_next.lower()[-3:], 
            '+1:word.isdigit()': word_next.isdigit(),
            "+1:pos_tag": tag_next,  # POS tag of the next token
            "+1:pos_tag[:2]": tag_next[:2],
            '+1:word:shape': word_shape(word_next),
            '+1:word.istitle()': word_next.istitle(),
        })
    else:
        # If no next token is available, mark the end of the sequence
        features['EOS'] = True

    return features


# Function to extract labels from a sentence
def labels_sent(sent_index):
    return [ner_tag for _, _, ner_tag in sent_index]


# Function to generate features for a sentence
def features_sent(sent_index):
    return [features_extract(sent_index, index) for index in range(len(sent_index))]


# Function to train a CRF model
def train_crf_model_func(X_train, Y_train, max_iter):
    # Initialize a CRF model with specified parameters
    crf_ner = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,  # L1 regularization coefficient
        c2=0.1,  # L2 regularization coefficient
        max_iterations=max_iter,  # Maximum number of iterations 
        all_possible_transitions=False 
    )
    # Fit the CRF model to the training data
    crf_ner.fit(X_train, Y_train)
    return crf_ner


# Function to execute the NER task
def train_crf_model():
    max_iter=100
    # Load the training dataset
    training_set = load_training_data() 
    # Extract features and labels for training
    X_train = [features_sent(index) for index in training_set]  
    Y_train = [labels_sent(index) for index in training_set]
    # Train a CRF model using the training data
    crf_ner_model = train_crf_model_func(X_train, Y_train, max_iter)
    return crf_ner_model

# Function to extract named entities
def extract_named_entities(text, crf_ner_model):
    # Tokenize the text into sentences
    sentence_text = nltk.sent_tokenize(text)

    # Tokenize each sentence into words and POS tag them
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentence_text]
    pos_tagged_sentences = [nltk.pos_tag(sent) for sent in tokenized_sentences]

    # Extract features for each sentence
    X_text = [features_sent(sent) for sent in pos_tagged_sentences]

    # Predict NE labels for each sentence using the CRF NER model
    Y_pred = crf_ner_model.predict(X_text)

    # Combine consecutive tokens with the same NE label
    combined_entities = []
    for sent_idx, sent_pred in enumerate(Y_pred):
        entities = []
        current_entity = []
        current_label = None
        for token_idx, (token, label) in enumerate(zip(tokenized_sentences[sent_idx], sent_pred)):
            label_type = label[2:] if label.startswith('B-') or label.startswith('I-') else None
            if label_type != current_label:
                if current_entity:
                    entities.append((' '.join(current_entity), current_label))
                current_entity = [token]
                current_label = label_type
            else:
                current_entity.append(token)
        if current_entity:
            entities.append((' '.join(current_entity), current_label))
        combined_entities.extend(entities)

    # Filter out named entities that are not in the specified types
    ne_tags_filter = {}
    for entity, label in combined_entities:
        if label in ["PERSON"]:
            ne_tags_filter.setdefault(label, []).append(entity)

    return ne_tags_filter


# Define the function to remove duplicates and leading and trailing special characters incase present
def remove_duplicates_and_special_chars(names_list):
    # Define the special characters to be removed
    special_characters = ':;,-+/()[]{}“‘”’"\''
    # Remove special characters and whitespace, then remove duplicates from the list
    cleaned_names = list(set("".join(char for char in name.lstrip('.').rstrip('.') if char not in special_characters).strip() for name in names_list))
    return cleaned_names


# Define a function to clean - post process
def post_process_names(names_list):
    # Remove duplicates and leading/trailing special characters
    cleaned_names = remove_duplicates_and_special_chars(names_list)
    # Define common words
    # common_words=["and", "but", "or","the", "a", "an","i", "me", "my", "myself", "we", "our","ours", "you", "your", "yours", "he", 
    #               "him", "his", "she", "her", "it", "its", "they", "them", "their", "theirs","on", "in", "at", "by", "for", "with", 
    #               "about", "to", "from", "up", "down", "of", "off", "out", "into", "over", "under", "above", "below", "between", "before", 
    #               "after", "behind", "through", "across", "against", "along", "among", "around", "beyond", "within", "without","is", "am", 
    #               "are", "was", "were", "will", "would", "could", "should", "have", "has", "had", "been", "do", "does", "did", "can", "may", 
    #               "might", "must", "shall", "will", "would", "could", "should","some", "any", "all", "no", "none", "not", "such", "so", 
    #               "as", "than", "like", "other", "another", "others", "several", "enough", "little", "quite", "rather", "very", "indeed",
    #               "here", "there", "where", "now", "then", "today", "yesterday", "tomorrow", "never", "certainly", "precisely", "kindly",
    #               "quite", "having", "yes", "besides", "however", "moreover", "nevertheless", "therefore", "thus", "yet", "ago", "hence", 
    #               "hither", "thither", "whither", "whence", "whenever", "wherever", "wheresoever", "whereafter", "whereat", "whereby", 
    #               "wherefore", "wherefrom", "wherein", "whereof", "whereto", "whereunto", "whereuppon", "wherewithal", "turn", "lie", 
    #               "barter", "tartar", "wood", "pray", "mademoiselle", "gesellschaft", "none", "stolen", "poor","rich","this", "that", 
    #               "these", "those", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday","january","february",
    #               "march","april", "may","june","july","august","september","october","november","december","zero","zeroes","one","ones",
    #               "two","twos","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen","fourteen","fifteen",
    #               "sixteen","seventeen","eighteen","nineteen","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety",
    #               "hundred","thousand","hundreds","thousands","lakh","lakhs","crore","crores","million","millions","who", "what", "where", 
    #               "when", "why", "how", "soon", "later", "early", "late", "always", "never","better", "best", "worse", "worst", "more", 
    #               "most", "less", "least", "each", "every", "anyone", "anything", "something", "nothing", "everything", "nobody", "somebody", 
    #               "anybody", "everyone", "no one", "someone","now","d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", 
    #               "doesn", "hadn", "hasn", "haven", "isn","ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren", 
    #               "won", "wouldn"]
    
    filtered_names = [name for name in cleaned_names if name not in None and len(name) > 3]
    return filtered_names
   
   
# Perform post-processing and filtering to the characters
def extract_character_names(text, crf_ner_model):
    ne_tags_filter = extract_named_entities(text, crf_ner_model)
    character_names = ne_tags_filter.get("PERSON", [])
    # Extract character names using regex
    regex_names = extract_character_names_regex(text)
    # Combine and remove duplicates
    all_names = list(set(character_names + regex_names))
    # Apply post-processing and filtering
    filtered_names = post_process_names(all_names)
    return filtered_names


def exec_ner( file1_chapter = None, file2_chapter = None, file3_chapter = None, ontonotes_file = None ) :

    # CHANGE CODE BELOW TO TRAIN A NER MODEL AND/OR USE REGEX GENERATE A SET OF BOOK CHARACTERS AND FILTERED SET OF NE TAGS (task 4)

    # Input >> www.gutenberg.org sourced plain text file for a chapter of a book
    # Output >> characters.txt = plain text set of extracted character names. one line per character name.

    # hardcoded output to show exactly what is expected to be serialized (you should change this)
    # only the allowed types for task 4 PERSON will be serialized
    
    # Define the named entity types to be extracted
    ne_types = ["PERSON"]

    dictNE1 = {ne_type: [] for ne_type in ne_types}
    dictNE2 = {ne_type: [] for ne_type in ne_types}
    dictNE3 = {ne_type: [] for ne_type in ne_types}

    # Mapping chapter indices to their respective file paths and NE dictionaries
    values_dict = {1: file1_chapter, 2: file2_chapter, 3: file3_chapter}
    update_dict = {1: dictNE1, 2: dictNE2, 3: dictNE3}

    # Load the pre-trained CRF NER model
    crf_ner_model = train_crf_model()

    for index_chap in range(1, len(values_dict) + 1):
        # Read the chapter file
        read_handle = codecs.open(values_dict[index_chap], 'r', 'utf-8', errors='replace')
        file_read = read_handle.read()
        read_handle.close()
        read_text = file_read.strip()

        # Extract character names
        character_names = extract_character_names(read_text, crf_ner_model)

        # Update the "PERSON" list with the extracted character names
        update_dict[index_chap]["PERSON"].extend(character_names)

    # Remove special characters and duplicates from the final "PERSON" list for each chapter
    update_dict[1]["PERSON"] = remove_duplicates_and_special_chars(update_dict[1]["PERSON"])
    update_dict[2]["PERSON"] = remove_duplicates_and_special_chars(update_dict[2]["PERSON"])
    update_dict[3]["PERSON"] = remove_duplicates_and_special_chars(update_dict[3]["PERSON"])

    # DO NOT CHANGE THE BELOW CODE WHICH WILL SERIALIZE THE ANSWERS FOR THE AUTOMATED TEST HARNESS TO LOAD AND MARK

    # write out all PERSON entries for character list for subtask 4
    writeHandle = codecs.open('characters1.txt', 'w', 'utf-8', errors='replace')
    if 'PERSON' in dictNE1:
        for strNE in dictNE1['PERSON']:
            writeHandle.write(strNE.strip().lower() + '\n')
    writeHandle.close()

    writeHandle = codecs.open('characters2.txt', 'w', 'utf-8', errors='replace')
    if 'PERSON' in dictNE2:
        for strNE in dictNE2['PERSON']:
            writeHandle.write(strNE.strip().lower() + '\n')
    writeHandle.close()

    writeHandle = codecs.open('characters3.txt', 'w', 'utf-8', errors='replace')
    if 'PERSON' in dictNE3:
        for strNE in dictNE3['PERSON']:
            writeHandle.write(strNE.strip().lower() + '\n')
    writeHandle.close()

if __name__ == '__main__':
    if len(sys.argv) < 8:
        raise Exception('missing command line args : ' + repr(sys.argv))
    ontonotes_file = sys.argv[1]
    book1_file = sys.argv[2]
    chapter1_file = sys.argv[3]
    book2_file = sys.argv[4]
    chapter2_file = sys.argv[5]
    book3_file = sys.argv[6]
    chapter3_file = sys.argv[7]

    logger.info('ontonotes = ' + repr(ontonotes_file))
    logger.info('book1 = ' + repr(book1_file))
    logger.info('chapter1 = ' + repr(chapter1_file))
    logger.info('book2 = ' + repr(book2_file))
    logger.info('chapter2 = ' + repr(chapter2_file))
    logger.info('book3 = ' + repr(book3_file))
    logger.info('chapter3 = ' + repr(chapter3_file))

    # DO NOT CHANGE THE CODE IN THIS FUNCTION

    exec_ner(chapter1_file, chapter2_file, chapter3_file, ontonotes_file)
