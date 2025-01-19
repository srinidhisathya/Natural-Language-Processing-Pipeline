# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import sys, codecs, json, math, time, warnings, re, logging

warnings.simplefilter(action='ignore', category=FutureWarning)

import nltk, numpy, scipy, sklearn, sklearn_crfsuite, sklearn_crfsuite.metrics

LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger.info('logging started')

# Function to load the dataset
def load_training_data():
    # Specify the dataset file path and range of files to read
    dataset = ontonotes_file
    range_file = 25

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
        "pos[:2]": ner[:2] 
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
            "-1:pos_tag[:2]": tag_prev[:2] 
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
            "+1:pos_tag[:2]": tag_next[:2] 
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
    max_iter=20
    # Load the training dataset
    training_set = load_training_data() 
    # Extract features and labels for training
    X_train = [features_sent(index) for index in training_set]  
    Y_train = [labels_sent(index) for index in training_set]
    # Train a CRF model using the training data
    crf_ner_model = train_crf_model_func(X_train, Y_train, max_iter)
    return crf_ner_model


def exec_ner(file1_chapter=None, file2_chapter=None, file3_chapter=None, ontonotes_file=None):
    # CHANGE CODE BELOW TO TRAIN A CRF NER MODEL TO TAG THE CHAPTER OF TEXT (task 3)

    # Input >> www.gutenberg.org sourced plain text read_handle for a chapter of a book
    # Output >> ne.json = { <ne_type> : [ <phrase>, <phrase>, ... ] }
    
   # Initialize dictionaries to store Named Entities (NEs) for each chapter
    dictNE1 = {"CARDINAL": [], "ORDINAL": [], "DATE": [], "NORP": []}
    dictNE2 = {"CARDINAL": [], "ORDINAL": [], "DATE": [], "NORP": []}
    dictNE3 = {"CARDINAL": [], "ORDINAL": [], "DATE": [], "NORP": []}

    # Mapping chapter indices to their respective file paths and NE dictionaries
    values_dict = {1: file1_chapter, 2: file2_chapter, 3: file3_chapter}
    update_dict = {1: dictNE1, 2: dictNE2, 3: dictNE3}

     # Load the pre-trained CRF NER model
    crf_ner_model = train_crf_model()

    # Iterate over each chapter
    for index_chap in range(1, len(values_dict) + 1):
        # Read the chapter file
        read_handle = codecs.open(values_dict[index_chap], 'r', 'utf-8', errors='replace')
        file_read = read_handle.read()
        read_handle.close()
        read_text = file_read.strip()

        # Tokenize the text into sentences
        sentence_text = nltk.sent_tokenize(read_text)
        
        # Tokenize each sentence into words and POS tag them
        tokenize_word = [nltk.word_tokenize(word) for word in sentence_text]
        tokenize_word = [nltk.pos_tag(sentence_text) for sentence_text in tokenize_word]

        # Extract features for each sentence
        X_text = [features_sent(sent_index) for sent_index in tokenize_word]

        # Predict NE labels for each sentence using the CRF NER model
        Y_pred = crf_ner_model.predict(X_text)

        # Iterate over each sentence and predicted label
        for sent_index, sent_pred in enumerate(Y_pred):
            for index, pred_label in enumerate(sent_pred):
                # Extract token and label from the sentence
                token = tokenize_word[sent_index][index][0].lower()
                label = pred_label[2:]

                # Check if the label corresponds to allowed NE types
                if label == "CARDINAL" or label == "ORDINAL" or label == "NORP":
                    if token not in update_dict[index_chap][label]:
                        update_dict[index_chap][label].append(token)
                elif label == "DATE":
                    if pred_label == "B-DATE":
                        # Initialize the date string
                        date_str = token + " "
                        counter = index + 1
                        # Combine consecutive tokens labeled as I-DATE
                        while counter < len(sent_pred) and Y_pred[sent_index][counter] == "I-DATE":
                            date_str += tokenize_word[sent_index][counter][0].lower() + " "
                            counter += 1
                        if date_str.strip() not in update_dict[index_chap]["DATE"]:
                            update_dict[index_chap]["DATE"].append(date_str.strip())

    # DO NOT CHANGE THE BELOW CODE WHICH WILL SERIALIZE THE ANSWERS FOR THE AUTOMATED TEST HARNESS TO LOAD AND MARK

    # FILTER NE dict by types required for task 3
    listAllowedTypes = ['DATE', 'CARDINAL', 'ORDINAL', 'NORP']
    listKeys = list(dictNE1.keys())
    for strKey in listKeys:
        for nIndex in range(len(dictNE1[strKey])):
            dictNE1[strKey][nIndex] = dictNE1[strKey][nIndex].strip().lower()
        if not strKey in listAllowedTypes:
            del dictNE1[strKey]

    listKeys = list(dictNE2.keys())
    for strKey in listKeys:
        for nIndex in range(len(dictNE2[strKey])):
            dictNE2[strKey][nIndex] = dictNE2[strKey][nIndex].strip().lower()
        if not strKey in listAllowedTypes:
            del dictNE2[strKey]

    listKeys = list(dictNE3.keys())
    for strKey in listKeys:
        for nIndex in range(len(dictNE3[strKey])):
            dictNE3[strKey][nIndex] = dictNE3[strKey][nIndex].strip().lower()
        if not strKey in listAllowedTypes:
            del dictNE3[strKey]

    # write filtered NE dict
    writeHandle = codecs.open('ne1.json', 'w', 'utf-8', errors='replace')
    strJSON = json.dumps(dictNE1, indent=2)
    writeHandle.write(strJSON + '\n')
    writeHandle.close()

    writeHandle = codecs.open('ne2.json', 'w', 'utf-8', errors='replace')
    strJSON = json.dumps(dictNE2, indent=2)
    writeHandle.write(strJSON + '\n')
    writeHandle.close()

    writeHandle = codecs.open('ne3.json', 'w', 'utf-8', errors='replace')
    strJSON = json.dumps(dictNE3, indent=2)
    writeHandle.write(strJSON + '\n')
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
