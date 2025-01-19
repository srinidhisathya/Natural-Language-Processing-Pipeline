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

def preprocess_file(contents):
    # Preprocessing the contents.
    
    # Defining the variations of Table of content headers
    table_var = ['CONTENTS.','CONTENTS','Contents','Index','Table of Contents','TABLE OF CONTENTS']
    end_stopper = '\n\n\n' # For the end of the section we define a marker value
    
    # Locating the position of the beginning of the segment
    start_index = next((contents.find(variation) for variation in table_var if variation in contents), -1)

    # Identify the position of the end marker occurring after the start of the Table of contents section
    end_index = contents.find(end_stopper, start_index) if start_index != -1 else -1

    # If both the start and end markers are discovered, eliminate the Table of contents segment
    updated_text = contents[end_index:] if start_index != -1 and end_index != -1 else contents
    
    return updated_text


def get_chapters(contents_novel):
    # Get the chapters from the book
    
    # Define Delimiter
    delimiter = '\n'
   
    chapters = {}
    
    label_types = ['book', 'part', 'section', 'volume']
    # Current Book, Part, Section and Volume label
    labels = {label_type: None for label_type in label_types}    
    # Establishing a RE patter for chapter headers
    chapter_regex = r'(?:CHAPTER|Chapter|Chap\.?|CHAP\.?)\s+((?:\d+|(?:[IVXLCDM]+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)|(?:thirty(?:-[A-Z]+)?|forty(?:-[A-Z]+)?|fifty)))(?:\.[-.:]*)?\s*\n*(.*?)(?=\s*(?:Book|Volume|Part|Section|ACT|$|\s+\.))'

    # Using delimiter to split the book contents
    splitted_line = re.split(delimiter, contents_novel)

    # Iterating through each line 
    for i in range(len(splitted_line)):
        # Retrieving the current line and removing any whitespace before and after it
        line = splitted_line[i].strip()

        # Iterating over each label type- Book, Part, Section, Volume 
        
        for label_type in label_types:
            # Establishing the RE pattern for the current label type
            label_pattern_regex = (
                r'(?:_?' + label_type.capitalize() + r'|_?' + label_type.upper() + r')\s+'
                r'(\d+|[IVXLCDM]+|ONE|One|TWO|Two|THREE|Three|FOUR|Four|FIVE|Five|SIX|Six|SEVEN|Seven|EIGHT|Eight|NINE|Nine|TEN|Ten|ELEVEN|Eleven|TWELVE|Twelve|'
                r'THIRTEEN|Thirteen|FOURTEEN|Fourteen|FIFTEEN|Fifteen|SIXTEEN|Sixteen|SEVENTEEN|Seventeen|EIGHTEEN|Eighteen|NINETEEN|Nineteen|TWENTY|Twenty|'
                r'TWENTY ONE|Twenty one|TWENTY TWO|Twenty two|TWENTY THREE|Twenty three|TWENTY FOUR|Twenty four|TWENTY FIVE|Twenty five|)\.?\s*'
            )
            # Attempting to find a match for the current label type in the line
            match = re.match(label_pattern_regex, line)
            if match:
                # Updating the corresponding label entry with the matched text, enclosed in parentheses
                labels[label_type] = f'({match.group().strip()})'
                break

        # Iterating over each match found in the line using the chapter_regex pattern
        matches = [match for match in re.finditer(chapter_regex, line)]
        for match in matches:
            # Extract chapter number and title
            index_match, title_match = match.group(1, 2) if match else (None, None)
            index_match = index_match.strip() if index_match else None
            title_match = title_match.strip() if title_match else None

            # Building chapter information using the most recent labels
            chap_entry = ' '.join(filter(None, [labels[index] for index in ['book', 'volume', 'part', 'section']] + [index_match]))

            if not title_match:
                for count in range(i + 1, len(splitted_line)):
                    # Getting the next line and remove leading/trailing whitespaces
                    next_line = splitted_line[count].strip()
                    # Checking if the line contains alphabetical characters and is not empty
                    if next_line and re.match(r'[a-zA-Z]', next_line):
                        title_match = next_line
                        break
                    
            # Assigning the chapter prefix, index and its title to the chapters dictionary
            chapters[chap_entry] = title_match
    return chapters


def exec_regex_toc(file1_book, file2_book, file3_book):
    
    file_paths = [file1_book, file2_book, file3_book]
    contents = []

    # Read content of each book file, preprocess, and extract chapters
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        contents.append(preprocess_file(content))

    # Extracting the chapters from the book contents after it is preprocessed
    table_of_contents = [get_chapters(content) for content in contents]

    # Assigning the table of contents to respective variables- dictTOCs
    dictTOC1, dictTOC2, dictTOC3 = table_of_contents

	# DO NOT CHANGE THE BELOW CODE WHICH WILL SERIALIZE THE ANSWERS FOR THE AUTOMATED TEST HARNESS TO LOAD AND MARK
    writeHandle = codecs.open( 'toc1.json', 'w', 'utf-8', errors = 'replace' )
    strJSON = json.dumps( dictTOC1, indent=2 )
    writeHandle.write( strJSON + '\n' )
    writeHandle.close()
    
    writeHandle = codecs.open( 'toc2.json', 'w', 'utf-8', errors = 'replace' )
    strJSON = json.dumps( dictTOC2, indent=2 )
    writeHandle.write( strJSON + '\n' )
    writeHandle.close()
    
    writeHandle = codecs.open( 'toc3.json', 'w', 'utf-8', errors = 'replace' )
    strJSON = json.dumps( dictTOC3, indent=2 )
    writeHandle.write( strJSON + '\n' )
    writeHandle.close()

if __name__ == '__main__':
	if len(sys.argv) < 8 :
		raise Exception( 'missing command line args : ' + repr(sys.argv) )
	ontonotes_file = sys.argv[1]
	book1_file = sys.argv[2]
	chapter1_file = sys.argv[3]
	book2_file = sys.argv[4]
	chapter2_file = sys.argv[5]
	book3_file = sys.argv[6]
	chapter3_file = sys.argv[7]

	logger.info( 'ontonotes = ' + repr(ontonotes_file) )
	logger.info( 'book1 = ' + repr(book1_file) )
	logger.info( 'chapter1 = ' + repr(chapter1_file) )
	logger.info( 'book2 = ' + repr(book2_file) )
	logger.info( 'chapter2 = ' + repr(chapter2_file) )
	logger.info( 'book3 = ' + repr(book3_file) )
	logger.info( 'chapter3 = ' + repr(chapter3_file) )

	# DO NOT CHANGE THE CODE IN THIS FUNCTION

	exec_regex_toc( book1_file, book2_file, book3_file )
