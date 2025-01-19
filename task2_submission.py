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

def exec_regex_questions( file1_chapter = None, file2_chapter = None, file3_chapter = None ) :

    def get_question_lines(content_chap):
        # RE pattern for matching questions
        regex_question = r'(?:^|(?<=[.!?\\]))\s*([^\n?!.]*\?)'
        # Identifying all matches for the regex_question in the chapter text
        find_question = re.findall(regex_question, content_chap)
        # Removing whitespace and wrapping quotations from every question
        question_line_strip = {ques.strip().lstrip('“‘”’"\'').rstrip('“‘”’"\'').strip() for ques in find_question}
        return question_line_strip
       
    # Reading each chapter file
    setQuestions1 = set()
    setQuestions2 = set()
    setQuestions3 = set()

    for chapter_file, setQuestions in zip([file1_chapter, file2_chapter, file3_chapter], [setQuestions1, setQuestions2, setQuestions3]):
        with open(chapter_file, 'r', encoding='utf-8') as file:
            content_chap = file.read()

        # Rebuilding paragraphs
        line_rebuilt = ""
        for each_line in content_chap.splitlines():            
            # Eliminate any leading or trailing whitespaces and newlines
            each_line = each_line.strip()
            # Appending to the rebuilt text
            line_rebuilt += each_line + " "

        # Splitting the text into sentences with RE
        sentence_lines = re.split(r'(?<=[.!?])\s+(?![0-9])', line_rebuilt)

        # Joining successive lines until a period is met to make complete sentences
        sentence_line_final = []
        sentence_line_current = ""
        for sentence in sentence_lines:
            # If the sentence concludes with a lowercase letter followed by an uppercase letter, it is probably part of the same sentence
            if sentence and sentence[-1].islower() and (not sentence.endswith(("Mr.", "Mrs.", "Ms.", "Dr.", "etc."))):
                sentence_line_current += sentence + " "
            else:
                sentence_line_current += sentence
                sentence_line_final.append(sentence_line_current.strip())
                sentence_line_current = ""

        # Extracting questions from the complete sentences
        questions_chap = get_question_lines(" ".join(sentence_line_final))
        # Applying the condition to remove leading and trailing quotation marks from each question

        for question in questions_chap:
            question = question.strip().lstrip('“‘”’"\'').rstrip('“‘”’"\'').strip()
        
        # Assigning the respective questions to setQuestions1, setQuestions2, and setQuestions3
        setQuestions.update(questions_chap)

    # DO NOT CHANGE THE BELOW CODE WHICH WILL SERIALIZE THE ANSWERS FOR THE AUTOMATED TEST HARNESS TO LOAD AND MARK
                
    writeHandle = codecs.open( 'questions1.txt', 'w', 'utf-8', errors = 'replace' )
    for strQuestion in setQuestions1 :
        writeHandle.write( strQuestion + '\n' )
    writeHandle.close()
    writeHandle = codecs.open( 'questions2.txt', 'w', 'utf-8', errors = 'replace' )
    for strQuestion in setQuestions2 :
         writeHandle.write( strQuestion + '\n' )
    writeHandle.close()
    writeHandle = codecs.open( 'questions3.txt', 'w', 'utf-8', errors = 'replace' )
    for strQuestion in setQuestions3 :
         writeHandle.write( strQuestion + '\n' )
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

	exec_regex_questions( chapter1_file, chapter2_file, chapter3_file )

