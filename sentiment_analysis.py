#!/usr/bin/env python
import re, random, math, collections, itertools

PRINT_ERRORS=0

#This negation word list has been picked up from vaderSentiment from following site
#https://github.com/cjhutto/vaderSentiment/blob/master/vaderSentiment/vaderSentiment.py
NEGATE = \
    ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
     "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
     "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
     "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
     "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
     "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
     "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
     "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]

#------------- Function Definitions ---------------------


def readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia):

    #reading pre-labeled input and splitting into lines
    posSentences = open('rt-polarity.pos', 'r', encoding="ISO-8859-1")
    posSentences = re.split(r'\n', posSentences.read())

    negSentences = open('rt-polarity.neg', 'r', encoding="ISO-8859-1")
    negSentences = re.split(r'\n', negSentences.read())

    posSentencesNokia = open('nokia-pos.txt', 'r')
    posSentencesNokia = re.split(r'\n', posSentencesNokia.read())

    negSentencesNokia = open('nokia-neg.txt', 'r', encoding="ISO-8859-1")
    negSentencesNokia = re.split(r'\n', negSentencesNokia.read())
 
    posDictionary = open('positive-words.txt', 'r', encoding="ISO-8859-1")
    posWordList = re.findall(r"[a-z\-]+", posDictionary.read())

    negDictionary = open('negative-words.txt', 'r', encoding="ISO-8859-1")
    negWordList = re.findall(r"[a-z\-]+", negDictionary.read())
    
    for i in posWordList:
        sentimentDictionary[i] = 1
    for i in negWordList:
        sentimentDictionary[i] = -1

    #create Training and Test Datsets:
    #We want to test on sentences we haven't trained on, to see how well the model generalses to previously unseen sentences

  #create 90-10 split of training and test data from movie reviews, with sentiment labels    
    for i in posSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="positive"
        else:
            sentencesTrain[i]="positive"

    for i in negSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="negative"
        else:
            sentencesTrain[i]="negative"

    #create Nokia Datset:
    for i in posSentencesNokia:
            sentencesNokia[i]="positive"
    for i in negSentencesNokia:
            sentencesNokia[i]="negative"

#----------------------------End of data initialisation ----------------#

#calculates p(W|Positive), p(W|Negative) and p(W) for all words in training data
def trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord):
    posFeatures = [] # [] initialises a list [array]
    negFeatures = [] 
    freqPositive = {} # {} initialises a dictionary [hash function]
    freqNegative = {}
    dictionary = {}
    posWordsTot = 0
    negWordsTot = 0
    allWordsTot = 0

    #iterate through each sentence/sentiment pair in the training data
    for sentence, sentiment in sentencesTrain.items():
        wordList = re.findall(r"[\w']+", sentence)
        
        for word in wordList: #calculate over unigrams
            allWordsTot += 1 # keeps count of total words in dataset
            if not (word in dictionary):
                dictionary[word] = 1
            if sentiment=="positive" :
                posWordsTot += 1 # keeps count of total words in positive class

                #keep count of each word in positive context
                if not (word in freqPositive):
                    freqPositive[word] = 1
                else:
                    freqPositive[word] += 1    
            else:
                negWordsTot+=1# keeps count of total words in negative class
                
                #keep count of each word in positive context
                if not (word in freqNegative):
                    freqNegative[word] = 1
                else:
                    freqNegative[word] += 1

    for word in dictionary:
        #do some smoothing so that minimum count of a word is 1
        if not (word in freqNegative):
            freqNegative[word] = 1
        if not (word in freqPositive):
            freqPositive[word] = 1

        # Calculate p(word|positive)
        pWordPos[word] = freqPositive[word] / float(posWordsTot)

        # Calculate p(word|negative) 
        pWordNeg[word] = freqNegative[word] / float(negWordsTot)

        # Calculate p(word)
        pWord[word] = (freqPositive[word] + freqNegative[word]) / float(allWordsTot) 

#---------------------------End Training ----------------------------------

#implement naive bayes algorithm
#INPUTS:
#  sentencesTest is a dictonary with sentences associated with sentiment 
#  dataName is a string (used only for printing output)
#  pWordPos is dictionary storing p(word|positive) for each word
#     i.e., pWordPos["apple"] will return a real value for p("apple"|positive)
#  pWordNeg is dictionary storing p(word|negative) for each word
#  pWord is dictionary storing p(word)
#  pPos is a real number containing the fraction of positive reviews in the dataset
def testBayes(sentencesTest, dataName, pWordPos, pWordNeg, pWord,pPos):
    pNeg=1-pPos

    #These variables will store results (you do not need them)
    total=0
    correct=0
    totalpos=0
    totalpospred=0
    totalneg=0
    totalnegpred=0
    correctpos=0
    correctneg=0

    #for each sentence, sentiment pair in the dataset
    for sentence, sentiment in sentencesTest.items():
        wordList = re.findall(r"[\w']+", sentence)#collect all words

        pPosW=pPos
        pNegW=pNeg

        for word in wordList: #calculate over unigrams
            if word in pWord:
                if pWord[word]>0.00000001:
                    pPosW *=pWordPos[word]
                    pNegW *=pWordNeg[word]

        prob=0;            
        if pPosW+pNegW >0:
            prob=pPosW/float(pPosW+pNegW)


        total+=1
        if sentiment=="positive":
            totalpos+=1
            if prob>0.5:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
                if PRINT_ERRORS:
                    print ("ERROR (pos classed as neg %0.2f):" %prob + sentence)
        else:
            totalneg+=1
            if prob<=0.5:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
                if PRINT_ERRORS:
                    print ("ERROR (neg classed as pos %0.2f):" %prob + sentence)
    
    calculateMetricsForTestBayes(correct, total, correctpos, totalpospred, totalpos, correctneg, totalnegpred, totalneg, dataName)
 
 
# TODO for Step 2: Add some code here to calculate and print: (1) accuracy; (2) precision and recall for the positive class; 
# (3) precision and recall for the negative class; (4) F1 score;
def calculateMetricsForTestBayes(correct, total, correctpos, totalpospred, totalpos, correctneg, totalnegpred, totalneg, dataName):
        
        print("\nMetrics for: ", dataName)
        print("Accuracy: ", correct/total)
        
        precision_pos = correctpos/totalpospred
        recall_pos = correctpos/totalpos
        f1_score_pos = (2 * precision_pos * recall_pos) / (precision_pos + recall_pos)
        print("Precision for the positive class: ", precision_pos) 
        print("Recall for the positive class: ", recall_pos)
        print("F-Measure for the positive class: ",  f1_score_pos) 
        
        precision_neg = correctneg/totalnegpred
        recall_neg = correctneg/totalneg
        f1_score_neg = (2 * precision_neg * recall_neg) / (precision_neg + recall_neg)
        print("Precision for the negative class: ", precision_neg) 
        print("Recall for the negative class: ", recall_neg)
        print("F-Measure for the negative class: ",  f1_score_neg) 


# This is a simple classifier that uses a sentiment dictionary to classify 
# a sentence. For each word in the sentence, if the word is in the positive 
# dictionary, it adds 1, if it is in the negative dictionary, it subtracts 1. 
# If the final score is above a threshold, it classifies as "Positive", 
# otherwise as "Negative"
def testDictionary(sentencesTest, dataName, sentimentDictionary, threshold):
    total=0
    correct=0
    totalpos=0
    totalneg=0
    totalpospred=0
    totalnegpred=0
    correctpos=0
    correctneg=0
    for sentence, sentiment in sentencesTest.items():
        Words = re.findall(r"[\w']+", sentence)
        score=0
        for word in Words:
            if word in sentimentDictionary:
               score+=sentimentDictionary[word]
 
        total+=1
        if sentiment=="positive":
            totalpos+=1
            if score>=threshold:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
                if PRINT_ERRORS:
                    print ("ERROR (pos classed as neg %0.2f):" %score + sentence)
        else:
            totalneg+=1
            if score<threshold:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
                if PRINT_ERRORS:
                   print ("ERROR (neg classed as pos %0.2f):" %score + sentence)
                  
    calculateMetricsForTestDictionary(correct, total, correctpos, totalpospred, totalpos, correctneg, totalnegpred, totalneg, dataName)

# This is a negation check function which will return true incase a 
# negation word is encountered in the sentence  
def checkNegation(sentence):
    isNegation = False
    
    Words = re.findall(r"[\w']+", sentence)
    for word in Words:
        if word in NEGATE:
            isNegation = True

    return isNegation
 

# This is a modified classifier that uses a modified sentiment dictionary to classify 
# a sentence. For each word in the sentence, if the word is in the positive 
# dictionary, it adds 1, if it is in the negative dictionary, it subtracts 1. 
# If the word is in the most useful positive words it adds 2, if in most useful
# negative words, it subtracts -2
# It also performs a check if there is any negation word encountered 
# in the sentence from the list of NEGATE words defined above. If yes, 
# it changes the polarity of the final score 
# If the final score is above a threshold, it classifies as "Positive", 
# otherwise as "Negative"
def testDictionaryModified(sentencesTest, dataName, sentimentDictionary, threshold):
    total=0
    correct=0
    totalpos=0
    totalneg=0
    totalpospred=0
    totalnegpred=0
    correctpos=0
    correctneg=0
    
    for sentence, sentiment in sentencesTest.items():
        Words = re.findall(r"[\w']+", sentence)
        score=0
        for word in Words:
            if word in sentimentDictionary:
               score+=sentimentDictionary[word]
    
        isNegation = checkNegation(sentence)
        
        if(isNegation):
            score = -(score)

        total+=1
        if sentiment=="positive":
            totalpos+=1
            if score>=threshold:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
                if PRINT_ERRORS:
                    print ("ERROR (pos classed as neg %0.2f):" %score + sentence)
        else:
            totalneg+=1
            if score<threshold:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
                if PRINT_ERRORS:
                   print ("ERROR (neg classed as pos %0.2f):" %score + sentence)
                
    calculateMetricsForTestDictionary(correct, total, correctpos, totalpospred, totalpos, correctneg, totalnegpred, totalneg, dataName)
    
    
# TODO for Step 5: Add some code here to calculate and print: (1) accuracy; (2) precision and recall for the positive class; 
# (3) precision and recall for the negative class; (4) F1 score;

def calculateMetricsForTestDictionary(correct, total, correctpos, totalpospred, totalpos, correctneg, totalnegpred, totalneg, dataName):
    calculateMetricsForTestBayes(correct, total, correctpos, totalpospred, totalpos, correctneg, totalnegpred, totalneg, dataName)

                

#Print out n most useful predictors
def mostUseful(pWordPos, pWordNeg, pWord, n):
    predictPower={}
    for word in pWord:
        if pWordNeg[word]<0.0000001:
            predictPower=1000000000
        else:
            predictPower[word]=pWordPos[word] / (pWordPos[word] + pWordNeg[word])
            
    sortedPower = sorted(predictPower, key=predictPower.get)
    head, tail = sortedPower[:n], sortedPower[len(predictPower)-n:]
    print ("NEGATIVE:")
    print (head)
    print ("\nPOSITIVE:")
    print (tail)
    print ("Count of Negative words in sentimentDictionary:", len(set(head) & set(sentimentDictionary.keys())))
    print ("Count of Positive words in sentimentDictionary:", len(set(tail) & set(sentimentDictionary.keys())))
    return head, tail

#modification of sentiment dictionary
#INPUTS:
#  sentimentDictionaryModified is a dictonary to stres weights for most useful positive and negative words
#  head is a list of most useful negative words
#  tail is a list of most useful positive words
# This function eliminates the initial unwanted lines in positive-words.txt and negative-words.txt files
# It also takes into consideration the most useful negative and positive words, 
# and sets their weights to -2 and 2 respectively.
def modifySentimentDictionary(sentimentDictionaryModified, head, tail):
    values = ""
    with open('positive-words.txt', 'r', encoding="ISO-8859-1") as posDictionary:
        posWordList = re.findall(r"[a-z0-9\-]+", values.join(posDictionary.readlines()[35:]))
      
    with open('negative-words.txt', 'r', encoding="ISO-8859-1") as negDictionary:
        negWordList = re.findall(r"[a-z0-9\-]+", values.join(negDictionary.readlines()[35:]))

        
    for i in posWordList:
        sentimentDictionaryModified[i] = 1
    for i in negWordList:
        sentimentDictionaryModified[i] = -1
        
    for word in set(head) & set(sentimentDictionaryModified.keys()):
        sentimentDictionaryModified[word]  = -2
        
    for word in set(tail) & set(sentimentDictionaryModified.keys()):
        sentimentDictionaryModified[word]  = 2



#---------- Main Script --------------------------


sentimentDictionary={} # {} initialises a dictionary [hash function]
sentencesTrain={}
sentencesTest={}
sentencesNokia={}

#initialise datasets and dictionaries
readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia)

pWordPos={} # p(W|Positive)
pWordNeg={} # p(W|Negative)
pWord={}    # p(W) 

#build conditional probabilities using training data
trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord)

#run naive bayes classifier on datasets
testBayes(sentencesTrain,  "Films (Train Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5)
testBayes(sentencesTest,  "Films  (Test Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5)
testBayes(sentencesNokia, "Nokia   (All Data,  Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.7)


#run sentiment dictionary based classifier on datasets
testDictionary(sentencesTrain,  "Films (Train Data, Rule-Based)\t", sentimentDictionary, -4)
testDictionary(sentencesTest,  "Films  (Test Data, Rule-Based)\t",  sentimentDictionary, -4)
testDictionary(sentencesNokia, "Nokia   (All Data, Rule-Based)\t",  sentimentDictionary, -3)

# print most useful words
head, tail = mostUseful(pWordPos, pWordNeg, pWord, 50)

sentimentDictionaryModified = {}
modifySentimentDictionary(sentimentDictionaryModified, head, tail)

testDictionaryModified(sentencesTrain,  "Films (Train Data, Rule-Based)\t", sentimentDictionaryModified, 1)
testDictionaryModified(sentencesTest,  "Films  (Test Data, Rule-Based)\t",  sentimentDictionaryModified, 1)
testDictionaryModified(sentencesNokia, "Nokia   (All Data, Rule-Based)\t",  sentimentDictionaryModified, 1)





