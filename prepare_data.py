import os
import csv
import re
import operator

sentence_list, sentiment_list = [], []

dataDir = "/Users/abaker/smData/IMDBraw/"
saveDir = "/Users/abaker/smData/IMDB/"


def cleanText(text):
    text = text.lower()
    text = text.replace('<br />', '')
    text = text.replace('\'', '')
    text = text.replace('\"', '')
    text = text.replace('/', '')
    text = text.replace('\\', ' ')
    text = re.sub("\.{2,}", ".", text)
    text = text.replace('.', ' <EOS> ')
    text = text.replace(',', '')
    text = text.replace(':', '')
    text = text.replace('!', '')
    text = text.replace('@', '')
    text = text.replace('#', '')
    text = text.replace('$', '')
    text = text.replace('%', '')
    text = text.replace('(', '')
    text = text.replace(')', '')
    text = text.replace(']', '')
    text = text.replace('[', '')
    text = text.replace('}', '')
    text = text.replace('{', '')
    text = text.replace('?', '')
    text = text.replace('-', ' ')
    text = text.replace('*', ' ')
    text = text.replace(';', '')
    text = re.sub('\d+', ' <NUM> ', text)
    text = re.sub('\s+', ' ', text)
    return text


vocabSize = 10000

counts = {}
for dataset in ["train", "test"]:
    for sentiment in ["pos", "neg"]:
        filepath = os.path.join(dataDir, dataset, sentiment)
        filelist = [file for file in os.listdir(filepath) if file.endswith(".txt")]
        for i in range(0, len(filelist)):
            with open(os.path.join(filepath, filelist[i]), 'r') as file:
                text = file.read()
            text = cleanText(text)
            for word in text.split():
                if word not in counts:
                    counts[word] = 1
                else:
                    counts[word] = counts[word] + 1

# Sort counts and cut to top 10000 (including "UNK")
sortedCounts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
sortedCounts = sortedCounts[0:vocabSize-1]
vocab = {}
for word, count in sortedCounts:
    vocab[word] = len(vocab) + 1
vocab['UNK'] = len(vocab) + 1

# Loop through files and replace words that aren't in vocab with "UNK"
counts = {}
for dataset in ["train", "test"]:
    for sentiment in ["pos", "neg"]:
        filepath = os.path.join(dataDir, dataset, sentiment)
        filelist = [file for file in os.listdir(filepath) if file.endswith(".txt")]
        for i in range(0, len(filelist)):
            with open(os.path.join(filepath, filelist[i]), 'r') as file:
                text = file.read()
            text = cleanText(text)
            words = text.split()
            words_unk = []
            for word in words:
                if word not in vocab:
                    word = 'UNK'
                words_unk.append(vocab[word])
            # Save to file
            with open(os.path.join(saveDir, dataset, sentiment + str(i+1) + ".csv"), 'w') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(words_unk)
