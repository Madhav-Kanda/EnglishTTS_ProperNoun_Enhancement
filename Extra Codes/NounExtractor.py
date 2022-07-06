import os
import codecs
import nltk
# nltk.download('all')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

path =r"/home/kanda/IISc_Internship/Classificationtext"
os.chdir(path)

def read_files(file_path):
    with codecs.open(file_path) as f:
        return f.read()

#files name
file_noun = open("/home/kanda/IISc_Internship/ClassifyNoun/noun.txt", "a")
stop_words = set(stopwords.words('english'))



for file in os.listdir(path):
   if file.endswith('.txt'):
    file_path ='/home/kanda/IISc_Internship/Classificationtext/'+str(file)
    file_content=read_files(file_path)
    tokenized = sent_tokenize(file_content)
    for i in tokenized:
        wordsList = nltk.word_tokenize(i)
        wordsList = [w for w in wordsList if not w in stop_words]
        tagged = nltk.pos_tag(wordsList)
        print(tagged)
        for w,tag in tagged:
          if tag=="NNP" or tag=="NNPS":
            file_noun.write(w)
            file_noun.write("\n")
          
file_noun.close()
