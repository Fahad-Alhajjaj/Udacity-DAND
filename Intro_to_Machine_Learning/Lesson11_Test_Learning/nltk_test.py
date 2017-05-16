from nltk.corpus import stopwords

sw = stopwords.words('english')

print type(sw)
print len(sw)


from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")