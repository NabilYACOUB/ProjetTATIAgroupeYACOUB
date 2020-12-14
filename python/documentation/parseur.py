import xml.etree.ElementTree as ET
import numpy as np
from IPython.display import display
import pandas as pd
import nltk
import ssl
import re
from nltk.corpus import stopwords
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import plot_confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



#tree = ET.parse(r'C:\Users\C\Downloads\ABSA16_Restaurants_Train_SB1_v21.xml')
tree = ET.parse(r'C:\Users\C\Downloads\ABSA16_Restaurants_Train_SB1_v24.xml')
#tree = ET.parse(r'C:\Users\C\Downloads\ABSA16_Restaurants_Train_SB1_v2.xml')
root = tree.getroot()


listetext = []
for elem in root:
    for subelem in elem:
        for susubelem in subelem:
            for last in susubelem:
                #print(last.text)
                listetext.append(last.text)
                
#Cree des bugs  on garde pour le rapport de 10 pages à faire sur la creatoin du parseur et nos erreurs
#print(listetext)
# listetexte = []
# for i in range(0,len(listetext)-1,2):
#         listetexte.append(listetext[i])
        
#print(listetexte)

#Regle le bug et parse les balise text
x_filtered = [i for i in listetext if "\n" not in i]
#print(x_filtered)
print(len(x_filtered))


listereview = []

for review_id in root.findall('Review'):
    value = review_id.get('rid')
    #print(value)
    listereview.append(value)

#print(listereview)
print(len(listereview))






listesentence = []

for sentence_id in root.findall('Review/sentences/sentence'):
    value = sentence_id.get('id')
    #print(value)
    listesentence.append(value)

print(listesentence)
print(len(listesentence))




listeopinion_target = []

for opinion_target in root.findall('Review/sentences/sentence/Opinions/Opinion'):
    value = opinion_target.get('target')
    #print(value)
    listeopinion_target.append(value)

#print(listeopinion_target)
print(len(listeopinion_target))





listeopinion_category = []

for opinion_category in root.findall('Review/sentences/sentence/Opinions/Opinion'):
    value = opinion_category.get('category')
    #print(value)
    listeopinion_category.append(value)

#print(listeopinion_category)
print(len(listeopinion_category))





listeopinion_polarity = []

for opinion_polarity  in root.findall('Review/sentences/sentence/Opinions/Opinion'):
    value = opinion_polarity.get('polarity')
    #print(value)
    listeopinion_polarity.append(value)

#print(listeopinion_polarity)
print(len(listeopinion_polarity))






listeopinion_from = []

for opinion_from  in root.findall('Review/sentences/sentence/Opinions/Opinion'):
    value = opinion_from.get('from')
    #print(value)
    listeopinion_from.append(value)

#print(listeopinion_from)
print(len(listeopinion_from))






listeopinion_to = []

for opinion_to in root.findall('Review/sentences/sentence/Opinions/Opinion'):
    value = opinion_to.get('to')
    #print(value)
    listeopinion_to.append(value)

#print(listeopinion_to)
print(len(listeopinion_to))

liste_r = []
liste_r = listesentence
s = pd.Series(liste_r)

s.str[:-2]



#oN CREE UN FICHIER SSV POUR MANIPLUER AVEC SKR ET NLTK
df = pd.DataFrame(data={"Avis_rid": s.str[:-2],"Phrase_id": listesentence,"Cible": listeopinion_target,"Categorie": listeopinion_category,"Polarite": listeopinion_polarity,"FROM": listeopinion_from,"TO": listeopinion_to,"Avis_texte": x_filtered})

df.to_csv("./Tableau_de_classification_d'avis_de_restaurant.csv", sep=',',index=False)
df = pd.read_csv("./Tableau_de_classification_d'avis_de_restaurant.csv", nrows=1711)

'''

# Premiere analyse
AVIS = pd.read_csv("./Tableau_de_classification_d'avis_de_restaurant.csv")
print(AVIS.head())
X_avis = AVIS['Avis_texte']
y_avis = AVIS['Polarite']

X_avis.shape, y_avis.shape


AVIS_TEST = pd.read_csv("./Tableau_de_classification_d'avis_de_restaurant.csv")
X_test = AVIS_TEST['Avis_texte']
y_test = AVIS_TEST['Polarite']
X_test.shape, y_test.shape

cv = CountVectorizer()
cv.fit(X_avis)
X_avis_encoded = cv.transform(X_avis)
X_avis_encoded.shape

X_test_encoded = cv.transform(X_test)
X_test_encoded.shape

lr = LogisticRegression()
lr.fit(X_avis_encoded,y_avis)

y_pred = lr.predict(X_test_encoded)
accuracy_score(y_test,y_pred)



# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('stopwords')

# estop = stopwords.words("english")



class MyTokenizer:
    def __init__(self):
            self.stemmer = nltk.PorterStemmer().stem
    def __call__(self,corpus):
        return [self.stemmer(token) for token in nltk.word_tokenize(corpus)]


pipeline = Pipeline([("feature_extraction", 
                      CountVectorizer(tokenizer=MyTokenizer(),binary=True,max_df=0.85,ngram_range=(1,1))), 
                     ("classification", LogisticRegression(max_iter=400,solver="newton-cg"))])
parameters = {}

grid = GridSearchCV(pipeline, parameters, scoring='accuracy', cv=3)
grid.fit(X_avis, y_avis);
grid.best_params_
grid.best_estimator_
grid.best_score_

y_pred = grid.predict(X_test)
accuracy_score(y_test,y_pred)

plot_confusion_matrix(grid, X_test, y_test, normalize='true');



'''
















'''


# Seconde analyse
print('Thes text have', len(x_filtered), 'characters')

tokens = []
for i in range(len(x_filtered)):
    tokens = nltk.word_tokenize(x_filtered[i])
    #print(tokens)
    #print('The text have', len(tokens), 'tokens')

porter = nltk.PorterStemmer()

print("Original:", tokens)
stokens = [porter.stem(t) for t in tokens]
print("\nAfter stemming", stokens)

print("\nDifferent tokens", [(w, v) for w, v in zip(tokens, stokens) if w!=v])


WNlemma = nltk.WordNetLemmatizer()
print("Original:", tokens)
ltokens = [WNlemma.lemmatize(t) for t in tokens]
print("\nAfter Lemming", stokens)

print("\nDifferent tokens", [(w, v) for w, v in zip(tokens, stokens) if w!=v])


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

estop = stopwords.words('english')
print(estop)

print("Original:", tokens)
st_tokens = [word for word in tokens if not word in estop]
print("\nAfter stop word removing", st_tokens)
# print only different token
print("\nSuppressed tokens", [w for w in tokens if w not in st_tokens])

vocab = sorted(set(word for sentence in x_filtered for word in sentence.split()))
print("vocab length:", len(vocab), "\n\nvocab:", vocab)

def binary_transform(text, vocab):
    
 # create an N*M size zeros matrix
 # N: number of sentences
 # M: number of words in the vocabulary
 
    output = np.zeros((len(text), len(vocab)))
# for each sentence
    for i, sentence in enumerate(text):
# tokenize the sentence
        words = set(sentence.split())
# put 1 in the corresponding column if the word is present in the␣sentence
        for j, v in enumerate(vocab):
            output[i][j] = v in words
    return output


display(x_filtered)
pd.DataFrame(binary_transform(x_filtered, vocab), columns=vocab)


vec = CountVectorizer(binary=True)
vec.fit(x_filtered)




vocab2 = [w for w in sorted(vec.vocabulary_.keys())]
print("vocab length:", len(vocab2), "\n\nvocab:", vocab2)


display(x_filtered)
pd.DataFrame(vec.transform(x_filtered).toarray(), columns=sorted(vec.vocabulary_.keys()))

vec2 = CountVectorizer(binary=False)
vec2.fit(x_filtered)

display(x_filtered)
pd.DataFrame(vec2.transform(x_filtered).toarray(), columns=sorted(vec.vocabulary_.keys()))

vec = CountVectorizer().fit(x_filtered)
vec.get_feature_names()
vec.vocabulary_

X = vec.transform(x_filtered)
X.toarray()


vec3 = TfidfVectorizer()
vec3.fit(x_filtered)
X = vec3.transform(x_filtered).toarray()
pd.DataFrame(X, columns=sorted(vec.vocabulary_.keys()))


tokens = "hello"
bi_grams = list(ngrams("hello", 2))
print("Original:", tokens)
print("\n2-grams", bi_grams)



tokens = "blue car and blue window".split(' ')
tri_grams= list(ngrams(tokens, 3))
print("Original:", tokens)
print("\n3-grams", tri_grams)


'''





bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),
min_df=2)
bigram_vectorizer.fit(x_filtered)
display(x_filtered)
pd.DataFrame(bigram_vectorizer.transform(x_filtered).toarray(),
columns=sorted(bigram_vectorizer.vocabulary_.keys()))
X = df['Avis_texte']
y = df['Polarite']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

class MyTokenizer:
    def __init__(self):
        self.stemmer = PorterStemmer().stem
    def __call__(self, doc):
        return [self.stemmer(t) for t in word_tokenize(doc)]

pipeline = Pipeline([
('feature_extraction', CountVectorizer(tokenizer=MyTokenizer())),
('classification', LogisticRegression(multi_class='auto'))
])

parameters = {}
parameters['feature_extraction__max_df'] = [0.8, 0.9]
parameters['feature_extraction__ngram_range'] = [(1,1), (2,2)]
parameters['feature_extraction__binary'] = [True, False]
parameters['classification__C'] = [0.001,0.01,1,10,100]

grid = RandomizedSearchCV(pipeline, parameters, scoring='accuracy', cv=3)
grid.fit(X_train, y_train);
 

grid.best_params_
grid.best_estimator_
grid.best_score_
y_pred = grid.predict(X_test)
plot_confusion_matrix(grid, X_test, y_test, normalize='true');


'''




#   On regarde le jeu de donnees et comment il classifie pour nous aider à reconnaitre des avis et a les classifié à leur tour
plot_size = plt.rcParams["figure.figsize"] 
print(plot_size[0]) 
print(plot_size[1])

plot_size[0] = 8
plot_size[1] = 6
plt.rcParams["figure.figsize"] = plot_size 


#Les values pour la food quality , le restaurant general et le service general voir l ambience general sont majoritaires et exploitables
df.Categorie.value_counts().plot(kind='pie', autopct='%1.0f%%')


#Les values sont trop brouillon on ne se base dessus
df.Cible.value_counts().plot(kind='pie', autopct='%1.0f%%')

#   On a une majorite de positive et une minorite de negative et un jet tres faible de neutre. C est exploitable.
df.Polarite.value_counts().plot(kind='pie', autopct='%1.0f%%', colors=["green", "red", "yellow"])


#n a des donnes difficilement exploitables
df.FROM.value_counts().plot(kind='pie', autopct='%1.0f%%')
df.TO.value_counts().plot(kind='pie', autopct='%1.0f%%')

data_sentiment = df.groupby(['Categorie', 'Polarite']).Polarite.count().unstack()
data_sentiment.plot(kind='bar')

#   Remplacer confidence par la valeur confiance d'avoir la bonne polarite pour une phrase donnee, (pour la fin du projet)
#sns.barplot(x='Polarite', y='CONFIDENCE' , data=df)



''' 




vectorizer = TfidfVectorizer (max_features=1711, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(x_filtered).toarray()
#print(df.Polarite)
X_train, X_test, y_train, y_test = train_test_split(processed_features, df.Polarite, test_size=0.2, random_state=0)

text_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
text_classifier.fit(X_train, y_train)

predictions = text_classifier.predict(X_test)

print(confusion_matrix(y_test,predictions))
# print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))
