import xml.etree.ElementTree as ET
import pandas as pd


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
df = pd.DataFrame(data={"Avis_rid": s.str[:-2],"Phrase_id": listesentence,R"Cible": listeopinion_target,"Categorie": listeopinion_category,"Polarite": listeopinion_polarity,"FROM": listeopinion_from,"TO": listeopinion_to})

df.to_csv("./Tableau_de_classification_d'avis_de_restaurant.csv", sep=',',index=False)




