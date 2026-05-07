import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

train_pairs = [("Hola cómo estás","¡Hola!"), ("Cómo estás tú","¡Hasta luego!"),("Hola tú ok","¡Hasta luego!")]
questions = [q for q,_ in train_pairs] # lista de preguntas()
answers = [a for _,a in train_pairs] # lista de respuestas

#print(f"Lista de preguntas = {questions}")
#print(f"Lista de respuestas = {answers}")

coun_vect = CountVectorizer()

#vectorizer = CountVectorizer()

#x = vectorizer.fit_transform(questions)

count_matrix = coun_vect.fit_transform(questions)
count_array = count_matrix.toarray()
df = pd.DataFrame(data=count_array,columns = coun_vect.get_feature_names_out())
print(count_matrix)
print("\n")
print(df)