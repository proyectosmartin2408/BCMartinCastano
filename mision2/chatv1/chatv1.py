"""CountVectorizer convierte texto en un vector"""
from sklearn.feature_extraction.text import CountVectorizer
"""MultinomialNB modelo de inteligencia artificial que aprende relaciones entre texto y respuestas"""
from sklearn.naive_bayes import MultinomialNB

def  buildandtrainmodel (train_pairs): 
  # train_pairs:  Lista de pares (Pregunta, Rerspuesta)
  # ejemplo [("Hola","!Hola"), ("Adiós","Hasta luego")]

  # Separamos las preguntas y respuestas en dos listas

  questions = [q for q, _ in train_pairs] # lista de preguntas
  answers = [a for _ , a in train_pairs] # lista de respuestas

  # creamos el vectorizador, que traducirá el texto a números

  vectorizar = CountVectorizer()