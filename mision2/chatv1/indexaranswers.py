from sklearn.feature_extraction.text import CountVectorizer

train_pairs = [("Hola cómo estás","¡Hola!"), ("Cómo estás tú","¡Hasta luego!"),("Hola tú ok","¡Hasta luego!")]
questions = [q for q,_ in train_pairs] # lista de preguntas()
answers = [a for _,a in train_pairs] # lista de respuestas

unique_answers = sorted(set(answers))

print(f"Unique_answers{unique_answers}")

answer_to_label = {a: i for i, a in enumerate(unique_answers)}

print(f"answer_to_label {answer_to_label}")

y = [answer_to_label[a] for a in answers]

print(f"label {y}")