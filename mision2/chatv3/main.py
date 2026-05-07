from chatbot.data import training_data
from chatbot.model import build_and_train_model, load_model, predict_answers


# ===================================
# FUNCION PRINCIPAL
# ====================================

def main():

    model, vectorizer, unique_answers = load_model()

    # verificamos si no existe alguno de los elementos del modelo
    if model is None:        
        # Entrenamos el modelo con los datos
        model, vectorizer, unique_answers = build_and_train_model(training_data)

    # Mostrar un mensaje inicial al usuario
    print("\n 🤖 Chatbot supervisado listo. Escribe 'salir' para terminar. \n")

    while True:
        # Pedimos una frase al usuario
        user = input(" ✍️   Cliente: ").strip()  # strip elimina espacios al inicio y final
        if user.lower() == "":
            print(" 🤖 Agente BOT:" "¡Hasta pronto! \n\n")
            break

        # modelo predice la respuesta
        response = predict_answers(model, vectorizer, unique_answers, user)

        # Mostrar la respuesta en pantalla
        print(" \n 🤖 Agente BOT: ", response, "\n")


# ==================================
# LAMADO A LA FUNCION PRINCIPAL
# ==================================
if __name__ == "__main__":
    main()
