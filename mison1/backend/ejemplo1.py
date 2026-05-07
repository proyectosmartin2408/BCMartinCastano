import pandas as pd

# crear datos (simulacion estudiantes)

datos={
    "nombre":["Ana","Luis","Carlos","Sofía","pedro"],
    "edad":[30,70,18,21,22],
    "nota":[3,5,2.2,4.8,3.0]
}

df = pd.DataFrame(datos)
print(df)
print("Promedio: ",df["nota"].mean())

print("Promedio Edad: ",df["edad"].mean())