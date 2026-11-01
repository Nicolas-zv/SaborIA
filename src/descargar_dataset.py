
from pathlib import Path
from io import StringIO

import pandas as pd
from datasets import load_dataset

# 1. Definir las rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 2. Descargar el dataset
print("Descargando dataset de Hugging Face...")

dataset = load_dataset("mrcsgh/horeca-spanish-reviews")

# 3. Obtener el contenido original
df_original = dataset["train"].to_pandas()

print("Registros originales:", len(df_original))

# 4. Separar los datos por tabulaciones
contenido = "\n".join(df_original["text"].astype(str))

df = pd.read_csv(
    StringIO(contenido),
    sep="\t",
    dtype=str
)

print("\nColumnas detectadas:")
print(df.columns.tolist())

# 5. Convertir estrellas a valores numéricos
df["estrellas"] = pd.to_numeric(
    df["estrellas"],
    errors="coerce"
)

# 6. Seleccionar únicamente restaurantes
df = df[
    df["negocio"].str.strip().str.lower() == "restaurante"
].copy()

# 7. Eliminar registros incompletos
df = df.dropna(subset=["texto", "estrellas"])

# 8. Clasificar el sentimiento
def clasificar_sentimiento(estrellas):
    if estrellas <= 2:
        return "negativo"
    elif estrellas == 3:
        return "neutral"
    else:
        return "positivo"

df["sentimiento"] = df["estrellas"].apply(
    clasificar_sentimiento
)

# 9. Guardar el dataset procesado
ruta_salida = DATA_DIR / "resenas_restaurantes.csv"

df.to_csv(
    ruta_salida,
    index=False,
    encoding="utf-8-sig"
)

# 10. Mostrar resultados
print("\nDataset procesado correctamente.")
print("Total de reseñas de restaurantes:", len(df))

print("\nDistribución de sentimientos:")
print(df["sentimiento"].value_counts())

print("\nPrimeras 5 reseñas:")
print(
    df[["texto", "estrellas", "sentimiento"]].head()
)

print("\nArchivo guardado en:")
print(ruta_salida)
