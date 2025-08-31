import os
import openai
import psycopg2
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd


load_dotenv()

# Cliente OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Load environment variables from .env
load_dotenv()

# Fetch variables
DATABASE_SUPA_URL = os.getenv("DATABASE_SUPA_URL")


def insert_embedding(titulo, descripcion):
    # Connect to the database
    try:
        connection = psycopg2.connect(DATABASE_SUPA_URL)
        print("Connection successful!")
        
        # Create a cursor to execute SQL queries
        cur = connection.cursor()

        # Genera embedding
        emb = client.embeddings.create(
            input=descripcion,
            model="text-embedding-3-small"
        ).data[0].embedding
        
        # Inserta en tabla
        cur.execute(
        """
        INSERT INTO actividades (titulo, descripcion, embedding)
        VALUES (%s, %s, %s)
        """,
        (titulo, descripcion, emb)
        )
        connection.commit()

        # Close the cursor and connection
        cur.close()
        connection.close()
        print("Connection closed.")

    except Exception as e:
        print(f"Failed to connect: {e}")
        


def get_k_similares(description, k=3):
    """
    embedding: lista o numpy array con el embedding de consulta
    k: cantidad de resultados a retornar
    """
    try:
        # Genera embedding
        embedding = client.embeddings.create(
            input=description,
            model="text-embedding-3-large"
        ).data[0].embedding

        connection = psycopg2.connect(DATABASE_SUPA_URL)
        cur = connection.cursor()

        # Convierte el embedding a string para la query
        embedding_str = str(list(embedding))  # Formato correcto para pgvector

        # Query usando pgvector (distancia coseno)
        query = f"""
        SELECT id,titulo, descripcion
        FROM actividades
        ORDER BY embedding <=> '{embedding_str}'::vector
        LIMIT {k};
        """

        cur.execute(query)
        resultados = cur.fetchall()

        k_similares = []
        for fila in resultados:
            k_similares.append({
                "id": fila[0],
                "titulo": fila[1],
                "descripcion": fila[2]
                #"embedding": fila[2]
            })

        cur.close()
        connection.close()
        return k_similares

    except Exception as e:
        print(f"Error: {e}")
        return []


def main_load():
    df = pd.read_csv("clean_data.csv")
    try:
        connection = psycopg2.connect(DATABASE_SUPA_URL)
        cur = connection.cursor()
        
        for _, row in df.iterrows():
            emb = client.embeddings.create(
                input=row["description"],
                model="text-embedding-3-large"
            ).data[0].embedding
            emb = np.array(emb)
            emb = emb / np.linalg.norm(emb)
            # Convertir a lista para psycopg2
            emb_list = emb.tolist()
            cur.execute(
                """
                INSERT INTO actividades (titulo, descripcion, embedding)
                VALUES (%s, %s, %s)
                """,
                (row["title"], row["description"], emb_list)
            )
        
        connection.commit()
        cur.close()
        connection.close()
    except Exception as e:
        print(e)
