from flask import Flask, request, jsonify
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import pandas as pd
from dotenv import load_dotenv
import openai
from openai import OpenAI
import os
import redis
import json
import firebase_admin
from firebase_admin import credentials, auth
from auth import firebase_auth_required
from flask import g
import re
from datetime import datetime

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
# Obtiene los valores
redis_host = os.getenv("REDIS_HOST")
redis_port = int(os.getenv("REDIS_PORT"))
redis_password = os.getenv("REDIS_PASSWORD")
cred_json = os.getenv("GOOGLE_CREDENTIALS")

#FIREBASE
cred_dict = json.loads(cred_json)
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred)

# Carga la base de datos persistida
db_turismo = Chroma(persist_directory="./bd2", embedding_function=OpenAIEmbeddings())
data = pd.read_csv("clean_data.csv")

# Conexión a Redis Cloud
redis_client = redis.Redis(
    host= redis_host,
    port=redis_port,
    password=redis_password,
    decode_responses=True
)

app = Flask(__name__)
@app.route('/recomendar', methods=['POST'])
@firebase_auth_required
def recomendar():
    body = request.get_json()
    query = body.get("query")
    top_k = body.get("top_k", 1)

    #user_id = body.get("user_id", "anonimo")
    user_id = g.user['uid']

    chat_id = body.get("chat_id", "default")
    #Si chatId es null, se crea un nuevo chat y 
    # se retorna id, para setearlo en el arredlo de metadata del viewModel

    if not query:
        return jsonify({"error": "Falta el campo 'query'"}), 400
    if not chat_id:
        return jsonify({"error": "Falta el campo 'chat_id'"}), 400

    #Clave Redis: user:<user_id>:chat:<chat_id>
    redis_key = f"user:{user_id}:chat:{chat_id}"
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    key_exists = redis_client.exists(redis_key)
    
    if not key_exists:
        msg = {
                "role": "system",
                "content": 
                    "Eres un asistente de actividades turísticas especializado únicamente en la ciudad de Lima. Solo puedes hablar sobre turismo, lugares, recomendaciones de actividades, cultura y geografía. Si el usuario pregunta algo fuera de este tema, responde diciendo que solo puedes ayudar con temas turísticos."
                    "Si el usuario pregunta por otra ciudad, debes responder amablemente que solo manejas información sobre Lima. "
                    "Si el usuario no menciona una ciudad o hace una pregunta ambigua, asume que se refiere a Lima."
            }
        redis_client.rpush(redis_key, json.dumps(msg))
        # Crear el metadato del chat si es la primera vez
        meta_key = f"{redis_key}:meta"
        redis_client.hset(meta_key, mapping={
            "created_at": datetime.utcnow().isoformat() + "Z",
            "title": query,
            "chat_id": chat_id
        })
        print("Se ha guardado la clave meta ....")

    ## Cargar historial de redis
    context_raw_messages = redis_client.lrange(redis_key, 0, -1)
    base_messages = [json.loads(m) for m in context_raw_messages]
    

    #######################################
    #######   DETERMINAR INTENCION ########

    prompt = f"""
    Analiza el siguiente mensaje del usuario y responde con una sola palabra que indique la intención:
    - "recomendar" si el usuario quiere una recomendación turística,
    - "charla" si solo está conversando,
    - "otro" si no está claro.

    Mensaje: "{query}"
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano", 
            messages=[
                {"role": "system", "content": "Eres un asistente que clasifica intenciones del usuario."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        intent = response.choices[0].message.content.strip().lower()
    except Exception as e:
        return jsonify({"error": "Error al determinar la intención", "detalle": str(e)}), 500



    if intent == "recomendar":
        #######   BUSQUEDA VECTORIAL ########
        recs = db_turismo.similarity_search(query, k=top_k)
        ids = [doc.metadata["id"] for doc in recs]
        
        # Filtra DataFrame original por los IDs encontrados
        recomendados = data[data["id"].isin(ids)][["id", "title_2","description_2"]]
        lugares_lista = recomendados.to_dict(orient="records")
        
        #######   RESPONDER CON MML USANDO CONTEXTO DE LUGARES RECOMENDADOS  ######
        #prompt para el LLM
        lista_texto = "\n".join([f"- {lugar['title_2']}: {lugar['description_2']}" for lugar in lugares_lista])
        
        prompt = f"""
        El usuario está buscando actividades turísticas basados en: "{query}"

        Aquí hay una lista de actividades similares:

        {lista_texto}

        Redáctale una recomendación amable y atractiva mencionando actividades turísticas que se adapten bien a su búsqueda.
        """
        
        try:
            ## Agregar mensaje del usuario a Redis
            base_messages.append({"role": "user", "content": prompt})
            redis_client.rpush(redis_key, json.dumps({"role": "user", "content": query}))
            
            respuesta = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages = base_messages,
                temperature=0.7
            )
            ## Agregar mensaje de system a Redis
            recomendacion = respuesta.choices[0].message.content.strip()
            redis_client.rpush(redis_key, json.dumps({"role": "assistant", "content": recomendacion}))
            
            return jsonify({
                "intencion": intent,
                "respuesta": recomendacion,
                "lugares": lugares_lista
            })

        except Exception as e:
            return jsonify({
                "error": "Error al generar recomendación con OpenAI",
                "detalle": str(e)
            }), 500

    else: ## Caso donde no busca recomendación.

        base_messages.append({"role": "user", "content": query})
        redis_client.rpush(redis_key, json.dumps({"role": "user", "content": query}))

        respuesta = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages = base_messages,
            temperature=0.7
        )
        
        ## Agregar mensaje de system a Redis
        rpta_txt = respuesta.choices[0].message.content.strip()
        redis_client.rpush(redis_key, json.dumps({"role": "assistant", "content": rpta_txt}))
            
        return jsonify({
            "intencion": intent,
            "respuesta": rpta_txt
        })
    
@app.route('/chat/<chatId>', methods=['DELETE'])
@firebase_auth_required
def borrar_conversacion(chatId):
    user_id = g.user['uid']
    key_secure = f"user:{user_id}:chat:{chatId}"
    if redis_client.exists(key_secure):
        redis_client.delete(key_secure)
        return jsonify({"message": "Conversación eliminada"}), 200
    else:
        return jsonify({"error": "Clave no encontrada"}), 404


@app.route('/chat/<key>', methods=['GET']) #Obtiene chat con id de chat
@firebase_auth_required
def obtener_conversacion(key):
    user_id = g.user['uid']
    key_secure = f"user:{user_id}:chat:{key}"
    if redis_client.exists(key_secure):
        mensajes_raw = redis_client.lrange(key_secure, 0, -1)
        mensajes = [json.loads(m) for m in mensajes_raw]
        return jsonify({"mensajes": mensajes}), 200
    else:
        return jsonify({"error": "Clave no encontrada"}), 404



@app.route('/conversaciones', methods=['GET'])
@firebase_auth_required
def obtener_conversaciones():
    user_id = g.user['uid']
    conversaciones = []

    # Buscar todas las claves de metadatos del usuario
    patron = f"user:{user_id}:chat:*:meta"
    claves_meta = redis_client.keys(patron)

    for meta_key in claves_meta:
        metadatos = redis_client.hgetall(meta_key)

        if all(k in metadatos for k in ("created_at", "title", "chat_id")):
            conversaciones.append({
                "created_at": metadatos["created_at"],
                "title": metadatos["title"],
                "chat_id": metadatos["chat_id"]
            })

    return jsonify({"conversaciones": conversaciones}), 200


@app.route('/mejor_sugerencia', methods=['POST'])
def mejor_sugerencia():
    data = request.get_json()

    top_query_results = [str(i) for i in data.get('top_query_results', [])]
    liked_ids = [str(i) for i in data.get('liked_ids', [])]

    if len(top_query_results)<0 or len(liked_ids)<0:
        return jsonify({'error': 'Se requieren top_query_results y liked_ids'}), 400

    # Obtener vectores de liked_ids
    #liked_docs = db_turismo._collection.get(liked_ids)
    #liked_vectors = liked_docs.get('embeddings', [])

    liked_vectors = db_turismo.get(
        ids=liked_ids,
        include=["embeddings"]
    )["embeddings"]

    # Calcular vector promedio del perfil del usuario
    profile_vector = np.mean(liked_vectors, axis=0)

    # Obtener vectores de top_query_results
    #top_docs = db_turismo._collection.get(ids=top_query_results)
    #top_vectors = top_docs.get('embeddings', [])
    top_ids = top_query_results
    
    top_query_vectors = db_turismo.get(
        ids=top_query_results,
        include=["embeddings"]
    )["embeddings"]

    # Calcular similitudes
    
    similarities = cosine_similarity([profile_vector], top_query_vectors)[0]

    # Índice del más parecido
    best_match_index = np.argmax(similarities)
    best_match_id = top_query_results[best_match_index]
    
    
    return jsonify({
        'mejor_id': best_match_id,
        'similitud': similarities[best_match_index]
    })


if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)
