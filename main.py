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


load_dotenv()
# Obtiene los valores
redis_host = os.getenv("REDIS_HOST")
redis_port = int(os.getenv("REDIS_PORT"))
redis_password = os.getenv("REDIS_PASSWORD")

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
def recomendar():
    body = request.get_json()
    query = body.get("query")
    top_k = body.get("top_k", 2)

    user_id = body.get("user_id", "anonimo")
    chat_id = body.get("chat_id", "default")

    if not query:
        return jsonify({"error": "Falta el campo 'query'"}), 400
    if not user_id:
        return jsonify({"error": "Falta el campo 'user_id'"}), 400
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
        ids = []
        for doc in recs:
            try:
                id_ = int(doc.page_content.strip('"').split()[0]) 
                ids.append(id_)
            except:
                continue
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
            base_messages.append({"role": "user", "content": query})
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
                "recomendacion": recomendacion,
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
    
@app.route('/chat/<key>', methods=['DELETE'])
def borrar_conversacion(key):
    if redis_client.exists(key):
        redis_client.delete(key)
        return jsonify({"message": "Conversación eliminada"}), 200
    else:
        return jsonify({"error": "Clave no encontrada"}), 404


if __name__ == '__main__':
    app.run(host="0.0.0.0")
