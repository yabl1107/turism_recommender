from flask import Flask, request, jsonify
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import pandas as pd
from dotenv import load_dotenv
import openai
from openai import OpenAI
import os

load_dotenv()

# Carga la base de datos persistida
db_turismo = Chroma(persist_directory="./bd2", embedding_function=OpenAIEmbeddings())
data = pd.read_csv("clean_data.csv")


app = Flask(__name__)
@app.route('/recomendar', methods=['POST'])
def recomendar():
    body = request.get_json()
    query = body.get("query")
    top_k = body.get("top_k", 2)

    if not query:
        return jsonify({"error": "Falta el campo 'query'"}), 400

    #######################################
    #######   DETERMINAR INTENCION ########
    #######################################

    prompt = f"""
    Analiza el siguiente mensaje del usuario y responde con una sola palabra que indique la intención:
    - "recomendar" si el usuario quiere una recomendación turística,
    - "charla" si solo está conversando,
    - "otro" si no está claro.

    Mensaje: "{query}"
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
        #####################################
        #######   BUSQUEDA VECTORIAL ########
        #####################################
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
        
        

        ###################################################################
        #######   RESPONDER CON MML USANDO CONTEXTO DE LUGARES RECOMENDADOS
        ###################################################################

        # Genera un prompt para el LLM
        lista_texto = "\n".join([f"- {lugar['title_2']}: {lugar['description_2']}" for lugar in lugares_lista])
        prompt = f"""
        El usuario está buscando actividades turísticas basados en: "{query}"

        Aquí hay una lista de actividades similares:

        {lista_texto}

        Redáctale una recomendación amable y atractiva mencionando actividades turísticas que se adapten bien a su búsqueda.
        """

        

        try:
            respuesta = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un asistente de actividades turísticas especializado únicamente en la ciudad de Lima. Solo puedes hablar sobre turismo, lugares, recomendaciones de actividades, cultura y geografía. Si el usuario pregunta algo fuera de este tema, responde diciendo que solo puedes ayudar con temas turísticos."
                        "Si el usuario pregunta por otra ciudad, debes responder amablemente que solo manejas información sobre Lima. "
                        "Si el usuario no menciona una ciudad o hace una pregunta ambigua, asume que se refiere a Lima."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
            )

            recomendacion = respuesta.choices[0].message.content.strip()

        except Exception as e:
            return jsonify({
                "error": "Error al generar recomendación con OpenAI",
                "detalle": str(e)
            }), 500

        
        #####################################################
        #######   BUSQUEDA VECTORIAL 
        #####################################################
        
        return jsonify({
            "recomendacion": recomendacion,
            "lugares": lugares_lista
        })

    else:
        respuesta = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un asistente turístico especializado únicamente en la ciudad de Lima. Solo puedes hablar sobre turismo, lugares, recomendaciones de actividades, cultura y geografía. Si el usuario pregunta algo fuera de este tema, responde diciendo que solo puedes ayudar con temas turísticos."
                    "Si el usuario pregunta por otra ciudad, debes responder amablemente que solo manejas información sobre Lima. "
                    "Si el usuario no menciona una ciudad o hace una pregunta ambigua, asume que se refiere a Lima."
                )
            },
            {"role": "user", "content": query}
        ],
        temperature=0.7
        )
        return jsonify({
            "intencion": intent,
            "respuesta": respuesta.choices[0].message.content.strip()
        })
    

if __name__ == '__main__':
    app.run(debug=True)
