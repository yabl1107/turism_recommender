from flask import Flask, request, jsonify
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import pandas as pd
from dotenv import load_dotenv

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

    # Convierte a JSON y devuelve
    return jsonify(recomendados.to_dict(orient="records"))

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
