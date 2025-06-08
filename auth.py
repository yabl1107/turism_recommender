import os
import json
from functools import wraps
from flask import request, jsonify
import firebase_admin
from firebase_admin import credentials, auth
from dotenv import load_dotenv
from flask import g

# Decorador para proteger rutas
def firebase_auth_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Falta token de autorización"}), 401

        id_token = auth_header.split(" ")[1]

        try:
            decoded_token = auth.verify_id_token(id_token)
            g.user = decoded_token
        except Exception as e:
            return jsonify({"error": "Token inválido o expirado", "detail": str(e)}), 401

        return f(*args, **kwargs)
    return decorated_function
