from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Importar el resto de tus bibliotecas y funciones Python

app = FastAPI()

# Inicialización del modelo BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Función para obtener representaciones de BERT

'''
  toma un texto como entrada(lo tokeniza)lo pasa a través del modelo BERT,
   y devuelve una representación vectorial del texto.
'''
def obtener_representacion(texto):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

# Pesos de las características
pesos = {
    "similitud_areas_interes": 0.25,
    "coincidencia_habilidades": 0.20,
    "diferencia_experiencia": 0.15,
    "requisito_educacion": 0.10,
    "coincidencia_jornada": 0.10,
    "coincidencia_modalidad_ubicacion": 0.10,
    "diferencia_salario": 0.10
}

# Modelos Pydantic para validación de datos
class Candidato(BaseModel):
    name: str
    surname: str
    dni: str
    email: str
    jobLocation: str
    jobMode: str
    jobSchedule: str
    education: str
    experience: str
    skills: List[str]
    interests: str
    desiredSalary: str

class Oferta(BaseModel):
    empresa: str
    puesto: str
    descripcion: str
    experiencia: str
    educacion: str
    jornada: str
    modalidad: str
    lugar_trabajo: str
    salario_ofrecido: str

class MatchRequest(BaseModel):
    candidato: Candidato
    ofertas: List[Oferta]


@app.get("/")
def index():
    return "JobMatch"


@app.post("/match")
async def match(request: MatchRequest):
    try:
        candidato = request.candidato
        ofertas = request.ofertas

        # Generar descripción del candidato
        desc_candidato = (f"{candidato.name} {candidato.surname} está interesado en {candidato.interests}. "
                  f"Tiene habilidades en {', '.join(candidato.skills)} y {candidato.experience} de experiencia. "
                  f"Nivel de educación: {candidato.education}. Jornada preferida: {candidato.jobSchedule}. "
                  f"Modalidad y ubicación: {candidato.jobMode}, {candidato.jobLocation}. "
                  f"Salario deseado: {candidato.desiredSalary}.")

        vec_candidato = obtener_representacion(desc_candidato)

        resultados = []
        for oferta in ofertas:
            # Generar descripción de la oferta
            desc_oferta = (f"{oferta.empresa} busca un {oferta.puesto}. {oferta.descripcion} "
                           f"Requisitos: {oferta.experiencia} de experiencia, {oferta.educacion} como nivel de educación. "
                           f"Jornada: {oferta.jornada}. Modalidad y ubicación: {oferta.modalidad}, {oferta.lugar_trabajo}. "
                           f"Salario ofrecido: {oferta.salario_ofrecido}.")

            vec_oferta = obtener_representacion(desc_oferta)

            # Calcular similitud
            similitud = cosine_similarity([vec_candidato], [vec_oferta])[0][0]

            # Aquí implementamos más lógica para calcular la puntuación final
            # usando los pesos y otras características

            resultados.append({
                "oferta": oferta.dict(),
                "similitud": float(similitud)
            })

        # Ordenar resultados por similitud
        resultados.sort(key=lambda x: x["similitud"], reverse=True)

        return resultados

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)