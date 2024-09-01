from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import re

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
    "similitud_areas_interes": 0.35,
    "coincidencia_habilidades": 0.30,
    "diferencia_experiencia": 0.5,
    "requisito_educacion": 0.10,
    "coincidencia_jornada": 0.10,
    "coincidencia_modalidad_ubicacion": 0.05,
    "diferencia_salario": 0.05
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


import re
from sklearn.metrics.pairwise import cosine_similarity

def obtener_numero_de_experiencia(experiencia):
    # Extraer el primer número encontrado en la cadena
    match = re.search(r'\d+', experiencia)
    if match:
        return int(match.group(0))
    return 0

def obtener_numero_de_salario(salario):
    # Extraer todos los números encontrados en la cadena y devolver una lista de enteros
    return [int(num) for num in re.findall(r'\d+', salario)]

def calcular_puntuacion_final(candidato, oferta):
    # Calcular similitud de áreas de interés
    vec_candidato_interests = obtener_representacion(candidato.interests)
    vec_oferta_descripcion = obtener_representacion(oferta.descripcion)
    similitud_areas_interes = cosine_similarity([vec_candidato_interests], [vec_oferta_descripcion])[0][0]

    # Calcular coincidencia de habilidades
    oferta_skills = set(re.findall(r'\w+', oferta.descripcion))
    common_skills = set(candidato.skills).intersection(oferta_skills)
    coincidencia_habilidades = len(common_skills) / max(len(candidato.skills), 1)

    # Calcular diferencia de experiencia
    candidato_exp = obtener_numero_de_experiencia(candidato.experience)
    oferta_exp = obtener_numero_de_experiencia(oferta.experiencia)
    if max(candidato_exp, oferta_exp) > 0:
        diferencia_experiencia = 1 - abs(candidato_exp - oferta_exp) / max(candidato_exp, oferta_exp)
    else:
        diferencia_experiencia = 0

    # Coincidencia de educación
    requisito_educacion = 1 if candidato.education.lower() == oferta.educacion.lower() else 0

    # Coincidencia de jornada
    coincidencia_jornada = 1 if candidato.jobSchedule.lower() == oferta.jornada.lower() else 0

    # Coincidencia de modalidad y ubicación
    coincidencia_modalidad_ubicacion = 1 if (candidato.jobMode.lower() == oferta.modalidad.lower() and candidato.jobLocation.lower() == oferta.lugar_trabajo.lower()) else 0

    # Diferencia de salario
    candidato_salarios = obtener_numero_de_salario(candidato.desiredSalary)
    oferta_salarios = obtener_numero_de_salario(oferta.salario_ofrecido)
    
    if len(candidato_salarios) >= 2 and len(oferta_salarios) >= 2:
        promedio_salario_candidato = (candidato_salarios[0] + candidato_salarios[1]) / 2
        promedio_salario_oferta = (oferta_salarios[0] + oferta_salarios[1]) / 2
        if max(promedio_salario_candidato, promedio_salario_oferta) > 0:
            diferencia_salario = 1 - abs(promedio_salario_candidato - promedio_salario_oferta) / max(promedio_salario_candidato, promedio_salario_oferta)
        else:
            diferencia_salario = 0
    else:
        diferencia_salario = 0  # Asignar 0 si no se pueden calcular los promedios

    # Calcular puntuación final
    puntuacion_final = (
        pesos["similitud_areas_interes"] * similitud_areas_interes +
        pesos["coincidencia_habilidades"] * coincidencia_habilidades +
        pesos["diferencia_experiencia"] * diferencia_experiencia +
        pesos["requisito_educacion"] * requisito_educacion +
        pesos["coincidencia_jornada"] * coincidencia_jornada +
        pesos["coincidencia_modalidad_ubicacion"] * coincidencia_modalidad_ubicacion +
        pesos["diferencia_salario"] * diferencia_salario
    )

    return puntuacion_final

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
            # Calcular la puntuación final utilizando la lógica mejorada
            puntuacion_final = calcular_puntuacion_final(candidato, oferta)
            # usando los pesos y otras características

            resultados.append({
                "oferta": oferta.dict(),
                "similitud": float(similitud),
                "puntuacion_final": float(puntuacion_final)
            })

        # Ordenar resultados por similitud
        resultados.sort(key=lambda x: x["similitud"], reverse=True)

        return resultados

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)