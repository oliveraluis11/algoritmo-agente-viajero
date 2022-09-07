from typing import List

from itertools import combinations
import uvicorn
from fastapi import FastAPI, APIRouter
from fastapi.openapi.utils import get_openapi
from haversine import haversine_vector, Unit
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from models import Punto

app = FastAPI()

punto = APIRouter()
import numpy
import array
import random
import json

with open("data.json", "r") as db:
    data = json.load(db)


@punto.get("/", response_model=List[Punto], response_model_exclude_none=True)
def find_all_points():
    puntos_list: List[Punto]
    puntos_list = list()
    for item in data:
        puntos_list.append(Punto(**item))
    return puntos_list


@punto.get("/optimal")
def find_algorithm():
    puntos_list: List[Punto]
    puntos_list = list()
    for item in data:
        puntos_list.append(Punto(**item))
    distance_map = calcular_distancias(puntos_list)
    IND_SIZE = len(puntos_list)

    rutaPuntos = []

    ruta, distancia = viajero(distance_map, IND_SIZE)
    ruta = str(ruta).split('[')[2].split(']')[0].split(',')
    ruta = list(map(int, ruta))  # convirtiendo indices a enteros
    for i in range(len(ruta)):
        rutaPuntos.append(puntos_list[ruta[i]])

    x = {"ruta": rutaPuntos, "distancia": distancia[0]}

    temp = [x.id for x in x["ruta"]]
    combinaciones = combinations(temp, 11)
    for c in list(combinaciones):
        print(c)


    return {
        "ruta": rutaPuntos,
        "distancia": distancia[0]
    }


@punto.get("/{id}", response_model=Punto, response_model_exclude_none=True)
def find_pont_by_id(id: int):
    item_search = None
    for item in data:
        if item["id"] == id:
            item_search = Punto(**item)
            break
    return item_search


def calcular_distancias(puntos: List[Punto]):
    l = len(puntos)
    arr_puntos = [(0, 0) for _ in range(l)]
    for i in range(l):
        arr_puntos[i] = puntos[i].datos()

    return haversine_vector(arr_puntos, arr_puntos, Unit.METERS, comb=True)


def viajero(distance_map, IND_SIZE):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", array.array, typecode='I',
                   fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register("indices", random.sample, range(IND_SIZE), IND_SIZE)

    toolbox.register("individual", tools.initIterate,
                     creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evalTSP(individual):
        distance = distance_map[individual[-1]][individual[0]]
        for gene1, gene2 in zip(individual[0:-1], individual[1:]):
            distance += distance_map[gene1][gene2]
        return distance,

    toolbox.register("mate", tools.cxPartialyMatched)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=30)
    toolbox.register("evaluate", evalTSP)

    def main():
        random.seed(169)

        pop = toolbox.population(n=1000)

        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", numpy.min)

        algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 40, stats=stats,
                            halloffame=hof)

        return hof

    hof = main()

    return hof, evalTSP(hof[0])


app.include_router(punto)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Algoritmos Evolutivos",
        version="2.5.0",
        description="Sistema de optimizacion de rutas para el Serenazgo de Nuevo Chimbote",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
