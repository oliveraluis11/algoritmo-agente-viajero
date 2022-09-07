from pydantic import BaseModel


class Punto(BaseModel):
    id: int
    latitud: float
    longitud: float

    def datos(self):
        return self.latitud, self.longitud
