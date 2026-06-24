import mombeu

mombeu.init(model="mombeu-v1")

print(mombeu.current_model())

from mombeu import _model

continuation = _model.generate_continuation(
    "Los sonidos, sincronizados con el crujido de las hojas, crean una sinfonía de la naturaleza que es a la vez majestuosa y calmante, que llena el corazón de un observador con asombro y serenidad."
)

print("CONTINUACION:", repr(continuation))
