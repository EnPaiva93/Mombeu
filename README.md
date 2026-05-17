# Mombeu

**Capa de inserción narrativa cultural guaraní para LLMs**

> *Como cuando el abuelo responde algo cotidiano y al final dice "eso me recuerda..." — Mombeu agrega un fragmento de cosmovisión guaraní a la salida de cualquier LLM, sin que el usuario lo pida.*

---

## ¿Qué es Mombeu?

Mombeu es una capa complementaria que se conecta a cualquier LLM existente. No es un chatbot ni un traductor — es un modelo pequeño especializado que intercepta respuestas generales y agrega al final un cierre narrativo basado en la cosmovisión guaraní, cuando la temática lo permite.

El usuario sigue usando el LLM de siempre. Mombeu trabaja en el fondo.

```
Usuario pregunta
        ↓
LLM grande responde  (GPT, Claude, Llama, etc.)
        ↓
Mombeu — encoder de pertinencia temática
    score > umbral?
    │
    ├── No → respuesta sin modificar
    │
    └── Sí → genera cierre narrativo
              basado en Ñande Ypykuéra + grafo
        ↓
Respuesta enriquecida al usuario
```

### Ejemplo

**Sin Mombeu:**
> Los truenos ocurren cuando el aire caliente sube rápidamente y genera descargas eléctricas en las nubes.

**Con Mombeu:**
> Los truenos ocurren cuando el aire caliente sube rápidamente y genera descargas eléctricas en las nubes. Eso me recuerda a **Tupã**, cuya voz retumba en el cielo para el pueblo guaraní, el trueno no era un fenómeno sino la presencia del creador.

---

## Motivación

Paraguay es el único país de América donde una lengua indígena —el guaraní— es co-oficial y hablada por la mayoría de la población. Sin embargo, los LLMs actuales ignoran casi completamente su cultura y cosmovisión.

Mombeu propone que cada respuesta sea una oportunidad de transmitir memoria colectiva — no como lección, sino como la tradición oral guaraní lo hace naturalmente: por evocación, por asociación, sin pedirlo.

---

## Arquitectura

Mombeu tiene tres componentes:

### 1. Encoder de pertinencia temática
Modelo de clasificación liviano que decide si una respuesta merece un cierre cultural. No genera texto — solo responde si insertar o no.

```python
# Modelo base
"intfloat/multilingual-e5-small"

# Escala de pertinencia
naturaleza, clima, cosmos        → alta
emociones, familia, comunidad    → alta
alimentación, tierra, animales   → media
historia, sociedad, identidad    → media
tareas técnicas, tecnología      → baja o nula
```

### 2. Modelo DPO de inserción narrativa
Modelo generativo pequeño fine-tuneado con LoRA + DPO que genera el cierre narrativo cuando el encoder lo habilita. Entrenado con pares chosen/rejected anotados por expertos en cosmovisión guaraní.

### 3. Grafo de conocimiento mitológico
Extraído del texto *Ñande Ypykuéra*. Captura personajes, relaciones y atributos de la cosmovisión guaraní. Cumple dos funciones:
- Validación automática de cierres antes de anotación humana
- Balanceo de personajes en el dataset

---

## Dataset

El dataset de preferencias es la contribución central del proyecto — el primer corpus de pares chosen/rejected basado en cosmovisión guaraní.

| Campo | Descripción |
|---|---|
| `prompt` | Respuesta del LLM grande |
| `chosen` | Cierre mitológico correcto y naturalmente integrado |
| `rejected` | Cierre incorrecto, forzado o culturalmente inexacto |

### Fuente cultural

Todos los cierres provienen exclusivamente del texto *Ñande Ypykuéra*. El modelo no inventa ni mezcla fuentes — esto garantiza coherencia cultural y trazabilidad completa.

### Criterios de anotación humana

- ¿El personaje mencionado existe en *Ñande Ypykuéra*? (Sí/No)
- ¿La relación descrita es correcta según el grafo? (Sí/No)
- ¿La conexión temática es natural? (1–3)
- ¿El cierre suena fluido con términos en guaraní? (1–3)

---

## Sobre el riesgo de exotización

Mombeu toma cuatro medidas explícitas para evitar folklorizar o estereotipar la cultura guaraní:

1. **Fuente única.** Solo *Ñande Ypykuéra* — sin mezcla de fuentes ni referencias de internet.
2. **Validación experta.** Anotadores con conocimiento especializado en cosmovisión guaraní.
3. **Grafo como validador.** Rechaza automáticamente cierres que contradicen el texto fuente.
4. **Naturalidad como criterio.** Un cierre correcto pero forzado es rechazado. El objetivo es que suene como el abuelo, no como un libro de folklore.

---

## Evaluación

Se comparan cuatro condiciones con evaluación humana:

| Condición | Descripción |
|---|---|
| Baseline | LLM grande sin capa cultural |
| Prompt engineering | Instrucción en el sistema para agregar cierres |
| SFT | Fine-tuning con ejemplos del formato |
| DPO | Alineamiento con preferencias culturales |

Dimensiones evaluadas: naturalidad, memorabilidad, percepción cultural, intrusividad.

---

## Instalación

```bash
git clone https://github.com/tu-usuario/mombeu
cd mombeu
pip install -r requirements.txt
```

## Uso rápido

```python
from mombeu import MombeuLayer

# Inicializar la capa
mombeu = MombeuLayer()

# Conectar a cualquier respuesta de LLM
respuesta_base = "Los truenos ocurren cuando..."
respuesta_enriquecida = mombeu.enrich(respuesta_base)

print(respuesta_enriquecida)
```

---

## Stack tecnológico

| Componente | Herramienta |
|---|---|
| Encoder de pertinencia | multilingual-e5-small + fine-tuning |
| Modelo generativo | LLaMA 3.1 8B o Mistral 7B |
| Fine-tuning | LoRA + SFT |
| Alineamiento | DPO |
| Grafo | NetworkX |
| Demo | Gradio en HuggingFace Spaces |

---

## Contribuciones

1. Arquitectura de capa cultural complementaria — encoder + modelo DPO que enriquece la salida de cualquier LLM sin modificar su arquitectura
2. Dataset de preferencias culturales guaraní — primer corpus de pares chosen/rejected basado en cosmovisión guaraní
3. Grafo de conocimiento mitológico — estructura verificable extraída de *Ñande Ypykuéra*

---

## Aliados institucionales

- **Secretaría de Políticas Lingüísticas (SPL)** — validación cultural y anotadores expertos
- **Ateneo de Lengua y Cultura Guaraní** — validación mitológica
- **SomosNLP** — plataforma del hackathon

---

## ODS vinculados

- **ODS 4** — Educación de calidad
- **ODS 10** — Reducción de desigualdades lingüísticas en IA
- **ODS 16** — Derechos lingüísticos y acceso cultural

---

## Hackathon

Este proyecto fue desarrollado para el **Hackathon SomosNLP 2026**, en la categoría de alineamiento cultural de LLMs.

---

## Licencia

MIT

---

> *"Porque cada respuesta es una oportunidad de recordar quiénes somos."*
