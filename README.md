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
> Los truenos ocurren cuando el aire caliente sube rápidamente y genera descargas eléctricas en las nubes. Eso me recuerda a **Tupã**, cuya voz retumba en el cielo — para el pueblo guaraní, el trueno no era un fenómeno sino la presencia del creador.

---

## Motivación

Paraguay es el único país de América donde una lengua indígena —el guaraní— es co-oficial y hablada por la mayoría de la población. Sin embargo, los LLMs actuales ignoran casi completamente su cultura y cosmovisión.

Mombeu propone que cada respuesta sea una oportunidad de transmitir memoria colectiva — no como lección, sino como la tradición oral guaraní lo hace naturalmente: por evocación, por asociación, sin pedirlo.

---

## Arquitectura

Mombeu tiene tres componentes internos:

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

## Instalación

```bash
pip install mombeu
```

No se descarga ningún modelo al instalar. El modelo se elige y descarga explícitamente con `mombeu.init()`.

---

## Uso rápido

```python
import mombeu

# 1. Inicializar una vez — descarga y cachea el modelo en el primer uso
mombeu.init(model="mombeu-v1", hf_token="hf_...")

# 2. Usar exactamente igual que openai.OpenAI
client = mombeu.OpenAI(api_key="sk-...")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "¿Qué son los truenos?"}],
)

print(response.choices[0].message.content)
# → Respuesta de OpenAI ... seguida del cierre narrativo guaraní
```

---

## Streaming

```python
mombeu.init(model="mombeu-v1", hf_token="hf_...")
client = mombeu.OpenAI(api_key="sk-...")

for chunk in client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "¿Qué son los truenos?"}],
    stream=True,
):
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="", flush=True)
```

**Comportamiento del streaming:**
- Todos los chunks de OpenAI se emiten en tiempo real, sin cambios.
- Al terminar el stream, el modelo local genera el cierre narrativo.
- El cierre se emite como un único chunk final.
- Se agrega un chunk de stop para terminar el stream correctamente.

---

## Modelos disponibles

```python
import mombeu

print(mombeu.list_models())
# ['mombeu-fast', 'mombeu-v1', 'mombeu-v2']
```

| Slug           | Descripción                                  |
|----------------|----------------------------------------------|
| `mombeu-v1`    | Modelo de inserción narrativa general        |
| `mombeu-v2`    | Mayor precisión cultural, más pesado         |
| `mombeu-fast`  | Liviano, menor latencia                      |

---

## API reference

### `mombeu.init(model, hf_token)`

Inicializa Mombeu. Debe llamarse antes de crear un cliente.

| Parámetro  | Tipo  | Descripción                                                    |
|------------|-------|----------------------------------------------------------------|
| `model`    | `str` | Slug del modelo. Ver `mombeu.list_models()`.                   |
| `hf_token` | `str` | Token de HuggingFace con acceso de lectura al repositorio.     |

Lanza `ValueError` si el slug no es soportado o el token es inválido.  
Lanza `RuntimeError` si el pipeline del modelo falla al cargar.

---

### `mombeu.OpenAI(**kwargs)`

Reemplazo directo de `openai.OpenAI`. Acepta los mismos argumentos de constructor.

Todos los atributos no relacionados con `chat.completions` se delegan transparentemente a la instancia subyacente de `openai.OpenAI` (e.g. `embeddings`, `images`, `audio`).

---

### `mombeu.list_models() → list[str]`

Retorna la lista ordenada de slugs de modelos soportados.

---

### `mombeu.current_model() → str | None`

Retorna el slug del modelo actualmente cargado, o `None` si aún no se inicializó.

---

### `mombeu.reset()`

Limpia todo el estado en memoria (modelo, pipeline). Útil para testing o cambio de modelo.

---

## Dataset

El dataset de preferencias es la contribución central del proyecto — el primer corpus de pares chosen/rejected basado en cosmovisión guaraní.

| Campo | Descripción |
|---|---|
| `prompt` | Respuesta del LLM grande |
| `chosen` | Cierre mitológico correcto y naturalmente integrado |
| `rejected` | Cierre incorrecto, forzado o culturalmente inexacto |

Todos los cierres provienen exclusivamente del texto *Ñande Ypykuéra*. El modelo no inventa ni mezcla fuentes — esto garantiza coherencia cultural y trazabilidad completa.

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

| Condición         | Descripción                                             |
|-------------------|---------------------------------------------------------|
| Baseline          | LLM grande sin capa cultural                            |
| Prompt engineering| Instrucción en el sistema para agregar cierres          |
| SFT               | Fine-tuning con ejemplos del formato                    |
| DPO               | Alineamiento con preferencias culturales                |

Dimensiones evaluadas: naturalidad, memorabilidad, percepción cultural, intrusividad.

---

## Stack tecnológico

| Componente              | Herramienta                          |
|-------------------------|--------------------------------------|
| Encoder de pertinencia  | multilingual-e5-small + fine-tuning  |
| Modelo generativo       | LLaMA 3.1 8B o Mistral 7B            |
| Fine-tuning             | LoRA + SFT                           |
| Alineamiento            | DPO                                  |
| Grafo                   | NetworkX                             |
| Wrapper Python          | openai SDK + transformers            |
| Demo                    | Gradio en HuggingFace Spaces         |

---

## Desarrollo

```bash
# Instalar con dependencias de desarrollo
pip install -e ".[dev]"

# Correr tests
pytest tests/ -v

# Formatear
black mombeu/ tests/

# Lint
ruff check mombeu/ tests/
```

---

## Notas técnicas

- El modelo se cachea en `~/.cache/huggingface/` tras la primera descarga.
- Se usa GPU automáticamente si CUDA está disponible; si no, cae a CPU.
- La generación del cierre agrega latencia proporcional a `max_new_tokens` (default: 150).
- Si la generación falla por cualquier motivo, se retorna la respuesta original de OpenAI sin modificar, con un warning.

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

## Licencia

MIT

---

> *"Porque cada respuesta es una oportunidad de recordar quiénes somos."*
