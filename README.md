# FinSight

Plataforma web de análisis financiero de acciones.
Trabajo de Fin de Grado — Ingeniería Informática, Universidad de León.
Autor: Javier Miñambres Calvo

---

## Arranque en local

### Prerequisitos

- [uv](https://docs.astral.sh/uv/) (gestión de Python y dependencias)
- Node.js 18+ y npm

Instalar `uv` (si no está disponible):
```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

### Backend (FastAPI)

```bash
cd backend

# 1. Crear entorno virtual e instalar dependencias (uv gestiona Python automáticamente)
uv sync

# 2. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus claves de API

# 3. Arrancar el servidor
uv run uvicorn app.main:app --reload
```

El backend queda disponible en `http://localhost:8000`.
Documentación interactiva: `http://localhost:8000/api/docs`

**Comandos uv habituales:**

```bash
uv add <paquete>          # añadir dependencia
uv add --dev <paquete>    # añadir dependencia de desarrollo
uv remove <paquete>       # eliminar dependencia
uv sync                   # instalar/actualizar según uv.lock
uv run <comando>          # ejecutar comando en el entorno virtual
```

**Endpoints disponibles:**

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/api/v1/health` | Estado del servicio |
| GET | `/api/v1/report/{ticker}` | Informe consolidado (4 módulos) |
| GET | `/api/v1/sentiment/{ticker}` | Análisis de sentimiento |
| GET | `/api/v1/prediction/{ticker}` | Predicción GRU |
| GET | `/api/v1/fundamental/{ticker}` | Análisis fundamental |
| GET | `/api/v1/technical/{ticker}` | Análisis técnico |
| GET | `/api/v1/search` | Búsqueda de símbolos por nombre o ticker |
| POST | `/api/v1/train/{ticker}` | Reentrenamiento del modelo GRU bajo demanda (solo desarrollo) |

---

### Frontend (React + Vite)

```bash
cd frontend

# 1. Instalar dependencias
npm install

# 2. Arrancar el servidor de desarrollo
npm run dev
```

El frontend queda disponible en `http://localhost:5173`.
Las peticiones a `/api` se redirigen automáticamente al backend en `localhost:8000`.

---

## Variables de entorno

Copia `backend/.env.example` a `backend/.env` y rellena los valores:

| Variable | Descripción |
|----------|-------------|
| `LLM_PROVIDER` | `openai` (activo, de pago) · `ollama` (desarrollo local, múltiples modelos) |
| `OPENAI_API_KEY` | Clave de la API de OpenAI. **Obligatoria siempre**, incluso con `LLM_PROVIDER=ollama`: los embeddings del grafo de conocimiento (`OPENAI_EMBEDDING_MODEL`) usan OpenAI sin importar el proveedor de chat elegido |
| `OPENAI_MODEL` | Modelo de chat (por defecto `gpt-5.4-mini`) |
| `OPENAI_EMBEDDING_MODEL` | Modelo de embeddings (por defecto `text-embedding-3-large`) |
| `OLLAMA_BASE_URL` | URL de Ollama local (por defecto `http://localhost:11434`) |
| `OLLAMA_MODEL` | Modelo Ollama a usar, seleccionable entre los disponibles localmente (por defecto `llama3.1:8b`) |
| `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` | Conexión a Neo4j AuraDB |
| `NEO4J_DATABASE` | Nombre de la base de datos AuraDB (específico de la instancia, no `neo4j`) |
| `NEWSAPI_KEY` | Clave de NewsAPI |
| `FINNHUB_API_KEY` | Clave de Finnhub |
| `FRONTEND_URL` | URL del frontend (por defecto `http://localhost:5173`) |
| `ENVIRONMENT` | `development` o `production` |
| `CACHE_TTL_SENTIMENT` | TTL caché sentimiento en segundos (por defecto `1800`) |
| `CACHE_TTL_FUNDAMENTAL` | TTL caché fundamental en segundos (por defecto `86400`) |
| `CACHE_TTL_TECHNICAL` | TTL caché técnico en segundos (por defecto `86400`) |
| `CACHE_TTL_SEARCH` | TTL caché búsqueda de símbolos en segundos (por defecto `86400`) |
| `PREDICTION_HORIZON_DAYS` | Horizonte de predicción GRU en días (por defecto `10`) |
| `MAX_NEWS_ARTICLES` | Máximo de artículos de noticias por consulta (por defecto `10`) |
| `GRAPH_HOP_DEPTH` | Profundidad k-hop en Neo4j (por defecto `2`) |
| `LRU_CACHE_MAX_MODELS` | Máximo de modelos GRU en caché LRU (por defecto `10`) |
| `RATE_LIMIT_REPORT` | Límite de peticiones a `/report` por IP (por defecto `10/minute`) |
| `RATE_LIMIT_ANALYSIS` | Límite de peticiones a los endpoints de análisis por IP (por defecto `10/minute`) |
| `RATE_LIMIT_SEARCH` | Límite de peticiones a `/search` por IP (por defecto `60/minute`) |
| `FUNDAMENTAL_DATA_DIR` | Directorio de universos de referencia del análisis fundamental (vacío → ubicación por defecto del paquete) |

---

## Despliegue en Render

Tanto el frontend como el backend se despliegan en Render (free tier), con deploys
automáticos desde Git y HTTPS con certificado automático. La configuración está
definida como Blueprint en [`render.yaml`](render.yaml); Render la detecta
automáticamente al crear los servicios desde el repositorio (New → Blueprint).

**Backend (Web Service):**

- **Build command:** `pip install uv && uv sync --no-dev --frozen`
- **Start command:** `uv run uvicorn app.main:app --host 0.0.0.0 --port $PORT`

**Frontend (Static Site):**

- **Build command:** `npm install && npm run build`
- **Publish directory:** `dist`

---

## Estructura del proyecto

```
/
├── frontend/          # React 18 + Vite + Tailwind CSS
├── backend/           # FastAPI (Python 3.11+)
│   ├── app/
│   │   ├── api/v1/    # Endpoints REST
│   │   ├── services/  # Lógica de negocio por módulo
│   │   ├── llm/       # Adaptador LLM (OpenAI / Ollama)
│   │   ├── models/    # Schemas Pydantic
│   │   ├── core/      # Configuración (pydantic-settings)
│   │   └── scheduler/ # APScheduler (reentrenamiento periódico)
│   ├── scripts/       # Procesos batch offline (construcción del grafo)
│   ├── ml_models/     # Ficheros .pt y .json por ticker
│   ├── pyproject.toml # Dependencias (fuente de verdad)
│   └── uv.lock        # Lockfile (commiteado en git)
├── docs/              # Documentación del proyecto
│   └── adr/           # Architecture Decision Records
├── data/              # CSVs históricos OHLC (en .gitignore)
└── notebooks/         # Jupyter notebooks de entrenamiento
```
