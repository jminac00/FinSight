# FinSight

Plataforma web de análisis financiero de acciones del Nasdaq.
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
| GET | `/api/v1/prediction/{ticker}` | Predicción LSTM |
| GET | `/api/v1/fundamental/{ticker}` | Análisis fundamental |
| GET | `/api/v1/technical/{ticker}` | Análisis técnico |

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
| `LLM_PROVIDER` | `groq` (producción) o `ollama` (desarrollo local) |
| `GROQ_API_KEY` | Clave de la API de Groq |
| `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` | Conexión a Neo4j AuraDB |
| `NEWSAPI_KEY` | Clave de NewsAPI |
| `FINNHUB_API_KEY` | Clave de Finnhub |
| `ENVIRONMENT` | `development` o `production` |

---

## Despliegue en Render (backend)

Configurar en el dashboard de Render:

- **Build command:** `pip install uv && uv sync --no-dev --frozen`
- **Start command:** `uv run uvicorn app.main:app --host 0.0.0.0 --port $PORT`

---

## Estructura del proyecto

```
/
├── frontend/          # React 18 + Vite + Tailwind CSS
├── backend/           # FastAPI (Python 3.11+)
│   ├── app/
│   │   ├── api/v1/    # Endpoints REST
│   │   ├── services/  # Lógica de negocio por módulo
│   │   ├── llm/       # Adaptador LLM (Groq / Ollama)
│   │   ├── models/    # Schemas Pydantic
│   │   ├── core/      # Configuración (pydantic-settings)
│   │   └── scheduler/ # APScheduler (reentrenamiento diario)
│   ├── ml_models/     # Ficheros .pt y .json por ticker
│   ├── pyproject.toml # Dependencias (fuente de verdad)
│   └── uv.lock        # Lockfile (commiteado en git)
├── data/              # CSVs históricos OHLC (en .gitignore)
└── notebooks/         # Jupyter notebooks de entrenamiento
```
