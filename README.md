# FinSight

Plataforma web de análisis financiero de acciones del Nasdaq.
Trabajo de Fin de Grado — Ingeniería Informática, Universidad de León.
Autor: Javier Miñambres Calvo

---

## Arranque en local

### Prerequisitos

- Python 3.11+
- Node.js 18+ y npm

---

### Backend (FastAPI)

```bash
cd backend

# 1. Crear y activar entorno virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus claves de API

# 4. Arrancar el servidor
uvicorn app.main:app --reload
```

El backend queda disponible en `http://localhost:8000`.
Documentación interactiva: `http://localhost:8000/api/docs`

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
│   └── ml_models/     # Ficheros .pt y .json por ticker
├── data/              # CSVs históricos OHLC (en .gitignore)
└── notebooks/         # Jupyter notebooks de entrenamiento
```
