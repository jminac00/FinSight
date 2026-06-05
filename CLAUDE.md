# FinSight — CLAUDE.md

Contexto completo del proyecto y directrices de comportamiento para Claude Code.
Leer completo antes de cualquier acción.

---

## 1. Directrices de Comportamiento

### 1.1 Pensar antes de codificar

**No asumir. No ocultar confusión. Exponer los tradeoffs.**

Antes de implementar cualquier cosa:
- Enunciar explícitamente los supuestos. Si hay incertidumbre, preguntar.
- Si existen varias interpretaciones válidas, presentarlas — no elegir en silencio.
- Si existe un enfoque más simple, decirlo. Rebatir cuando esté justificado.
- Si algo no está claro, parar. Nombrar qué es confuso. Preguntar.

### 1.2 Simplicidad primero

**El mínimo código que resuelve el problema. Nada especulativo.**

- Sin funcionalidades más allá de lo pedido.
- Sin abstracciones para código de un solo uso.
- Sin "flexibilidad" o "configurabilidad" que no hayan sido solicitadas.
- Sin manejo de errores para escenarios imposibles.
- Si se escriben 200 líneas y podrían ser 50, reescribir.

Pregunta de control: "¿Diría un senior engineer que esto está sobrecomplicado?" Si la respuesta es sí, simplificar.

### 1.3 Cambios quirúrgicos

**Tocar solo lo imprescindible. Limpiar únicamente el propio desorden.**

Al editar código existente:
- No "mejorar" código adyacente, comentarios ni formato.
- No refactorizar lo que no está roto.
- Respetar el estilo existente, aunque se haría de otra manera.
- Si se detecta código muerto no relacionado, mencionarlo — no borrarlo.

Cuando los cambios propios dejan huérfanos:
- Eliminar los imports/variables/funciones que los cambios propios hayan dejado sin uso.
- No eliminar código muerto preexistente salvo que se pida explícitamente.

Prueba de control: cada línea modificada debe trazarse directamente a la petición del usuario.

### 1.4 Ejecución orientada a objetivos

**Definir criterios de éxito. Iterar hasta verificarlos.**

Transformar las tareas en objetivos verificables:
- "Añadir validación" → "Escribir tests para inputs inválidos, luego hacerlos pasar"
- "Corregir el bug" → "Escribir un test que lo reproduzca, luego hacerlo pasar"
- "Refactorizar X" → "Asegurar que los tests pasan antes y después"

Para tareas de múltiples pasos, enunciar un plan breve antes de empezar:

```
1. [Paso] → verificar: [comprobación]
2. [Paso] → verificar: [comprobación]
3. [Paso] → verificar: [comprobación]
```

Criterios de éxito sólidos permiten iterar de forma autónoma.
Criterios débiles ("que funcione") exigen aclaraciones constantes.

### 1.5 Convenciones de código del proyecto

- **Python**: PEP 8, type hints en todas las funciones, docstrings en funciones y clases públicas.
- **Async**: usar `async`/`await` en FastAPI siempre que sea posible.
- **Frontend**: componentes funcionales React con hooks. Sin class components.
- **Sin valores hardcodeados**: todo configurable vía variables de entorno o `settings`.
- **Logs**: usar el módulo estándar `logging` de Python con logs estructurados.
- **Dependencias**: gestionadas con `uv`. Añadir con `uv add <pkg>`, nunca `pip install`. Commitear siempre `uv.lock`.

---

## 2. Contexto del Proyecto

### 2.1 Descripción general

TFG de Ingeniería Informática — Universidad de León (España).
Plataforma web de análisis financiero de acciones del mercado americano (Nasdaq).
**Deadline: finales de junio / principios de julio de 2026.**

Un único desarrollador principal es responsable de toda la aplicación web.
El colaborador del Grado en Finanzas entrega el análisis técnico y fundamental en Python.

**Objetivo**: proporcionar a cualquier usuario —con o sin conocimientos financieros—
un informe de análisis completo y comprensible de una acción bursátil, combinando
cuatro módulos de análisis y presentando el resultado en español.

### 2.2 Stack tecnológico

| Capa | Tecnología | Notas |
|------|-----------|-------|
| Frontend | React 18 + Vite + Tailwind CSS | SPA, desplegada en Vercel |
| Backend | FastAPI (Python 3.11+) | Desplegado en Render (free tier, sin GPU) |
| BD grafos | Neo4j AuraDB free tier (512 MB) | Módulo de sentimiento (GraphRAG) |
| GraphRAG lib | neo4j-graphrag (Python) | Retrieval vectorial sobre Neo4j |
| LLM activo | OpenAI API — gpt-5.4-mini | Chat completion + extracción estructural |
| LLM alternativo | Groq API — Llama 3.1 70B | Fallback gratuito (`LLM_PROVIDER=groq`) |
| LLM desarrollo | Ollama local — Llama 3.1 8B | RTX 4060, offline (`LLM_PROVIDER=ollama`) |
| Embeddings | OpenAI — text-embedding-3-large | 3 072 dims, vector index en Neo4j |
| ML | PyTorch — modelos LSTM por ticker | Ficheros .pt + .json de metadatos |
| Scheduler | APScheduler embebido en FastAPI | Reentrenamiento diario automático |
| Config | pydantic-settings + variables de entorno | Nunca valores hardcodeados |
| Gestión de deps | uv | `pyproject.toml` + `uv.lock` commiteado; nunca `pip` directo |
| Hosting frontend | Vercel (gratuito) | Deploys automáticos desde Git |
| Hosting backend | Render (gratuito) | Free tier Python, acepta ficheros .pt |
| Control de versiones | Git / GitHub | Repositorio compartido |

### 2.3 Estructura de carpetas objetivo

```
/
├── frontend/
│   └── src/
│       ├── components/
│       ├── pages/
│       └── services/          # Llamadas a la API REST
│
├── backend/
│   ├── app/
│   │   ├── api/v1/            # Endpoints REST
│   │   │   ├── sentiment.py
│   │   │   ├── deep_learning.py
│   │   │   ├── fundamental.py
│   │   │   ├── technical.py
│   │   │   └── report.py
│   │   ├── services/          # Lógica de negocio por módulo
│   │   │   ├── sentiment/
│   │   │   ├── deep_learning/
│   │   │   ├── fundamental/
│   │   │   └── technical/
│   │   ├── llm/               # Patrón Adaptador LLM (CRÍTICO)
│   │   │   ├── base.py        # Interfaz abstracta LLMService
│   │   │   ├── openai_provider.py
│   │   │   ├── groq_provider.py
│   │   │   ├── ollama_provider.py
│   │   │   └── factory.py     # get_llm_service() con lru_cache
│   │   ├── models/            # Schemas Pydantic (request/response)
│   │   ├── core/              # Config (pydantic-settings), middleware
│   │   └── scheduler/         # APScheduler jobs
│   ├── scripts/               # Procesos batch offline (construcción del grafo)
│   │   ├── build_knowledge_graph.py
│   │   └── kg_builder/        # Módulos de ingesta: loaders, embeddings, neo4j, LLM
│   │       └── data/          # Datasets de entrenamiento del grafo (FinEntity, FinMarBa)
│   ├── ml_models/             # Ficheros .pt + .json por ticker
│   ├── .env.example
│   ├── pyproject.toml         # Dependencias (fuente de verdad)
│   └── uv.lock                # Lockfile commiteado en git
│
├── data/                      # CSVs históricos OHLC para Deep Learning (en .gitignore)
├── notebooks/                 # Jupyter notebooks de entrenamiento
├── .gitignore
├── README.md
└── CLAUDE.md
```

### 2.4 Módulos funcionales

**Módulo 1 — Análisis de Sentimiento (GraphRAG)**
- Flujo: NewsAPI → embeddings semánticos → búsqueda Neo4j (k-hop, k=2) → OpenAI LLM
- Salida: `{label, score[-1,1], confidence[0,1], explanation, influential_news[{title,url}]}`
- Caché: TTL configurable (por defecto 30 min) — NewsAPI tiene límite de 100 req/día

**Módulo 2 — Deep Learning (LSTM + VMD)**
- Flujo: datos EOD Finnhub → preprocesamiento VMD (mismos params del entrenamiento) → LSTM → tendencia
- Salida: `{trend, predicted_price, current_price, pct_change, horizon_days, metrics{rmse,mae,mape,r2}, trained_at}`
- Lazy loading de modelos .pt con caché LRU (máx. configurable, por defecto 10 modelos)
- Actualización diaria automática vía APScheduler (~22:00 CET) con datos reales únicamente

**Módulo 3 — Análisis Fundamental**
- Flujo: Yahoo Finance / Finnhub → OpenAI LLM → explicación en lenguaje natural
- Salida: `{metrics: dict, llm_analysis: str, cached_at: datetime}`
- Caché: hasta el inicio de la siguiente jornada de mercado (datos trimestrales/anuales)

**Módulo 4 — Análisis Técnico**
- Flujo: Finnhub / Yahoo Finance EOD → RSI, MACD, Bollinger, medias móviles → señal
- Salida: `{signal: "alcista|bajista|neutral", indicators: dict, calculated_at: datetime}`
- Implementado por el colaborador de Finanzas; el desarrollador principal lo integra

**Informe consolidado**
- Los 4 módulos se ejecutan en paralelo con `asyncio.gather()`
- OpenAI LLM sintetiza los 4 resultados → conclusión global en español
- Incluye disclaimer legal prominente (RF-06, RNF-32)

### 2.5 Patrón LLM centralizado (CRÍTICO)

Todos los módulos llaman al LLM a través de `LLMService` (interfaz abstracta).
Cambiar de proveedor **no debe requerir modificar la lógica de ningún módulo**.

```
LLMService (base.py — clase abstracta)
    .complete(system_prompt: str, user_prompt: str) -> str   [async]
        ├── OpenAIProvider    → OpenAI API (gpt-5.4-mini)   — ACTIVO
        ├── GroqProvider      → Groq API (Llama 3.1 70B)    — ALTERNATIVO
        └── OllamaProvider    → Ollama local                 — DESARROLLO
```

El proveedor activo se selecciona con `LLM_PROVIDER=openai|groq|ollama` (variable de entorno).
`factory.py` expone `get_llm_service()` con `lru_cache` para singleton.

### 2.6 Variables de entorno requeridas

El fichero `.env.example` debe contener exactamente estas variables (sin valores):

```bash
# LLM
LLM_PROVIDER=openai                    # openai | groq | ollama
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5.4-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
GROQ_API_KEY=
GROQ_MODEL=llama-3.1-70b-versatile
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# Servicios externos
NEO4J_URI=
NEO4J_USERNAME=
NEO4J_PASSWORD=
NEWSAPI_KEY=
FINNHUB_API_KEY=

# Aplicación
FRONTEND_URL=http://localhost:5173
ENVIRONMENT=development                # development | production

# Caché (segundos)
CACHE_TTL_SENTIMENT=1800               # 30 min
CACHE_TTL_FUNDAMENTAL=86400            # 1 día
CACHE_TTL_TECHNICAL=86400              # 1 día

# Parámetros de análisis
PREDICTION_HORIZON_DAYS=10
MAX_NEWS_ARTICLES=10
GRAPH_HOP_DEPTH=2
LRU_CACHE_MAX_MODELS=10
```

### 2.7 Restricciones críticas del sistema

| Restricción | Impacto | Estrategia |
|-------------|---------|------------|
| Sin GPU en producción (Render free tier) | Inferencia LSTM en CPU, mayor latencia | Caché LRU de modelos; informar al usuario del tiempo estimado |
| Render se "duerme" tras 15 min de inactividad | Cold start de 30–60 s | Petición de calentamiento previa a la demo del TFG |
| NewsAPI: 100 req/día en free tier | Agotamiento fácil con múltiples usuarios | Caché agresiva con TTL de 30 min por empresa |
| Datos solo EOD (no tiempo real intradía) | No se puede hacer análisis intradiario | Todo el análisis de precio se basa en datos de cierre |
| Modelos LSTM solo para acciones del Nasdaq | Soporte parcial para otras acciones | Informar al usuario con RF-27; documentado en la memoria |
| Procesamiento interno en inglés | Noticias, embeddings y prompts en inglés | Prompt engineering explícito para respuestas en español |
| Coste OpenAI | API de pago (LLM + embeddings) | Uso acotado al proceso batch offline y análisis por demanda; sin streaming continuo |

### 2.8 Restricciones de seguridad y calidad

- **HTTPS obligatorio** en producción (certificado automático por Vercel/Render).
- **Rate limiting** en los endpoints de análisis (prevenir abuso y agotamiento de cuotas).
- **Validación del parámetro ticker**: alfanumérico, 2–5 caracteres, mayúsculas.
- **CORS** configurado para aceptar solo el dominio del frontend en producción.
- **XSS**: escapar todo contenido generado por el LLM antes de renderizarlo en el DOM.
- **Claves API**: nunca en el código fuente ni en el repositorio. Solo variables de entorno.
- **WCAG 2.1 nivel AA**: estándar legal en España; se valora positivamente en el TFG.
- **Disclaimer legal**: visible de forma permanente en la interfaz (RF-06) y en el informe (RF-37).

---

**Estas directrices son efectivas si:** los diffs contienen menos cambios innecesarios,
hay menos reescrituras por sobrecomplicación, y las preguntas de aclaración llegan
antes de la implementación en lugar de después de los errores.