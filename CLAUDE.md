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

- **Idioma del código: inglés**. Todo el código del repositorio debe ser comprensible íntegramente en inglés: nombres de ficheros, variables, funciones, clases, comentarios, docstrings, mensajes de log y de commit. La interfaz de usuario y el informe final, en cambio, se presentan en español (es contenido de producto, no código).
- **Python**: PEP 8, type hints en todas las funciones, docstrings en funciones y clases públicas.
- **Async**: usar `async`/`await` en FastAPI siempre que sea posible.
- **Frontend**: componentes funcionales React con hooks. Sin class components.
- **Sin valores hardcodeados**: todo configurable vía variables de entorno o `settings`.
- **Logs**: usar el módulo estándar `logging` de Python con logs estructurados.
- **Dependencias**: gestionadas con `uv`. Añadir con `uv add <pkg>`, nunca `pip install`. Commitear siempre `uv.lock`.

---

## 2. Flujo de Desarrollo (OBLIGATORIO)

Este proceso se aplica a **cada unidad de trabajo**, sin excepción.

### 2.1 Ciclo completo por unidad de trabajo

```
1. Issue en GitHub → verificar: título claro, criterios de aceptación, labels y milestone presentes
2. Rama desde main  → verificar: nombre sigue la convención (feature/|fix/|test/ + descripción-corta)
3. Tests primero (TDD) → verificar: los tests fallan antes de implementar
4. Implementación → verificar: los tests pasan; todos los tests previos siguen pasando
5. Pull Request → verificar: referencia el número de issue; CI verde antes de mergear
```

### 2.2 GitHub Issues

Cada issue debe incluir:
- Título descriptivo en inglés.
- Descripción con criterios de aceptación claros.
- Labels `módulo:` y `tipo:` apropiados.
- Milestone correspondiente.

### 2.3 Ramas

Convención de nombres (desde `main`):
- `feature/short-description` — nueva funcionalidad
- `fix/short-description` — corrección de bug
- `test/short-description` — solo tests

### 2.4 TDD ligero

1. Escribir los tests → deben **fallar**.
2. Escribir la implementación mínima → tests deben **pasar**.
3. Verificar que todos los tests previos siguen pasando.

### 2.5 Pull Requests

- El PR referencia el número de issue (`Closes #N`).
- No se mergea hasta que **todos los checks de CI estén en verde**.
- CI backend: Ruff lint + pytest.
- CI frontend: ESLint + build de producción Vite.
- CodeQL en cada push a `main`.

### 2.6 Commits (Conventional Commits — ESTRICTO)

Formato: `type(scope): description`

| Tipo | Uso |
|------|-----|
| `feat` | Nueva funcionalidad |
| `fix` | Corrección de bug |
| `test` | Añadir o modificar tests |
| `docs` | Solo documentación |
| `refactor` | Refactorización sin cambio de comportamiento |
| `ci` | Cambios en pipelines CI/CD |
| `chore` | Mantenimiento, dependencias |

Ejemplos válidos:
```
feat(sentiment): add NewsAPI client with pagination
test(api): add contract tests for report endpoint
fix(deep-learning): handle missing model file gracefully
```

**Granularidad de commits:**
- No agrupar todos los cambios de una rama en un único commit. Dividir el trabajo en
  commits pequeños, agrupados por objetivo o funcionalidad.
- Cada commit debe explicar **un solo cambio** coherente.
- Ejemplo: si en una rama se escriben los tests y después se implementa la
  funcionalidad, hacer dos commits (uno `test(...)` y otro `feat(...)`), no uno solo.

**Reglas absolutas para commits, issues y PRs:**
- Escritos en **inglés**.
- **Sin ninguna referencia a herramientas de IA, modelos de lenguaje o asistentes de IA** de ningún tipo. El texto debe leerse como si lo hubiera escrito el desarrollador.

---

## 3. Contexto del Proyecto

### 3.1 Descripción general

TFG de Ingeniería Informática — Universidad de León (España).
Plataforma web de análisis financiero de acciones del mercado americano (Nasdaq).
**Deadline: finales de junio / principios de julio de 2026.**

Un único desarrollador principal es responsable de toda la aplicación web.
El colaborador del Grado en Finanzas entrega el análisis técnico y fundamental en Python.

**Objetivo**: proporcionar a cualquier usuario —con o sin conocimientos financieros—
un informe de análisis completo y comprensible de una acción bursátil, combinando
cuatro módulos de análisis y presentando el resultado en español.

### 3.2 Stack tecnológico

| Capa | Tecnología | Notas |
|------|-----------|-------|
| Frontend | React 18 + Vite + Tailwind CSS | SPA, desplegada en Render |
| Backend | FastAPI (Python 3.11+) | Desplegado en Render (free tier, sin GPU) |
| BD grafos | Neo4j AuraDB free tier (512 MB) | Módulo de sentimiento (GraphRAG) |
| GraphRAG lib | neo4j-graphrag (Python) | Retrieval vectorial sobre Neo4j |
| LLM activo | OpenAI API — gpt-5.4-mini | Chat completion + extracción estructural, de pago (`LLM_PROVIDER=openai`) |
| LLM desarrollo | Ollama local — múltiples modelos | RTX 4060, offline (`LLM_PROVIDER=ollama`, modelo vía `OLLAMA_MODEL`) |
| Embeddings | OpenAI — text-embedding-3-large | 3 072 dims, vector index en Neo4j; obligatorio siempre, sin importar `LLM_PROVIDER` |
| ML | PyTorch — modelos GRU por ticker | Ficheros .pt + .json de metadatos |
| Scheduler | APScheduler embebido en FastAPI | Reentrenamiento diario automático |
| Config | pydantic-settings + variables de entorno | Nunca valores hardcodeados |
| Gestión de deps | uv | `pyproject.toml` + `uv.lock` commiteado; nunca `pip` directo |
| Hosting | Render (gratuito) | Frontend y backend en la misma plataforma; deploys automáticos desde Git; acepta ficheros .pt |
| Control de versiones | Git / GitHub | Repositorio compartido |

### 3.3 Estructura de carpetas

```
/
├── .github/
│   └── workflows/
│       ├── backend.yml        # CI: Ruff + pytest
│       ├── frontend.yml       # CI: ESLint + Vite build
│       └── codeql.yml         # Análisis de seguridad en pushes a main
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Disclaimer.jsx
│   │   │   ├── ErrorMessage.jsx
│   │   │   ├── LoadingSpinner.jsx
│   │   │   └── SearchBar.jsx
│   │   ├── pages/
│   │   │   ├── HomePage.jsx
│   │   │   ├── PrivacyPage.jsx
│   │   │   └── ReportPage.jsx
│   │   ├── services/
│   │   │   └── api.js         # Llamadas a la API REST
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── eslint.config.js
│   ├── index.html
│   ├── package.json
│   ├── postcss.config.js
│   ├── tailwind.config.js
│   └── vite.config.js
│
├── backend/
│   ├── app/
│   │   ├── api/v1/            # Endpoints REST
│   │   │   ├── health.py
│   │   │   ├── sentiment.py
│   │   │   ├── deep_learning.py
│   │   │   ├── fundamental.py
│   │   │   ├── technical.py
│   │   │   └── report.py
│   │   ├── services/          # Lógica de negocio por módulo
│   │   │   ├── sentiment/service.py
│   │   │   ├── deep_learning/service.py
│   │   │   ├── fundamental/service.py
│   │   │   └── technical/service.py
│   │   ├── llm/               # Patrón Adaptador LLM (CRÍTICO)
│   │   │   ├── base.py        # Interfaz abstracta LLMService
│   │   │   ├── openai_provider.py
│   │   │   ├── ollama_provider.py
│   │   │   └── factory.py     # get_llm_service() con lru_cache
│   │   ├── models/            # Schemas Pydantic (request/response)
│   │   │   ├── common.py
│   │   │   ├── sentiment.py
│   │   │   ├── deep_learning.py
│   │   │   ├── fundamental.py
│   │   │   ├── technical.py
│   │   │   └── report.py
│   │   ├── core/
│   │   │   └── config.py      # pydantic-settings
│   │   ├── scheduler/
│   │   │   └── jobs.py        # APScheduler jobs
│   │   └── main.py            # Punto de entrada FastAPI
│   ├── scripts/               # Procesos batch offline (construcción del grafo)
│   │   ├── build_knowledge_graph.py
│   │   └── kg_builder/
│   │       ├── loaders/
│   │       │   ├── finentity.py
│   │       │   └── finmarba.py
│   │       ├── data/          # Datasets FinEntity + FinMarBa (en .gitignore)
│   │       ├── embeddings.py
│   │       ├── entity_resolver.py
│   │       ├── llm_extractor.py
│   │       ├── neo4j_client.py
│   │       ├── schema.py
│   │       └── topics.py
│   ├── tests/
│   │   ├── api/
│   │   │   └── test_health.py
│   │   └── conftest.py
│   ├── ml_models/             # Ficheros .pt + .json por ticker (en .gitignore)
│   ├── .env.example
│   ├── pyproject.toml         # Dependencias (fuente de verdad)
│   └── uv.lock                # Lockfile commiteado en git
│
├── docs/                      # Documentación del proyecto
│   └── adr/                   # Architecture Decision Records (decisiones de diseño)
│
├── data/                      # CSVs OHLC de la fase de investigación (gitignored; fuente activa: yfinance)
├── notebooks/                 # Jupyter notebooks de entrenamiento
├── .gitignore
├── .gitattributes
├── LICENSE
├── README.md
└── CLAUDE.md
```

> El documento de requisitos (SRS, IEEE 830) es **privado y está en constante
> cambio**, por lo que no forma parte del repositorio (está en `.gitignore`). Las
> decisiones de diseño que introducen sus revisiones se registran como ADRs en
> [`docs/adr/`](docs/adr/).

### 3.4 Módulos funcionales

**Módulo 1 — Análisis de Sentimiento (GraphRAG)**
- Flujo: Finnhub company_news (fallback: NewsAPI) → embeddings semánticos → búsqueda Neo4j (k-hop, k=2) → OpenAI LLM
- Noticias vía cadena de proveedores (Adapter + Chain of Responsibility, ver [ADR-0005](docs/adr/0005-news-provider-fallback-chain.md)): Finnhub primario (sin delay, solo Norteamérica), NewsAPI detrás (universal, 24 h de delay); `NEWS_PROVIDER` elige la cabeza de la cadena
- Salida: `{label, score[-1,1], confidence[0,1], explanation, influential_news[{title,url}]}`
- Caché: TTL configurable (por defecto 30 min) — protege las cuotas de ambos proveedores

**Módulo 2 — Deep Learning (GRU)**
- Flujo: datos EOD yfinance (`app.core.market_data`) → ventana lookback → GRU → retorno a 10 días → tendencia (banda neutral). Decisión de arquitectura en [ADR-0006](docs/adr/0006-gru-architecture-no-vmd.md); fuente de datos en [ADR-0008](docs/adr/0008-dl-data-source-and-nasdaq-scope.md)
- Salida: `{trend (alcista|bajista|neutral), predicted_return_pct, predicted_price (derivado), current_price, horizon_days, metrics{rmse,mae,directional_accuracy}, trained_at}`
- Receta congelada (`notebooks/results/optuna/gru_frozen_config.json`); reentrenamiento de pesos por ticker
- Lazy loading de modelos .pt con caché LRU (máx. configurable, por defecto 10 modelos)
- Actualización diaria automática vía APScheduler (~22:00 CET) con datos reales únicamente
- Cobertura: acciones del Nasdaq (universo validado por el HPO); otros mercados no están soportados (ver ADR-0008)

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

### 3.5 Patrón LLM centralizado (CRÍTICO)

Todos los módulos llaman al LLM a través de `LLMService` (interfaz abstracta).
Cambiar de proveedor **no debe requerir modificar la lógica de ningún módulo**.

```
LLMService (base.py — clase abstracta)
    .complete(system_prompt: str, user_prompt: str) -> str   [async]
        ├── OpenAIProvider    → OpenAI API (gpt-5.4-mini)   — ACTIVO, DE PAGO
        └── OllamaProvider    → Ollama local (múltiples modelos) — DESARROLLO
```

El proveedor activo se selecciona con `LLM_PROVIDER=openai|ollama` (variable de entorno).
`factory.py` expone `get_llm_service()` con `lru_cache` para singleton.

El embedder (`text-embedding-3-large`) es siempre OpenAI y **no** sigue el switch de
`LLM_PROVIDER`: el grafo de conocimiento en Neo4j se construyó con esos embeddings, por
lo que `OPENAI_API_KEY` es obligatoria incluso con `LLM_PROVIDER=ollama` (ver
[ADR-0009](docs/adr/0009-drop-groq-llm-provider.md)).

### 3.6 Variables de entorno requeridas

El fichero `.env.example` debe contener exactamente estas variables (sin valores):

```bash
# LLM
LLM_PROVIDER=openai                    # openai | ollama
OPENAI_API_KEY=                        # obligatoria siempre: los embeddings del grafo la usan sin importar LLM_PROVIDER
OPENAI_MODEL=gpt-5.4-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# Servicios externos
NEO4J_URI=
NEO4J_USERNAME=
NEO4J_PASSWORD=
NEWSAPI_KEY=
FINNHUB_API_KEY=

# Noticias (proveedor primario; el otro actúa como fallback)
NEWS_PROVIDER=finnhub                  # finnhub | newsapi

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

### 3.7 Restricciones críticas del sistema

| Restricción | Impacto | Estrategia |
|-------------|---------|------------|
| Sin GPU en producción (Render free tier) | Inferencia GRU en CPU, mayor latencia | Caché LRU de modelos; informar al usuario del tiempo estimado |
| Render se "duerme" tras 15 min de inactividad | Cold start de 30–60 s | Petición de calentamiento previa a la demo del TFG |
| Finnhub (noticias, primario): 60 req/min, solo Norteamérica | Sin noticias frescas para tickers no norteamericanos | Cadena de fallback a NewsAPI (ADR-0005) |
| NewsAPI (noticias, fallback): 100 req/día y 24 h de delay | Agotamiento fácil; noticias no inmediatas | Solo recibe tickers que Finnhub no cubre + caché TTL 30 min |
| Datos solo EOD (no tiempo real intradía) | No se puede hacer análisis intradiario | Todo el análisis de precio se basa en datos de cierre |
| Modelos GRU solo para acciones del Nasdaq | Soporte parcial para otras acciones | El HPO se realizó sobre tickers US Nasdaq (AAPL, NVDA, PEP); aplicar la receta congelada fuera de ese dominio no está validado. Informar al usuario con RF-27; ver [ADR-0008](docs/adr/0008-dl-data-source-and-nasdaq-scope.md) |
| Procesamiento interno en inglés | Noticias, embeddings y prompts en inglés | Prompt engineering explícito para respuestas en español |
| Coste OpenAI | API de pago (LLM + embeddings) | Uso acotado al proceso batch offline y análisis por demanda; sin streaming continuo |

### 3.8 Restricciones de seguridad y calidad

- **HTTPS obligatorio** en producción (certificado automático de Render).
- **Rate limiting** en los endpoints de análisis (prevenir abuso y agotamiento de cuotas).
- **Validación del parámetro ticker**: alfanumérico, 2–5 caracteres, mayúsculas; además, comprobación de existencia contra Finnhub symbol lookup (match exacto, cacheada, fail-open) antes de buscar noticias.
- **CORS** configurado para aceptar solo el dominio del frontend en producción.
- **XSS**: escapar todo contenido generado por el LLM antes de renderizarlo en el DOM.
- **Claves API**: nunca en el código fuente ni en el repositorio. Solo variables de entorno.
- **WCAG 2.1 nivel AA**: estándar legal en España; se valora positivamente en el TFG.
- **Disclaimer legal**: visible de forma permanente en la interfaz (RF-06) y en el informe (RF-37).
- **Cookies y consentimiento (LSSI-CE art. 22, AEPD)**: banner de consentimiento granular por categoría (estrictamente necesarias, funcionales, analíticas); el consentimiento aplica a cualquier almacenamiento en cliente (cookies, `localStorage`, `sessionStorage`, `IndexedDB`). Ver [ADR-0002](docs/adr/0002-cookie-consent-management.md).

---

**Estas directrices son efectivas si:** los diffs contienen menos cambios innecesarios,
hay menos reescrituras por sobrecomplicación, y las preguntas de aclaración llegan
antes de la implementación en lugar de después de los errores.