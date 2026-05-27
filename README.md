# ScribeOCR — FastAPI + React

## Updated project structure

```
llama/
├── app.py                        ← Entry point (CLI unchanged; --web starts FastAPI)
├── requirements.txt              ← FastAPI stack + existing project deps
├── .env.example                  ← Backend env vars
│
├── api/                          ← ✦ NEW  FastAPI application
│   ├── __init__.py
│   ├── main.py                   ← create_app() factory + CORS + routers
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── job_schemas.py        ← Pydantic: JobProgress, FontsResponse …
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── logging_middleware.py ← Structured per-request logs
│   │   ├── error_middleware.py   ← Global exception → JSON mapping
│   │   ├── file_validator.py     ← Upload guards + buffer_upload helper
│   │   └── job_store.py          ← Thread-safe in-memory job registry
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── jobs.py               ← POST /api/start-job  GET /api/progress/{id}
│   │   ├── fonts.py              ← GET /api/fonts  GET /api/font-file/{key}
│   │   └── downloads.py          ← GET /api/download/{run_id}/{filename}
│   └── services/
│       ├── __init__.py
│       └── job_service.py        ← ThreadPoolExecutor bridge to pipeline
│
├── configs/
│   └── settings.py               ← ✦ UPDATED  pydantic-settings; all original
│                                    variable names preserved (PAGE_TEMPLATES,
│                                    FONT_OPTIONS, RUNS_DIR, preview_text …)
│
├── frontend/                     ← ✦ NEW  React + Vite + Zustand
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts            ← /api proxy → http://localhost:8000
│   ├── tsconfig.json
│   ├── .env.example
│   └── src/
│       ├── main.tsx
│       ├── App.tsx               ← Shell + global CSS
│       ├── types/
│       │   └── index.ts          ← All shared TypeScript types
│       ├── api/
│       │   └── client.ts         ← fetch wrappers mirroring FastAPI routes
│       ├── store/
│       │   └── useAppStore.ts    ← Zustand (form state + job state)
│       ├── hooks/
│       │   ├── useFonts.ts       ← Fetch fonts + inject @font-face
│       │   ├── useJobPoller.ts   ← 1 s polling loop
│       │   └── useJobSubmit.ts   ← Build FormData, call startJob, start poller
│       └── components/
│           ├── layout/
│           │   └── Sidebar.tsx
│           ├── steps/
│           │   ├── StepImages.tsx
│           │   ├── StepHandwriting.tsx
│           │   └── StepSettings.tsx
│           └── ui/
│               ├── ImageDropzone.tsx
│               ├── ProgressBar.tsx
│               └── ChipGroup.tsx
│
├── engine/                       ← unchanged
├── pipeline/                     ← unchanged
├── segmentation/                 ← unchanged
├── writer/                       ← unchanged
├── assets/                       ← unchanged
├── glyphs/                       ← unchanged
├── data/                         ← unchanged
└── logs/                         ← unchanged
```

## Quick start

### Backend
```bash
# From project root (llama/)
pip install -r requirements.txt
cp .env.example .env              # edit as needed
python app.py --web               # http://localhost:8000
# or: uvicorn api.main:app --reload
```

Interactive API docs: http://localhost:8000/docs

### Frontend
```bash
cd frontend
cp .env.example .env.local        # VITE_API_BASE_URL=/api  (uses Vite proxy)
npm install
npm run dev                       # http://localhost:5173
```

> Vite automatically proxies `/api/*` → `http://localhost:8000` in dev mode.

### CLI (unchanged)
```bash
python app.py data/input data/output 3
```

## Variable compatibility

| Original name      | Now lives in                         |
|--------------------|--------------------------------------|
| `PAGE_TEMPLATES`   | `configs/settings.py`                |
| `FONT_OPTIONS`     | `configs/settings.py` (property)     |
| `RUNS_DIR`         | `configs/settings.py`                |
| `preview_text`     | `configs/settings.py` (`PREVIEW_TEXT`) |
| `_update_job`      | `api/middleware/job_store.py` (alias)|
| `_get_job`         | `api/middleware/job_store.py` (alias)|
| `_safe_name`       | `api/services/job_service.py` (`safe_name`) |
| `_buffer_upload`   | `api/middleware/file_validator.py` (alias) |
