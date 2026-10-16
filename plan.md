# Plan — Detectron2 (fork `obaruch/detectron2`)

> **Artefacto 3 de 3 del runbook AI-Native SDLC** · `intent.md` → `specs.md` → **`plan.md`**
> Baseline: commit `9725c7f` · paquete `detectron2==0.6` · Estado: **Draft**
> Cada tarea `T-*` referencia los requisitos de `specs.md` que implementa y la puerta (`G-*`) que debe pasar.

---

## 1. Ciclo AI-native (runbook estándar)

Todo cambio en el repo sigue este ciclo. Ningún paso se salta.

| # | Etapa | Entrada | Salida | Responsable | Puerta |
|---|---|---|---|---|---|
| 1 | **Intent** | Necesidad / issue | Objetivo `INT-*` en `intent.md` (o referencia a uno existente) | Humano (agente puede redactar) | G-1 Intent aprobado |
| 2 | **Specs** | `INT-*` | `REQ-*`/`NFR-*`/`AC-*` en `specs.md` | Agente redacta, humano aprueba | G-2 Specs aprobadas |
| 3 | **Plan** | Specs | Tareas `T-*` con alcance, archivos, verificación | Agente | G-3 Plan revisado |
| 4 | **Implement** | `T-*` | Commits atómicos en la rama asignada | Agente | — |
| 5 | **Verify** | Diff | Lint + tests + (acc tests si aplica) en verde | Agente | G-4 Verificación local |
| 6 | **Review** | PR | Aprobación contra `AC-*` | Humano / bot de review | G-5 Review |
| 7 | **Release / Merge** | PR verde | Merge + changelog | Humano | G-6 CI verde y merge |
| 8 | **Learn** | Resultado | Actualización de intent/specs/plan (retro) | Agente + humano | — |

### 1.1 Puertas de calidad

| Puerta | Condición de paso |
|---|---|
| **G-1** | Objetivo trazable a `intent.md` §4; no contradice §5 (no-objetivos) |
| **G-2** | Requisitos con ID, fuente y criterio de aceptación; zonas sensibles (NFR-14) marcadas |
| **G-3** | Cada `T-*` lista archivos, `REQ-*`, comando de verificación y rollback |
| **G-4** | `dev/linter.sh` limpio · `pytest` del área verde · para cambios numéricos, `*_acc_test.yaml` dentro de tolerancia |
| **G-5** | Diff revisado adversarialmente; sin cambios fuera de alcance; plantilla de PR cumplida |
| **G-6** | CI verde en el head del PR; sin conflictos; aprobación humana |

## 2. Fases

```
PH-0 Baseline ──► PH-1 Gobernanza AI-native ──► PH-2 CI del fork ──► PH-3 Modernización toolchain ──► PH-4 Operación continua
   (hecho)            (este entregable)            (INT-03)              (INT-04)                         (INT-02/05/06)
```

---

### PH-0 — Baseline y descubrimiento  ✅

| ID | Tarea | Specs | Verificación | Estado |
|---|---|---|---|---|
| T-0.1 | Inventariar módulos, registries, tools, configs, tests y CI | §2 specs | Tabla §2 de `intent.md` y diagrama §2 de `specs.md` | ✅ |
| T-0.2 | Identificar toolchain y deriva (Python 3.7, torch 1.8–1.10, `actions@v2`, pins fvcore/iopath/black) | NFR-05, RISK-01/02/03/05 | `setup.py`, `.github/workflows/workflow.yml`, `.circleci/config.yml` | ✅ |
| T-0.3 | Registrar últimos cambios del baseline (`WarmupStepWithFixedGammaLR`, `SOLVER.NUM_DECAYS`, ONNX-only export) | REQ-05, REQ-10 | `git log`, `tests/test_scheduler.py` | ✅ |

### PH-1 — Gobernanza AI-native (este entregable)

| ID | Tarea | Specs | Archivos | Verificación | Estado |
|---|---|---|---|---|---|
| T-1.1 | Crear `intent.md` | AC-01 | `intent.md` | Revisión humana | ✅ |
| T-1.2 | Crear `specs.md` | AC-01 | `specs.md` | Revisión humana | ✅ |
| T-1.3 | Crear `plan.md` | AC-01 | `plan.md` | Revisión humana | ✅ |
| T-1.4 | Resolver preguntas abiertas Q-01…Q-04 de `intent.md` §10 | G-1 | `intent.md` | Decisiones registradas | ⏳ Humano |
| T-1.5 | (Opcional) Añadir `CLAUDE.md` que apunte a estos 3 artefactos y a los guardrails | NFR-11, NFR-14 | `CLAUDE.md` | Revisión humana | ⏳ Propuesta |

**Rollback PH-1:** borrar los 3 archivos (no hay cambios de código).

### PH-2 — CI operativa en el fork

| ID | Tarea | Specs | Archivos | Verificación | Rollback |
|---|---|---|---|---|---|
| T-2.1 | Quitar/ajustar la condición `repository_owner == 'facebookresearch'` para que push al fork ejecute lint y tests | NFR-13, AC-02 | `.github/workflows/workflow.yml` | Push a rama → job ejecutado | Revert del commit |
| T-2.2 | Actualizar `actions/checkout`, `setup-python`, `cache` a versiones soportadas | RISK-05 | idem | Workflow verde | Revert |
| T-2.3 | Añadir job Linux CPU de unit tests en GitHub Actions (hoy Linux corre en CircleCI) | AC-03, Q-03 | idem | `pytest -n 4 tests/` verde | Revert |
| T-2.4 | Decidir destino de CircleCI (GPU/Windows): mantener, migrar o retirar | Q-03 | `.circleci/config.yml` | Decisión registrada | — |

**Salida PH-2:** AC-02, AC-03, AC-04 cumplidos de forma automática.

### PH-3 — Modernización incremental de toolchain

> Precondición: Q-02 resuelta (versiones objetivo). Un PR por tarea. Cada tarea es reversible.

| ID | Tarea | Specs | Archivos | Verificación | Riesgo |
|---|---|---|---|---|---|
| T-3.1 | Definir y documentar matriz objetivo Python/PyTorch/torchvision | NFR-06, AC-06 | `specs.md`, `docs/tutorials/install.md` | Revisión humana | — |
| T-3.2 | Actualizar matriz de CI a la matriz objetivo manteniendo una versión mínima | NFR-05/06 | workflows | CI verde en toda la matriz | RISK-01 |
| T-3.3 | Compilar `detectron2._C` con la nueva toolchain (CPU y, si hay GPU, CUDA) | REQ-01 | `setup.py`, `layers/csrc/**` (sólo si es necesario, NFR-14) | `pip install -e .` + `collect_env` + `tests/layers/` | RISK-01 |
| T-3.4 | Revisar pins `fvcore`/`iopath` | RISK-02 | `setup.py` | `tests/test_scheduler.py`, `tests/test_registry.py`, `tests/test_checkpoint.py` | RISK-02 |
| T-3.5 | Evaluar mover `black` de `install_requires` a extra opcional con fallback en `LazyConfig.save` | RISK-03, NFR-07 | `setup.py`, `detectron2/config/lazy.py` | `tests/config/test_lazy_config.py` | RISK-03 |
| T-3.6 | Actualizar versiones de linter (isort/black/flake8) en **PR separado** de cualquier cambio funcional | NFR-01 | `dev/linter.sh`, `setup.cfg`, `.flake8`, workflows | Lint verde; diff sólo de formato | Diff masivo |
| T-3.7 | Actualizar `docs/notes/changelog.md` y `docs/notes/compatibility.md` | NFR-04 | docs | `make -C docs html` | — |

**Salida PH-3:** AC-06 cumplido; REQ-01…REQ-13 siguen en verde.

### PH-4 — Operación continua (runbook por cambio)

Aplicar el ciclo §1 a cada nueva necesidad. Plantilla de tarea:

```markdown
### T-<fase>.<n> — <título imperativo>
- Intent: INT-xx
- Specs: REQ-xx / NFR-xx / AC-xx
- Zona sensible (NFR-14): sí/no  → si sí, aprobación: <link>
- Archivos: <rutas>
- Pasos: 1… 2… 3…
- Verificación: <comandos exactos>
- Rollback: <revert / flag / config>
- Estado: ⏳ / 🚧 / ✅
```

## 3. Runbook operativo (comandos)

### 3.1 Setup local

```bash
python -m pip install -U pip
python -m pip install ninja opencv-python-headless onnx pytest pytest-xdist
python -m pip install torch torchvision            # versión según matriz (NFR-05/06)
python -m pip install -e '.[all]'                  # compila detectron2._C
python -m detectron2.utils.collect_env             # REQ-01
./datasets/prepare_for_tests.sh                    # datos mínimos para tests
```

### 3.2 Antes de cada commit (G-4)

```bash
dev/linter.sh                                      # isort + black -l 100 + flake8 + clang-format
# ó, en modo check (igual que CI):
isort -c -sp . && black -l 100 --check . && flake8 .
python -m pytest -sv tests/<área_afectada>
```

### 3.3 Antes de abrir/mergear PR

```bash
python -m pytest -n 4 --durations=15 -sv tests/    # AC-03
# Cambios en modeling/engine (GPU):
dev/run_instant_tests.sh
# Cambios numéricos (GPU + datasets, NFR-09 / AC-08):
dev/run_inference_tests.sh
```

### 3.4 Mapa área → tests

| Área cambiada | Tests mínimos |
|---|---|
| `config/` | `tests/config/`, `tests/test_model_zoo.py` |
| `data/` | `tests/data/` |
| `layers/` (+csrc) | `tests/layers/` |
| `modeling/` | `tests/modeling/`, `tests/test_model_analysis.py` |
| `structures/` | `tests/structures/` |
| `solver/` | `tests/test_solver.py`, `tests/test_scheduler.py` |
| `engine/`, `checkpoint/` | `tests/test_engine.py`, `tests/test_checkpoint.py`, `tests/test_events.py` |
| `export/` | `tests/test_export_torchscript.py`, `tests/test_export_caffe2.py`, `tests/test_export_onnx.py`, `tests/export/` |
| `tracking/` | `tests/tracking/` |
| `utils/visualizer.py` | `tests/test_visualizer.py` |
| `setup.py`, `projects/` | `tests/test_packaging.py`, `.circleci/import-tests.sh` |

### 3.5 Convención de commits y PR (NFR-11, NFR-15)

```
<área>: <resumen imperativo>

Refs: T-2.1 · NFR-13 · AC-02
```

PR: seguir `.github/pull_request_template.md`; abrir issue previo si > 100 líneas; incluir sección
"Verificación" con los comandos ejecutados y su resultado.

### 3.6 Rollback

- Cambios de código: `git revert <sha>` en una rama nueva + PR (nunca force-push en ramas compartidas).
- Cambios de config/defaults: nunca eliminar claves (CON-02); revertir el default.
- Toolchain: cada tarea PH-3 va en PR aislado → revert independiente.

## 4. Matriz de trazabilidad

| Intent | Specs | Tareas | Aceptación |
|---|---|---|---|
| INT-01 | NFR-11, NFR-15 | T-1.1…T-1.5 | AC-01, AC-07 |
| INT-02 | REQ-01…REQ-13, CON-01…06, NFR-01…05, NFR-09/10 | PH-4 (cada cambio), T-3.3, T-3.4 | AC-03, AC-04, AC-05, AC-08 |
| INT-03 | NFR-13 | T-2.1…T-2.4 | AC-02 |
| INT-04 | NFR-06, NFR-07, RISK-01/02/03/05 | T-3.1…T-3.7 | AC-06 |
| INT-05 | NFR-11, NFR-12, NFR-14 | T-1.5, §3.2–3.5 | AC-07 |
| INT-06 | REQ-03, REQ-12, NFR-04, NFR-08 | PH-4 | AC-05 |

## 5. Estado y próximos pasos

| Hecho | Pendiente (requiere decisión humana) |
|---|---|
| PH-0 completa; PH-1 T-1.1…T-1.3 entregadas (sólo documentación, sin cambios de código) | Q-01 sync con upstream · Q-02 versiones objetivo · Q-03 CircleCI · Q-04 acceso GPU · aprobar inicio de PH-2 |
