# Specs — Detectron2 (fork `obaruch/detectron2`)

> **Artefacto 2 de 3 del runbook AI-Native SDLC** · `intent.md` → **`specs.md`** → `plan.md`
> Baseline: commit `9725c7f` · paquete `detectron2==0.6` · Estado: **Draft**
> Cada requisito referencia el objetivo (`INT-*`) que satisface. Cada tarea de `plan.md` referencia los requisitos (`REQ-*`, `NFR-*`) que implementa.

---

## 1. Convenciones

| Prefijo | Significado |
|---|---|
| `REQ-*` | Requisito funcional (comportamiento que el sistema **debe** tener / conservar) |
| `NFR-*` | Requisito no funcional (calidad, rendimiento, compatibilidad, proceso) |
| `CON-*` | Contrato de interfaz pública que no se puede romper sin decisión explícita |
| `AC-*` | Criterio de aceptación verificable |
| `RISK-*` | Riesgo identificado |

Palabras clave: **DEBE** (obligatorio), **DEBERÍA** (recomendado), **PUEDE** (opcional).

## 2. Arquitectura del sistema (as-is)

```
                ┌───────────────────────────── tools/ ─────────────────────────────┐
                │ train_net.py · lazyconfig_train_net.py · plain_train_net.py       │
                │ benchmark.py · analyze_model.py · visualize_*.py · deploy/        │
                └──────────────────────────────┬────────────────────────────────────┘
                                               │
     ┌──────────── config/ ────────────┐       ▼       ┌────────── engine/ ──────────┐
     │ YACS: get_cfg(), defaults.py    │──►  build  ──►│ TrainerBase → SimpleTrainer │
     │ Lazy: LazyConfig, LazyCall,     │               │ → AMPTrainer; DefaultTrainer│
     │       instantiate()             │               │ hooks.py · launch.py        │
     └─────────────────────────────────┘               └──────┬───────────┬──────────┘
                                                              │           │
   ┌────────── data/ ──────────┐   ┌────────── modeling/ ─────▼───┐   ┌───▼── solver/ ──┐
   │ DatasetCatalog/Metadata   │   │ META_ARCH_REGISTRY           │   │ build_optimizer │
   │ DatasetMapper, transforms │──►│  backbone → proposal_gen →   │   │ build_lr_sched. │
   │ samplers, build_*_loader  │   │  roi_heads / dense heads     │   └─────────────────┘
   └───────────────────────────┘   └──────────────┬───────────────┘
                                                  │ usa
          ┌── structures/ ──┐  ┌── layers/ (+csrc C++/CUDA → detectron2._C) ──┐
          │ Boxes, Instances│  │ nms, roi_align(_rotated), deform_conv, losses │
          └─────────────────┘  └───────────────────────────────────────────────┘
   evaluation/ (COCO, LVIS, ...) · checkpoint/ · export/ (TorchScript, Caffe2, ONNX)
   tracking/ · utils/ (registry, comm, events, logger, visualizer) · model_zoo/
   projects/ (PointRend, DeepLab, Panoptic-DeepLab instalados como detectron2.projects.*)
```

### 2.1 Mecanismos de extensión

- **Registries** (`fvcore.common.registry.Registry`): `META_ARCH`, `BACKBONE`, `PROPOSAL_GENERATOR`,
  `RPN_HEAD`, `ANCHOR_GENERATOR`, `ROI_HEADS`, `ROI_BOX_HEAD`, `ROI_MASK_HEAD`, `ROI_KEYPOINT_HEAD`,
  `SEM_SEG_HEADS`, `TRACKER_HEADS`.
- **`@configurable`** (`config/config.py`): permite construir módulos desde `cfg` o desde kwargs.
- **LazyConfig**: configs en Python (`configs/common/`, `configs/new_baselines/`) instanciadas con `instantiate()`.
- **Hooks** del trainer (`HookBase`): `before_train/after_train/before_step/after_step`.

## 3. Requisitos funcionales (preservación del comportamiento)

| ID | Requisito | Fuente | INT |
|---|---|---|---|
| **REQ-01** | El paquete DEBE instalarse con `pip install -e .` compilando `detectron2._C` (CPU; CUDA si disponible o `FORCE_CUDA=1`). | `setup.py` | 02 |
| **REQ-02** | `get_cfg()` DEBE devolver los defaults de `config/defaults.py`; los YAML con `_BASE_` DEBEN fusionarse; `upgrade_config`/`downgrade_config` DEBEN mantener compatibilidad por `_C.VERSION`. | `config/` | 02 |
| **REQ-03** | `LazyConfig.load/save` e `instantiate` DEBEN funcionar sobre los configs de `configs/common` y `configs/new_baselines`. | `config/lazy.py` | 02, 06 |
| **REQ-04** | Todas las meta-arquitecturas registradas (R-CNN, RetinaNet, FCOS, PanopticFPN, SemanticSegmentor) DEBEN construirse vía `build_model(cfg)` y ejecutar `forward` en modo train y eval. | `modeling/` | 02 |
| **REQ-05** | Los schedulers `WarmupMultiStepLR`, `WarmupCosineLR`, `WarmupStepWithFixedGammaLR` DEBEN seleccionarse por `SOLVER.LR_SCHEDULER_NAME`; nombres desconocidos DEBEN lanzar `ValueError`. | `solver/build.py` | 02 |
| **REQ-06** | `DefaultTrainer`/`SimpleTrainer`/`AMPTrainer` DEBEN entrenar, guardar y reanudar checkpoints (`DetectionCheckpointer`, `resume_or_load`). | `engine/`, `checkpoint/` | 02 |
| **REQ-07** | `launch()` DEBE soportar single-process, multi-GPU y multi-nodo (`--num-machines`, `--dist-url`). | `engine/launch.py` | 02 |
| **REQ-08** | Los datasets builtin DEBEN registrarse al importar `detectron2.data` y resolverse bajo `DETECTRON2_DATASETS`. | `data/datasets/builtin.py` | 02 |
| **REQ-09** | Los evaluadores (COCO, LVIS, Cityscapes, VOC, Panoptic, SemSeg) DEBEN producir `OrderedDict` de métricas; `verify_results` DEBE validar `TEST.EXPECTED_RESULTS`. | `evaluation/` | 02 |
| **REQ-10** | El export a TorchScript (scripting y tracing), Caffe2 y ONNX DEBE funcionar para los modelos cubiertos por `tests/test_export_*.py`. | `export/` | 02 |
| **REQ-11** | `model_zoo.get(config_path, trained)` DEBE resolver configs empaquetados y URLs de checkpoints. | `model_zoo/` | 02 |
| **REQ-12** | Los proyectos `point_rend`, `deeplab`, `panoptic_deeplab` DEBEN ser importables como `detectron2.projects.*` tras la instalación. | `setup.py:PROJECTS` | 06 |
| **REQ-13** | Los trackers registrados en `TRACKER_HEADS` DEBEN mantener su API (`update(instances)`). | `tracking/` | 02 |

## 4. Contratos públicos (no romper)

| ID | Contrato |
|---|---|
| **CON-01** | Símbolos exportados en los `__init__.py` de cada subpaquete (`__all__`). |
| **CON-02** | Claves de `config/defaults.py`: no se eliminan ni renombran; nuevas claves DEBEN tener default que preserve el comportamiento previo. |
| **CON-03** | Formato de entrada/salida de modelos: `list[dict]` con `image`, `instances`, `sem_seg`, `panoptic_seg`, etc. (`docs/tutorials/models.md`). |
| **CON-04** | Formato de datasets estándar de detectron2 (`docs/tutorials/datasets.md`). |
| **CON-05** | Formato de checkpoint y conversión de pesos Caffe2/Detectron1 (`checkpoint/c2_model_loading.py`). |
| **CON-06** | CLI de `tools/train_net.py` y `default_argument_parser()`. |

## 5. Requisitos no funcionales

### 5.1 Calidad y estilo

| ID | Requisito | INT |
|---|---|---|
| **NFR-01** | El código DEBE pasar `isort -c -sp .` (isort 4.3.21), `black -l 100 --check .` (black 22.3.0) y `flake8 .` con `.flake8`/`setup.cfg`. | 02, 05 |
| **NFR-02** | C++/CUDA DEBE formatearse con `clang-format` (`.clang-format`). | 02 |
| **NFR-03** | Todo cambio funcional DEBE incluir o actualizar tests en `tests/` del área afectada. | 02, 05 |
| **NFR-04** | Cambios de API pública DEBEN reflejarse en `docs/` y en `docs/notes/changelog.md` cuando aplique. | 06 |

### 5.2 Compatibilidad

| ID | Requisito | INT |
|---|---|---|
| **NFR-05** | Baseline soportado: Python ≥ 3.7, PyTorch ≥ 1.8 (`setup.py`), matriz CI macOS torch 1.8/1.9/1.10. | 02 |
| **NFR-06** | La modernización (INT-04) DEBERÍA apuntar a versiones de Python y PyTorch con soporte vigente, definidas en una decisión registrada (ver Q-02) antes de implementarse. | 04 |
| **NFR-07** | `install_requires` NO DEBE incorporar dependencias no puras de Python (política de `setup.py`). | 02 |
| **NFR-08** | Las nuevas dependencias de import DEBEN añadirse a `docs/requirements.txt` o mockearse en `docs/conf.py`. | 06 |

### 5.3 Rendimiento y exactitud

| ID | Requisito | INT |
|---|---|---|
| **NFR-09** | Cambios en `layers/`, `modeling/`, `data/transforms/` o `solver/` NO DEBEN mover métricas AP fuera de la tolerancia de `TEST.EXPECTED_RESULTS` de `configs/quick_schedules/*_acc_test.yaml`. | 02 |
| **NFR-10** | No DEBEN introducirse regresiones de velocidad de entrenamiento/inferencia relevantes (referencia: `docs/notes/benchmarks.md`, `tools/benchmark.py`). | 02 |

### 5.4 Proceso AI-native

| ID | Requisito | INT |
|---|---|---|
| **NFR-11** | Cada commit/PR DEBE citar el `T-*` de `plan.md` y los `REQ-*`/`NFR-*` que toca. | 01, 05 |
| **NFR-12** | Ningún push sin lint y tests afectados en verde localmente. | 05 |
| **NFR-13** | CI DEBE ejecutarse en pushes y PRs del fork (hoy el job está condicionado a `repository_owner == 'facebookresearch'` en push). | 03 |
| **NFR-14** | Cambios en zonas sensibles (`layers/csrc/`, `config/defaults.py`, `export/`, firmas públicas) DEBEN tener aprobación humana previa registrada en el issue/PR. | 05 |
| **NFR-15** | Los PRs DEBEN seguir `.github/pull_request_template.md` (issue previo para PRs > 100 líneas; `dev/linter.sh` antes de enviar). | 01 |

## 6. Estrategia de verificación

| Nivel | Comando | Cuándo | Cubre |
|---|---|---|---|
| Lint | `dev/linter.sh` · o `isort -c -sp . && black -l 100 --check . && flake8 .` | Cada commit | NFR-01, NFR-02 |
| Env | `python -m detectron2.utils.collect_env` | Tras instalar | REQ-01 |
| Unit (CPU) | `python -m pytest -n 4 --durations=15 -sv tests/` (requiere `./datasets/prepare_for_tests.sh`) | Cada PR | REQ-02…REQ-13 |
| Unit dirigido | `pytest tests/<área>` (ej. `tests/test_scheduler.py`, `tests/modeling/`) | Cada cambio | Área tocada |
| Instant (GPU) | `dev/run_instant_tests.sh` (`configs/quick_schedules/*instant_test.yaml`) | Cambios en modeling/engine | REQ-04, REQ-06 |
| Inference acc (GPU) | `dev/run_inference_tests.sh` (`*inference_acc_test.yaml`) | Cambios numéricos | NFR-09 |
| Training acc (GPU) | `tools/train_net.py --config-file configs/quick_schedules/*training_acc_test.yaml` | Cambios en solver/loss | NFR-09 |
| Packaging | `tests/test_packaging.py`, `.circleci/import-tests.sh` | Cambios en setup/projects | REQ-01, REQ-12 |
| Docs | `make -C docs html` | Cambios de API/doc | NFR-04, NFR-08 |

## 7. Criterios de aceptación

| ID | Criterio | Verifica |
|---|---|---|
| **AC-01** | Existen `intent.md`, `specs.md`, `plan.md` en la raíz con IDs trazables entre sí. | INT-01 |
| **AC-02** | Un push a cualquier rama del fork dispara lint y unit tests y reporta estado. | NFR-13 |
| **AC-03** | `pytest tests/` pasa en CPU en el baseline y tras cada cambio. | REQ-02…REQ-13 |
| **AC-04** | Lint (`isort`/`black`/`flake8`) pasa sin cambios pendientes. | NFR-01 |
| **AC-05** | Ningún PR elimina/renombra símbolos de `__all__` ni claves de `defaults.py` sin nota en changelog y aprobación. | CON-01, CON-02 |
| **AC-06** | La matriz de versiones objetivo (Python/PyTorch) está documentada y la CI la ejecuta en verde. | NFR-06 |
| **AC-07** | Todo PR cita `T-*` y `REQ-*`/`NFR-*`. | NFR-11 |
| **AC-08** | Para cambios numéricos, se adjuntan resultados de `*_acc_test.yaml` dentro de tolerancia. | NFR-09 |

## 8. Riesgos

| ID | Riesgo | Impacto | Mitigación |
|---|---|---|---|
| **RISK-01** | Actualizar PyTorch rompe la compilación de `layers/csrc` (CUDA/ROCm) | Alto | Cambios de toolchain en PR aislado; compilar CPU y CUDA; `FORCE_CUDA=1` en CI GPU |
| **RISK-02** | `fvcore`/`iopath` fijados (`<0.1.6`, `<0.1.10`) chocan con versiones nuevas de dependencias | Medio | Actualizar en paso separado; `StepWithFixedGammaParamScheduler` y `Registry` vienen de fvcore — cubrir con `tests/test_scheduler.py`, `tests/test_registry.py` |
| **RISK-03** | `black==22.3.0` en `install_requires` fuerza conflicto de versiones en entornos de usuario | Medio | Evaluar moverlo a extra `dev` (requiere decisión: LazyConfig lo usa para formatear al guardar) |
| **RISK-04** | Deriva numérica silenciosa (AP) sin GPU en CI | Alto | NFR-09 + ejecución manual de acc tests antes de merge de cambios numéricos |
| **RISK-05** | Acciones `@v2` y runners Python 3.7 retirados en GitHub Actions | Alto | Fase de modernización de CI (PH-2) |
| **RISK-06** | Agente de IA aplica cambios fuera de alcance | Medio | Guardrails de `intent.md` §9, NFR-11/14, revisión humana |
| **RISK-07** | Divergencia con upstream dificulta sincronizar | Medio | Commits atómicos y trazables; registrar decisión Q-01 |

## 9. Fuera de alcance de estas specs

Nuevos modelos o datasets, cambios de defaults que alteren resultados publicados, rediseño de APIs,
re-entrenamiento del Model Zoo. Cualquiera de ellos requiere un nuevo ciclo `intent → specs → plan`.
