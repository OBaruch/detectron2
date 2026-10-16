# Intent — Detectron2 (fork `obaruch/detectron2`)

> **Artefacto 1 de 3 del runbook AI-Native SDLC** · `intent.md` → `specs.md` → `plan.md`
> Baseline: commit `9725c7f` (2022-09-09, "Add cfg.SOLVER.NUM_DECAYS in the default config") · versión del paquete `0.6` (`detectron2/__init__.py`)
> Estado: **Draft** · Owner: mantenedor del fork · Última revisión: 2026-09-29

---

## 1. Propósito del documento

Este archivo define **el porqué y el qué** del proyecto. Es la fuente de verdad que cualquier agente de IA
(Claude Code, Copilot, etc.) o persona debe leer **antes** de proponer cambios. Si una tarea no se puede
trazar a un objetivo de este documento, no se ejecuta sin aprobación humana explícita.

Cadena de trazabilidad del runbook:

```
Intent (INT-*)  ──►  Specs (REQ-* / NFR-* / AC-*)  ──►  Plan (PH-* / T-*)  ──►  Commit / PR  ──►  Verificación
```

## 2. Qué es el proyecto

Detectron2 es la librería de FAIR (Meta AI) para **detección y segmentación de objetos** sobre PyTorch.
Sucesora de Detectron y maskrcnn-benchmark. En este repositorio se mantiene como fork.

Capacidades presentes en el código actual:

| Área | Evidencia en el repo |
|---|---|
| Meta-arquitecturas | `modeling/meta_arch/`: `GeneralizedRCNN`, `RetinaNet`, `FCOS`, `PanopticFPN`, `SemanticSegmentor`, `DenseDetector` |
| Backbones | `modeling/backbone/`: ResNet, FPN, RegNet, ViT, MViT, Swin |
| Heads | `modeling/roi_heads/`: box, mask, keypoint, cascade, rotated; `proposal_generator/`: RPN, RRPN |
| Estructuras | `structures/`: `Boxes`, `RotatedBoxes`, `Instances`, `ImageList`, `BitMasks`/`PolygonMasks`, `Keypoints` |
| Datos | `data/`: catálogos, mappers, augmentations, samplers; datasets COCO, LVIS, Cityscapes, Pascal VOC, COCO-Panoptic |
| Entrenamiento | `engine/`: `SimpleTrainer`, `AMPTrainer`, `DefaultTrainer`, hooks, `launch` multi-GPU/multi-nodo |
| Solver | `solver/`: optimizadores, `WarmupMultiStepLR`, `WarmupCosineLR`, `WarmupStepWithFixedGammaLR` |
| Configuración | `config/`: YACS (`defaults.py`) + LazyConfig (`lazy.py`, `instantiate.py`) |
| Evaluación | `evaluation/`: COCO, LVIS, Cityscapes, Pascal VOC, Panoptic, SemSeg, Rotated COCO |
| Export / deploy | `export/`: TorchScript (scripting/tracing), Caffe2, ONNX; `tools/deploy/` (C++) |
| Tracking | `tracking/`: IoU / Hungarian trackers |
| Ops nativas | `layers/csrc/`: ROIAlignRotated, NMS rotado, deformable conv, box IoU rotado, cocoeval (C++/CUDA) |
| Proyectos | `projects/`: DensePose, PointRend, DeepLab, Panoptic-DeepLab, TensorMask, TridentNet, ViTDet, MViTv2, PointSup, Rethinking-BatchNorm |
| Model Zoo | `configs/` (+ `MODEL_ZOO.md`), `model_zoo/` API |

## 3. Problema / oportunidad

1. **El fork no tiene un proceso de desarrollo explícito.** Las contribuciones dependen de conocimiento tácito
   (`CONTRIBUTING.md`, `dev/linter.sh`, CircleCI) y no hay artefactos que un agente de IA pueda consumir para
   actuar de forma segura y trazable.
2. **Deriva de la toolchain.** El baseline fija Python 3.7 en CI, PyTorch 1.8–1.10 en macOS, acciones
   `actions/*@v2`, `black==22.3.0` como dependencia de runtime y `fvcore<0.1.6`/`iopath<0.1.10`. Varias de
   estas piezas están fuera de soporte a la fecha de este documento.
3. **CI condicionada al owner original.** Los jobs de `.github/workflows/workflow.yml` sólo corren si
   `github.repository_owner == 'facebookresearch'` o en `pull_request`; en pushes al fork no hay feedback.
4. **Riesgo de regresión alto.** El código es numérico (pérdidas, NMS, ROIAlign, schedulers). Un cambio
   "pequeño" puede alterar métricas AP sin romper ningún test unitario.

## 4. Objetivos (Intent statements)

| ID | Objetivo | Resultado esperado |
|---|---|---|
| **INT-01** | Establecer un SDLC AI-native reproducible para el fork | `intent.md`, `specs.md`, `plan.md` versionados y usados como puerta de entrada de todo cambio |
| **INT-02** | Preservar la corrección funcional y numérica de detectron2 | Ningún cambio altera APIs públicas ni métricas del Model Zoo sin decisión explícita |
| **INT-03** | Tener feedback de calidad automático en el fork | Lint + tests unitarios CPU corren en cada push/PR del fork |
| **INT-04** | Modernizar la toolchain de forma incremental y reversible | Soporte de versiones actuales de Python/PyTorch sin romper compatibilidad documentada |
| **INT-05** | Hacer que los agentes de IA trabajen con guardrails | Reglas claras de alcance, validación previa al push, trazabilidad commit → requisito |
| **INT-06** | Mantener la extensibilidad como librería | Registries, LazyConfig y `projects/` siguen siendo el mecanismo de extensión |

## 5. No-objetivos (explícitos)

- **No** reescribir arquitecturas ni cambiar defaults de `config/defaults.py` que afecten resultados publicados.
- **No** re-entrenar ni re-publicar el Model Zoo.
- **No** eliminar backends de export (Caffe2/ONNX/TorchScript) sin una spec aprobada.
- **No** introducir dependencias no puras de Python en `install_requires` (política explícita en `setup.py`).
- **No** mezclar cambios de formato masivo con cambios funcionales en el mismo PR.
- **No** hacer cambios de código como parte de la creación de este runbook (esta entrega es sólo documentación).

## 6. Stakeholders y roles

| Rol | Responsabilidad en el SDLC |
|---|---|
| **Owner del fork** (humano) | Aprueba intent/specs, decide trade-offs, aprueba y hace merge de PRs |
| **Agente de IA** (ej. Claude Code) | Propone specs/planes, implementa tareas `T-*`, ejecuta validaciones locales, abre PRs cuando se le pide |
| **Revisor** (humano o bot) | Revisa diffs contra `specs.md`; verifica criterios de aceptación |
| **Usuarios de la librería** | Investigadores/ingenieros que importan `detectron2` o usan `tools/*.py` |
| **Upstream** (`facebookresearch/detectron2`) | Fuente de cambios a sincronizar; referencia de comportamiento |

## 7. Restricciones

- **Técnicas:** PyTorch ≥ 1.8 (assert en `setup.py`), Python ≥ 3.7 (`python_requires`), extensión C++/CUDA
  `detectron2._C` compilada en instalación, estilo `black -l 100` + `isort 4.3.21` + `flake8`.
- **Hardware:** los tests de CI son CPU; entrenamientos/`*_acc_test.yaml` requieren GPU y datasets.
- **Licencia:** Apache-2.0 (`LICENSE`); cabeceras de copyright se preservan.
- **Datos:** los datasets no están en el repo; se resuelven vía `DETECTRON2_DATASETS`.

## 8. Métricas de éxito

| Métrica | Objetivo |
|---|---|
| PRs con trazabilidad `T-*` → `REQ-*` | 100 % |
| CI (lint + unit tests CPU) verde en `main` del fork | Siempre |
| Regresiones de API pública detectadas post-merge | 0 |
| Delta de AP en `configs/quick_schedules/*_inference_acc_test.yaml` (cuando se ejecuten) | Dentro de la tolerancia definida en cada YAML (`TEST.EXPECTED_RESULTS`) |
| Tiempo desde intent aprobado hasta PR verde | Medido, sin objetivo fijo en fase 0 |

## 9. Guardrails para agentes de IA

1. Leer `intent.md` → `specs.md` → `plan.md` antes de cualquier cambio; citar los IDs en commits/PRs.
2. Trabajar sólo en la rama asignada; nunca reescribir historia ajena ni forzar push a ramas compartidas.
3. Antes de push: `dev/linter.sh` (o equivalente `isort -c` / `black --check -l 100` / `flake8`) y los tests
   afectados (`pytest tests/<área>`).
4. Cambios en `layers/csrc/`, `config/defaults.py`, `export/` o firmas públicas ⇒ **requieren aprobación humana previa**.
5. No saltar, deshabilitar ni poner en cuarentena tests para conseguir verde.
6. Ante ambigüedad: proponer en el PR/issue, no decidir en silencio.

## 10. Supuestos y preguntas abiertas

| ID | Supuesto / Pregunta | Estado |
|---|---|---|
| Q-01 | ¿El fork seguirá sincronizándose con upstream o divergirá? | Abierta |
| Q-02 | ¿Versiones objetivo de Python/PyTorch para la modernización (INT-04)? | Abierta — propuesta en `specs.md` |
| Q-03 | ¿Se mantiene CircleCI (GPU/Windows) o se consolida todo en GitHub Actions? | Abierta |
| Q-04 | ¿Hay acceso a GPU para ejecutar `dev/run_inference_tests.sh`? | Abierta |
