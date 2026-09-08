# ROADMAP — ADMET Prediction Platform

Послідовність із 7 спринтів (~1 тиждень кожен, соло-розробка в неповний час,
з асистенцією Claude Code). Спринти йдуть у зазначеному порядку — кожен
наступний спирається на артефакти попереднього (дані → моделі → API → UI →
деплой → фінальна полірування). У межах спринту спершу закривати **P0**,
потім **P1**, **P2** — за наявності часу/бажання.

## Спринт 1 — Дані та інфраструктура датасету
**Мета:** отримати єдину, відтворювану, версійовану multi-task таблицю
(SMILES → мітки ADMET) з перевіреною ліцензією джерел.
- [Джерела даних та ліцензування](data/TODO_sources_licensing.md) — P0-задачі
- [Пайплайн препроцесингу](data/TODO_preprocessing_pipeline.md) — P0-задачі
- [Якість даних та валідація](data/TODO_quality_validation.md) — P0 (дублікати, leakage, scaffold split)
- [Версіонування та відтворюваність](data/TODO_versioning_reproducibility.md) — P0 (reproducible-скрипт, seed)
**Тривалість:** ~1 тиждень.

## Спринт 2 — Baseline та експерименти з архітектурою
**Мета:** мати задокументований baseline і чесне порівняння per-task vs
multi-task підходів на реальних метриках.
- [Baseline](ml/TODO_baseline.md) — P0
- [Feature Engineering](ml/TODO_feature_engineering.md) — P0
- [Експерименти: окремі моделі на задачу](ml/TODO_experiments_per_task_models.md) — P0
- [Експерименти: multi-task архітектура](ml/TODO_experiments_multitask.md) — P0
- [Трекінг експериментів та вибір переможця](ml/TODO_experiments_tracking.md) — P0
**Тривалість:** ~1-1.5 тижні (найбільш дослідницький і непередбачуваний спринт).

## Спринт 3 — Калібрування, невизначеність та фінальна оцінка моделі
**Мета:** зафіксувати фінальну модель разом з каліброваними ймовірностями,
uncertainty-оцінкою, applicability domain та model card.
- [Калібрування та невизначеність](ml/TODO_calibration_uncertainty.md) — P0
- [Оцінка та валідація](ml/TODO_evaluation_validation.md) — P0
- [Model Registry та документація моделей](ml/TODO_model_registry.md) — P0
**Тривалість:** ~1 тиждень.

## Спринт 4 — Backend API
**Мета:** робочий FastAPI-сервіс, що віддає ADMET-профіль з uncertainty,
applicability domain та явним дисклеймером, покритий тестами.
- [API Design](backend/TODO_api_design.md) — P0
- [База даних](backend/TODO_database.md) — P0 (схема, міграції)
- [Автентифікація та безпека](backend/TODO_auth_security.md) — P0 (валідація вводу, дисклеймер-заходи)
- [Тестування Backend](backend/TODO_testing.md) — P0
- [Логування та спостережуваність](backend/TODO_logging_observability.md) — P1
**Тривалість:** ~1 тиждень.

## Спринт 5 — Frontend Dashboard
**Мета:** зрозумілий UI, що явно комунікує невизначеність та обмеження
моделі, а не просто "чисті" відсотки.
- [UI-компоненти](frontend/TODO_ui_components.md) — P0
- [Сторінки та флоу](frontend/TODO_pages_flows.md) — P0 (головна сторінка + "Про проєкт")
- [Стан та шар даних](frontend/TODO_state_data_layer.md) — P0
- [Тестування Frontend](frontend/TODO_testing.md) — P0 (smoke test)
- [Доступність та адаптивність](frontend/TODO_accessibility_responsive.md) — P1
**Тривалість:** ~1 тиждень.

## Спринт 6 — DevOps та деплой
**Мета:** контейнеризований застосунок з CI/CD-gate'ами (тести + ML-регресія)
та живим публічним demo.
- [Контейнеризація](devops/TODO_containerization.md) — P0
- [CI/CD](devops/TODO_cicd.md) — P0
- [Інфраструктура та деплой](devops/TODO_infrastructure_deployment.md) — P0
- [Моніторинг та спостережуваність](devops/TODO_monitoring_observability.md) — P1
- [Безпека та відповідність](devops/TODO_security_compliance.md) — P0
**Тривалість:** ~1 тиждень.

## Спринт 7 — Полірування та model cards
**Мета:** довести проєкт до "портфоліо-якості" — довершені model cards,
README, і всі P1/P2 задачі, що покращують враження, але не блокують запуск.
- Залишкові P1/P2 з усіх файлів `ml/`, `frontend/`, `devops/` (порівняння
  молекул, dark theme, PDF-експорт, дашборд моніторингу тощо)
- Фінальна вичитка всіх model cards ([ml/TODO_model_registry.md](ml/TODO_model_registry.md))
  та наскрізного дисклеймера "not for clinical use" на всіх рівнях системи
**Тривалість:** ~0.5-1 тиждень.

### Примітки
- Спринт 2 — найризикованіший за часом: якщо multi-task експерименти
  затягуються, можна тимчасово зафіксувати per-task моделі як робочу
  проміжну версію і повернутись до multi-task пізніше без блокування
  Спринту 3-4.
- Загальна оцінка: ~6.5-7.5 тижнів соло-розробки в неповний час.
