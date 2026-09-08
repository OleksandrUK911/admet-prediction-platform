# TODO — ADMET Prediction Platform

## Опис проєкту

Дослідницька/освітня платформа для прогнозування ADMET-профілю молекули
(Absorption, Distribution, Metabolism, Excretion, Toxicity) на основі
multi-task ML-моделей. Демонструє роботу з класовим дисбалансом, каліброваними
ймовірностями, оцінкою невизначеності (uncertainty) та applicability domain —
теми, критичні для реального drug discovery.

**Важливо:** результати подаються як research/educational prediction, а не як
медичний висновок — з явним відображенням довірчого інтервалу/невизначеності.

**Мета:** показати вміння працювати з реалістичними, "брудними" біомедичними
даними та коректно комунікувати межі надійності моделі.

**Статус:** 🔴 Не розпочато (черга після проєкту №1)

## Мовна політика

Продукт (UI, README, код, коментарі, API-документація, дисклеймери) —
англійською мовою як основною. Українська — друга мова (i18n locale),
опційна. TODO-плани залишаються українською для зручності автора.

**[ROADMAP.md](ROADMAP.md)** — послідовний план з 7 спринтів (~1 тиждень кожен) від даних до деплою.

## Категорії

## Data
- [Джерела даних та ліцензування](data/TODO_sources_licensing.md)
- [Пайплайн препроцесингу](data/TODO_preprocessing_pipeline.md)
- [Якість даних та валідація](data/TODO_quality_validation.md)
- [Версіонування та відтворюваність](data/TODO_versioning_reproducibility.md)

## ML
- [Baseline](ml/TODO_baseline.md)
- [Feature Engineering](ml/TODO_feature_engineering.md)
- [Експерименти: окремі моделі на задачу](ml/TODO_experiments_per_task_models.md)
- [Експерименти: multi-task архітектура](ml/TODO_experiments_multitask.md)
- [Трекінг експериментів та вибір переможця](ml/TODO_experiments_tracking.md)
- [Оцінка та валідація](ml/TODO_evaluation_validation.md)
- [Калібрування та невизначеність](ml/TODO_calibration_uncertainty.md)
- [Model Registry та документація моделей](ml/TODO_model_registry.md)

## Backend
- [API Design](backend/TODO_api_design.md)
- [База даних](backend/TODO_database.md)
- [Автентифікація та безпека](backend/TODO_auth_security.md)
- [Тестування Backend](backend/TODO_testing.md)
- [Логування та спостережуваність](backend/TODO_logging_observability.md)

## Frontend
- [UI-компоненти](frontend/TODO_ui_components.md)
- [Сторінки та флоу](frontend/TODO_pages_flows.md)
- [Стан та шар даних](frontend/TODO_state_data_layer.md)
- [Доступність та адаптивність](frontend/TODO_accessibility_responsive.md)
- [Тестування Frontend](frontend/TODO_testing.md)
- [Layout дашборду](frontend/TODO_dashboard_layout.md)
- [Візуалізація даних](frontend/TODO_data_visualization.md)
- [Інтернаціоналізація та локалізація](frontend/TODO_i18n_localization.md)

## DevOps
- [Контейнеризація](devops/TODO_containerization.md)
- [CI/CD](devops/TODO_cicd.md)
- [Інфраструктура та деплой](devops/TODO_infrastructure_deployment.md)
- [Моніторинг та спостережуваність](devops/TODO_monitoring_observability.md)
- [Безпека та відповідність](devops/TODO_security_compliance.md)

## Загальний прогрес

| Категорія | Виконано | Всього |
|---|---|---|
| Data | 0 | 29 |
| ML | 0 | 62 |
| Backend | 0 | 35 |
| Frontend | 0 | 54 |
| DevOps | 0 | 30 |
| **Разом** | **0** | **210** |

### Примітки
- Старт цього проєкту — тільки після живого деплою проєкту №1 (Molecular Property Prediction).
- Використовує напрацювання з проєкту №1 (RDKit-пайплайн, базовий FastAPI-каркас) як фундамент.
- Наскрізна вимога, що повторюється у ML/Backend/Frontend/DevOps TODO: чітка комунікація "не для клінічного використання" на кожному рівні системи (API-відповідь, UI, документація, деплой).
