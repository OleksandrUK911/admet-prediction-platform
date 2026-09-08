# TODO — Baseline (ADMET Prediction)

## Прості baseline-моделі
- [ ] Baseline "мажоритарний клас" для кожної toxicity-задачі (нижня межа для порівняння) — **P0** | ~1h
- [ ] Логістична регресія на Morgan fingerprints per-task (простий, швидкий baseline) — **P0** | ~2h
- [ ] Random Forest per-task baseline — **P0** | ~2h
- [ ] Зведена таблиця baseline-метрик per-task (ROC-AUC, PR-AUC) як точка відліку — **P0** | ~2h

## Інфраструктура для порівняння
- [ ] Єдиний інтерфейс train/predict для всіх baseline та подальших моделей (спільний контракт) — **P0** | ~3h
- [ ] Скрипт, що прогонює всі baseline одразу і зберігає результати в порівняльну таблицю — **P1** | ~2h

### Примітки
- Baseline варто зафіксувати і закомітити результати ДО початку роботи над складнішими архітектурами — інакше складно об'єктивно оцінити, чи multi-task підхід і калібрування дійсно дають приріст.

### Залежності
- Потребує готового multi-task артефакту з `data/TODO_preprocessing_pipeline.md`. Спільний train/predict контракт, визначений тут, повторно використовується в `ml/TODO_experiments_per_task_models.md` та `ml/TODO_experiments_multitask.md`.
