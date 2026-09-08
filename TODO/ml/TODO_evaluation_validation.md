# TODO — Оцінка та валідація (ADMET Prediction)

## Метрики
- [ ] Метрики: ROC-AUC, PR-AUC per-task (не accuracy — через дисбаланс) — **P0** | ~2h
- [ ] Регресійні метрики (RMSE, MAE, R²) для задачі розчинності — **P0** | ~1h
- [ ] Зведена dashboard-таблиця метрик per-task для порівняння моделей — **P1** | ~2h

## Стратегія валідації
- [ ] Scaffold-based crossvalidation (узгоджено з data/TODO_quality_validation.md) — **P0** | ~3h
- [ ] K-fold crossvalidation для стабільної оцінки на малих toxicity-датасетах (ClinTox) — **P1** | ~2h
- [ ] Фіксований holdout test set, який не використовується до фінальної оцінки — **P0** | ~1h

## Аналіз помилок
- [ ] Аналіз false positive / false negative для toxicity-задач (які структурні класи молекул помиляються найчастіше) — **P1** | ~3h
- [ ] Порівняння моделей per-task (baseline vs multi-task vs tuned) — **P0** | ~2h
- [ ] Візуалізація калібрувальних кривих (reliability diagram) per-task — **P1** | ~2h

### Примітки
- Для toxicity-задач false negative (пропущена токсична сполука) і false positive мають різну "вартість" — варто явно обговорити цей trade-off у документації, навіть якщо конкретний threshold обирається евристично.

### Залежності
- Split-логіка тут МАЄ бути тією самою, що визначена в `data/TODO_quality_validation.md` (scaffold split) — не дублювати іншу реалізацію. Порівняльна таблиця тут напряму живиться результатами з `ml/TODO_experiments_tracking.md`.
