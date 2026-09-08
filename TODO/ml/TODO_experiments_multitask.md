# TODO — Експерименти: multi-task архітектура (ADMET Prediction)

## Архітектура моделі
- [ ] Multi-task класифікатор (спільний "тулуб" + окремі голови на задачу) на базі нейромережі або градієнтного бустингу з multi-output — **P0** | ~1d
- [ ] Проєктування shared backbone (спільні шари/представлення на основі fingerprints + дескрипторів) — **P0** | ~4h
- [ ] Task-specific heads (окрема вихідна голова на кожен ADMET-параметр з власною функцією втрат) — **P0** | ~4h

## Loss weighting та балансування задач
- [ ] Task weighting у сумарній loss-функції (щоб великі задачі, напр. Tox21, не домінували над градієнтом) — **P0** | ~4h
- [ ] Балансування класів (class weights) всередині кожної голови для toxicity-задач — **P0** | ~2h
- [ ] Експеримент з динамічним/adaptive task weighting (напр. uncertainty weighting) як опційне вдосконалення — **P2** | ~4h

## Підбір гіперпараметрів
- [ ] Hyperparameter tuning (Optuna/grid search) для обраної multi-task архітектури — **P1** | ~1d
- [ ] Порівняння multi-task vs single-task підходів за метриками та часом inference — **P0** | ~3h

### Примітки
- При multi-task навчанні задачі з великою кількістю прикладів (напр. Tox21) можуть домінувати над градієнтом і "заглушувати" менші задачі — варто експериментувати з task weighting у loss-функції.

### Залежності
- Потребує того самого feature pipeline, що й `ml/TODO_experiments_per_task_models.md` (див. `ml/TODO_feature_engineering.md`). Порівняння з per-task підходом фіксується та підсумовується в `ml/TODO_experiments_tracking.md`.
