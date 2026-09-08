# TODO — Якість даних та валідація (ADMET Prediction)

## Класовий дисбаланс
- [ ] Аналіз класового дисбалансу для кожної toxicity-задачі (частка позитивних/негативних прикладів) — **P0** | ~2h
- [ ] Візуалізація розподілу міток по всіх ADMET-задачах (одна зведена таблиця/графік) — **P1** | ~2h

## Applicability domain
- [ ] Виявлення молекул поза applicability domain train-датасету (напр. через descriptor-based distance до train-розподілу) — **P1** | ~4h
- [ ] Аналіз хімічного простору датасетів (scaffold diversity, Bemis-Murcko scaffolds) — **P1** | ~3h

## Дублікати та leakage
- [ ] Перевірка на дублікати молекул усередині та між train/val/test split — **P0** | ~2h
- [ ] Перевірка на data leakage між задачами (одна й та сама молекула в train одного датасету і test іншого) — **P0** | ~2h
- [ ] Scaffold split замість random split для реалістичнішої оцінки generalization — **P0** | ~3h

### Примітки
- Random split для молекулярних даних систематично завищує метрики через структурну подібність молекул — scaffold split є стандартом де-факто в ADMET/drug discovery літературі, і його відсутність — типовий "red flag" для рецензента.

### Залежності
- Scaffold split, визначений тут, є єдиним джерелом істини для стратегії валідації в `ml/TODO_evaluation_validation.md` — не переізобретати split логіку там. Applicability domain аналіз тут — основа для `ml/TODO_calibration_uncertainty.md` (in-domain/out-of-domain прапорець).
