# TODO — Feature Engineering (ADMET Prediction)

## Набори дескрипторів
- [ ] Molecular fingerprints (Morgan/ECFP4, різні радіуси та розмірність бітового вектора) — **P0** | ~2h
- [ ] RDKit фізико-хімічні дескриптори (MW, LogP, TPSA, HBD/HBA, кількість ротованих зв'язків тощо) — **P0** | ~2h
- [ ] Порівняння fingerprint-only vs descriptor-only vs комбінований feature set per-task — **P1** | ~4h

## Вибір ознак per-task
- [ ] Аналіз, які дескриптори найбільш релевантні для кожної ADMET-задачі (напр. TPSA/LogP для BBB, LogP/MW для solubility) — **P1** | ~3h
- [ ] Feature selection для high-dimensional toxicity fingerprints (variance threshold, кореляційний фільтр) — **P2** | ~2h
- [ ] Перевірка мультиколінеарності серед фізико-хімічних дескрипторів — **P2** | ~1h

## Масштабування та підготовка
- [ ] Стандартизація/нормалізація числових дескрипторів (fit тільки на train) — **P0** | ~1h
- [ ] Збереження feature pipeline (scaler, feature list) разом з моделлю для inference — **P0** | ~2h

### Примітки
- Для toxicity-задач fingerprints зазвичай дають кращий сигнал, ніж прості дескриптори, а для solubility/BBB — навпаки фізико-хімічні дескриптори часто конкурентні або кращі; це варто емпірично підтвердити, а не брати на віру.

### Залежності
- Збережений feature pipeline тут — той самий артефакт, що використовується інференсом у `backend/TODO_api_design.md` та повинен бути зафіксований у `ml/TODO_model_registry.md` разом з вагами моделі.
