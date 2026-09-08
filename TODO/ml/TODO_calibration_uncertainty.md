# TODO — Калібрування та невизначеність (ADMET Prediction)

## Калібрування ймовірностей
- [ ] Калібрований класифікатор (Platt scaling) per-task — **P0** | ~3h
- [ ] Порівняння з isotonic regression per-task — **P1** | ~2h
- [ ] Перевірка калібрування через reliability diagram та Expected Calibration Error (ECE) — **P0** | ~2h

## Оцінка невизначеності
- [ ] Оцінка uncertainty через ensemble моделей (напр. bagging кількох моделей на bootstrap-вибірках) — **P0** | ~4h
- [ ] Альтернатива/доповнення: MC dropout для нейромережевої архітектури — **P2** | ~4h
- [ ] Агрегація uncertainty-оцінки в єдиний "рівень довіри" для API-відповіді — **P0** | ~2h

## Applicability domain у моделі
- [ ] Визначення applicability domain (чи молекула схожа на train-дані) на основі descriptor-based distance або density estimation — **P0** | ~4h
- [ ] Позначення "низька довіра" для молекул поза applicability domain у виводі моделі — **P0** | ~2h
- [ ] Юніт-тести на межові випадки (молекула явно поза доменом — перевірка, що прапорець спрацьовує) — **P1** | ~2h

### Примітки
- Обов'язково пояснювати calibration та uncertainty в README — це ключова відмінність від "іграшкового" проєкту та головний технічний акцент усього портфоліо-кейсу.

### Залежності
- Applicability domain тут будується на аналізі хімічного простору з `data/TODO_quality_validation.md`. Вихідні "рівень довіри" та applicability-прапорець напряму споживаються `backend/TODO_api_design.md` (поля відповіді) та `frontend/TODO_ui_components.md` (візуалізація невизначеності).
