# TODO — CI/CD (ADMET Prediction)

## GitHub Actions
- [ ] GitHub Actions: тести моделей на кожен PR (backend unit + integration тести) — **P0** | ~3h
- [ ] GitHub Actions: тести frontend (компонентні + smoke) — **P0** | ~2h
- [ ] Лінтинг та форматування (ruff/black для Python, eslint/prettier для frontend) у CI — **P1** | ~2h

## Перевірки, специфічні для ML
- [ ] Автоматична перевірка калібрування моделі при зміні даних (ECE не гірше порогового значення) — **P1** | ~4h
- [ ] Автоматична перевірка, що метрики per-task не деградували порівняно з попередньою версією моделі (regression gate) — **P1** | ~4h

## Деплой-пайплайн
- [ ] Автоматична збірка та публікація Docker-образів при мерджі в main — **P0** | ~2h
- [ ] Автоматичний деплой на staging при мерджі, ручне підтвердження для production — **P1** | ~3h

### Примітки
- Перевірка калібрування в CI можлива тільки завдяки детермінованому reproducible-пайплайну генерації даних (див. data/TODO_versioning_reproducibility.md) — без цього автоматичний gate буде "шумним" і ненадійним.

### Залежності
- ML-специфічні gate'и потребують детермінованого пайплайна з `data/TODO_versioning_reproducibility.md` та зафіксованих порогових метрик з `ml/TODO_experiments_tracking.md`. Тестові gate'и запускають набори з `backend/TODO_testing.md` та `frontend/TODO_testing.md`.
