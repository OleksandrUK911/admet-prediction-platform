# Frontend Spec — ADMET Prediction

Детальна специфікація вигляду й поведінки фронтенду: layout сторінок,
композиція компонентів, стани UI, конкретні дані для графіків. Доповнення до
`TODO/frontend/*.md` (чекліст задач) — тут описано **як саме** кожен екран
виглядає, узгоджено з `backend-spec/api-contract.md`.

## Файли
- [app-shell.md](app-shell.md) — header/навігація/дисклеймер-банер
- [predict-page.md](predict-page.md) — головна сторінка з ADMET-дашбордом
- [comparison-page.md](comparison-page.md) — порівняння кількох молекул
- [about-page.md](about-page.md) — методологія/обмеження/disclaimer
- [components.md](components.md) — специфікації UI-компонентів
- [data-visualization.md](data-visualization.md) — радар-графік, confidence-бари, Tox21 panel
- [design-system.md](design-system.md) — кольори (зокрема risk-кодування), типографіка, spacing

## Статус
🟡 Чернетка — написано ДО реалізації (портфоліо-правило: проєкт №2 не
починається, поки №1 не задеплоєно; це виключно документація/планування, без
завантаження даних чи коду). Список ADMET-задач узгоджено з датасетами,
обраними в `TODO/data/TODO_sources_licensing.md` (ESOL/AqSolDB, Tox21,
ClinTox, BBBP) — якщо вибір датасетів зміниться, ці файли теж треба оновити.
