# UI Components

Поля/ключі відповідають [../backend-spec/api-contract.md](../backend-spec/api-contract.md).

## AdmetResultCard
Одна картка на кожен елемент `profile` (5 карток: Solubility, BBB
Penetration, Toxicity, Clinical Trial Toxicity, FDA Approval). Кожна:
- Заголовок (назва endpoint) + risk-badge кольором (див. пороги в
  [design-system.md](design-system.md))
- Значення: для `solubility` — число + одиниці (logS); для решти —
  probability як `NN%`
- Confidence — завжди показаний як **діапазон**, не крапка: наприклад
  "58% (± 12%, moderate confidence)", а не просто "58%" (наскрізний принцип
  — не показувати чисті відсотки без невизначеності)
- Якщо картка в out-of-domain режимі — напівпрозорий оверлей з іконкою ⚠ і
  текстом "Low confidence — outside training data distribution"

## ApplicabilityDomainBadge
Компактний бейдж (іконка + короткий текст) біля радара:
- **In-domain:** зелена галочка, "Within known chemical space"
- **Out-of-domain:** жовтий трикутник, "Outside training distribution — treat results with caution", tooltip з поясненням, що це означає (посилання на About page)

## Tox21PanelExpander
Акордеон під основним дашбордом, заголовок "Toxicity (Tox21 panel) — click to
expand 12 assays". Розгорнутий стан — таблиця 12 рядків:
| Assay code | What it tests (human label) | Probability | Confidence |
Людські підписи для кодів (мінімальний набір, решта — по аналогії):
- `NR-AR` → "Androgen receptor activity"
- `NR-AhR` → "Aryl hydrocarbon receptor activity"
- `SR-p53` → "p53 stress response (DNA damage signal)"
- `SR-MMP` → "Mitochondrial membrane potential disruption"
(Повний список підписів для всіх 12 — окрема P1-задача перекладу
термінології, не блокує MVP: без підпису показувати хоча б offical assay code.)

## ErrorBanner
Як у проєкті №1 — інлайн під input, з Retry для 5xx.
