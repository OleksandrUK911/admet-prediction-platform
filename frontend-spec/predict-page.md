# Predict page

Шлях `/`. Двоколонковий на десктопі (>960px), одна колонка на мобільному.

## Layout

```
┌──────────────────────────────────────────────────────────┐
│  SMILES input ................................ [Predict]  │
│  Try an example: [Aspirin] [Thalidomide] [Caffeine]        │
├──────────────────────────────────┬─────────────────────────┤
│                                  │  Applicability domain:   │
│      ADMET Radar (5 axes)       │  ✅ In-domain             │
│   Solubility / BBB / Toxicity   │  (or ⚠ Out-of-domain —    │
│   (Tox21) / Clinical Trial      │   see below)             │
│   Toxicity / FDA Approval       │                          │
│                                  │                          │
├──────────────────────────────────┴─────────────────────────┤
│  [Solubility card]  [BBB card]  [Toxicity card]             │
│  [Clinical Tox card]  [FDA Approval card]                   │
│  (5 картки в ряд на desktop, 1 колонка на мобільному)        │
├──────────────────────────────────────────────────────────┤
│  ▸ Toxicity (Tox21 panel) — click to expand 12 assays        │
└──────────────────────────────────────────────────────────┘
```

## Компоненти
- `SmilesInput` + `ExampleChips` (Aspirin, Thalidomide — навмисно вибрана
  молекула з відомою токсичністю для демонстрації, Caffeine)
- `AdmetRadar` — див. [data-visualization.md](data-visualization.md)
- `ApplicabilityDomainBadge`
- `AdmetResultCard` × 5 (по одній на кожну вісь профілю)
- `Tox21PanelExpander` — акордеон з 12 рядками (assay code + людяна назва + probability + confidence)

## Стани

| Стан | Що видно |
|---|---|
| Idle | Лише input + приклади |
| Loading | Skeleton замінює радар + усі 5 карток одночасно |
| Success, in-domain | Повний дашборд + зелений `ApplicabilityDomainBadge` |
| Success, out-of-domain | Дашборд рендериться, але кожна картка отримує візуальний оверлей "low confidence — molecule outside training distribution" + жовтий/помаранчевий `ApplicabilityDomainBadge` з tooltip-поясненням. **Не error-стан** — дані показуються, лише з явним попередженням (окремий UX-сценарій, не просто помилка, як зазначено в `TODO_pages_flows.md`). |
| Error: invalid SMILES | Банер під input, дашборд не рендериться |
| Error: server unavailable | Банер + Retry |

## Флоу
1. Ввід SMILES → `POST /admet-profile`.
2. Response містить `applicability_domain.in_domain` — визначає, чи показувати оверлей low-confidence на картках (не блокує рендер, лише позначає).
3. Кожна картка й вісь радара показує `confidence` поруч зі значенням — ніколи просто "чисте" число без невизначеності (наскрізний принцип проєкту, з `TODO_ui_components.md`).
