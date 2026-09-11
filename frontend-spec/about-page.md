# About page

Шлях `/about`. Статичний контент, легко доступний з головного екрана
(пункт навігації в header, не в футері — прямо вимагається приміткою в
`TODO_pages_flows.md`).

## Секції (одна під одною)
1. **What this is** — коротко: SMILES → RDKit → калібровані ADMET-передбачення з uncertainty
2. **Datasets** — таблиця: Solubility (ESOL/AqSolDB), BBB (BBBP), Toxicity (Tox21, 12 assays), Clinical Trial Toxicity + FDA Approval (ClinTox) — джерело, розмір, ліцензія, посилання (з `data/README.md`, коли буде створений)
3. **How confidence works** — пояснення calibration (Platt/isotonic) і applicability domain людською мовою, без формул: "confidence — це не точність моделі загалом, а оцінка надійності саме цього передбачення"
4. **Known limitations** — перелік: невеликі train-датасети per-task, Tox21 asssays не покривають усі механізми токсичності, applicability domain — евристика, не гарантія
5. **Disclaimer** — повний текст, той самий що в банері app-shell, тут розгорнутий

## Стани
Немає loading/error — статичний контент у білді.
