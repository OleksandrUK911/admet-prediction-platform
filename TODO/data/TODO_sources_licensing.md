# TODO — Джерела даних та ліцензування (ADMET Prediction)

## Вибір датасетів
- [ ] Обрати ADMET-датасети з MoleculeNet / TDC (Therapeutics Data Commons) — **P0** | ~2h
- [ ] Датасет розчинності: ESOL та/або AqSolDB (регресія logS) — **P0** | ~1h
- [ ] Датасет токсичності: Tox21 (12 nuclear receptor / stress response задач) та ClinTox (clinical trial toxicity + FDA approval) — **P0** | ~2h
- [ ] Датасет BBB-проникності: BBBP (blood-brain barrier penetration) — **P0** | ~1h
- [ ] Оцінити доцільність додавання CYP450 inhibition датасету (PubChem BioAssay через TDC) для розширення ADME-частини — **P2** | ~2h

## Ліцензування та атрибуція
- [ ] Перевірити ліцензію кожного датасету (MoleculeNet/TDC — здебільшого CC/ODC, але перевірити першоджерела: Tox21 — NIH, ClinTox — похідний з ClinicalTrials.gov) — **P0** | ~2h
- [ ] Задокументувати першоджерело та цитування для кожного датасету в `data/README.md` — **P1** | ~2h
- [ ] Явно вказати, що датасети використовуються в некомерційних research/educational цілях — **P1** | ~1h

### Примітки
- TDC надає уніфікований Python-інтерфейс (`PyTDC`) для завантаження — варто використати його замість ручного парсингу CSV, де можливо, щоб зменшити ризик помилок у канонічних SMILES.

### Залежності
- Вибір датасетів тут напряму визначає обсяг роботи в `data/TODO_preprocessing_pipeline.md` (канонізація/merge) та перелік задач у `ml/TODO_baseline.md` і `ml/TODO_experiments_multitask.md` — змінювати набір датасетів пізніше дорого.
