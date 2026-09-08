# TODO — Контейнеризація (ADMET Prediction)

## Docker
- [ ] Dockerfile backend (multi-task моделі можуть бути важчими — перевірити розмір образу) — **P0** | ~2h
- [ ] Multi-stage build для зменшення фінального розміру образу backend — **P1** | ~2h
- [ ] Dockerfile frontend (build + сервінг статики) — **P0** | ~2h
- [ ] docker-compose.yml (backend + frontend + postgres) — **P0** | ~2h

## Локальне середовище розробки
- [ ] `.env.example` з усіма необхідними змінними середовища — **P0** | ~1h
- [ ] Скрипт швидкого локального запуску (`docker-compose up`) з seed-даними для БД — **P1** | ~2h

### Примітки
- Multi-task моделі та RDKit-залежності можуть суттєво роздувати образ — варто явно виміряти розмір і за потреби перейти на slim base image.
