# Multi-stage build - RDKit/xgboost/scikit-learn wheels are large, so we
# install into a venv in the builder stage and copy only the venv + app
# into a slim final image (see TODO/devops/TODO_containerization.md's note
# on image size). Same convention as project #1's backend/Dockerfile.
FROM python:3.12-slim AS builder

WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY backend/requirements.txt .
# xgboost's PyPI wheel pulls in nvidia-nccl-cu13 (~250MB) for multi-GPU
# support we don't use (CPU-only inference) - uninstalled in this same
# stage/layer so the final image (COPY --from=builder below) never
# contains it; uninstalling in a later layer wouldn't shrink the image,
# since Docker layers are additive.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y nvidia-nccl-cu13

FROM python:3.12-slim

# RDKit's Draw/rdMolDraw2D needs these X11 libs even for headless SVG
# rendering - without them RDKit's import chain fails at runtime
# (ImportError: libXrender.so.1), which pip install alone doesn't catch
# since it only installs Python packages.
RUN apt-get update && apt-get install --no-install-recommends -y \
    libxrender1 libxext6 libexpat1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# backend/app needs its parent package layout (backend.app.*) and imports
# ml/features.py directly (backend/app/inference.py: "from ml.features
# import ..."), so ml/ has to be on the image too - but only the one module
# it actually needs, not the training scripts/tests in ml/.
COPY backend/__init__.py backend/__init__.py
COPY backend/app backend/app
COPY ml/__init__.py ml/__init__.py
COPY ml/features.py ml/features.py

# The production model bundle (models/production/models.joblib, ~125MB,
# gitignored - a build artifact of the ml/ pipeline, not source) and its
# committed metadata.json are baked into the image, same precedent as
# project #1's backend/Dockerfile. This keeps `docker run` on its own
# sufficient (no required volume mount) at the cost of a heavier image and
# a rebuild whenever the model is retrained - acceptable at this project's
# scale. If you'd rather not rebake the image on every retrain, bind-mount
# your own models/production over this path instead:
#   docker run -v ./models/production:/app/models/production:ro ...
COPY models/production models/production

# data/processed/admet_processed.csv (also gitignored, produced by
# ml/preprocess.py) is likewise baked in - backend/app/inference.py's
# ModelService uses it to fit the applicability-domain nearest-neighbor
# index. It's optional at runtime (ModelService degrades to
# in_domain=True/distance_score=0.0 with an honest comment if the file is
# missing) but baking it in keeps the applicability-domain check real in
# the shipped image rather than silently disabled.
COPY data/processed/admet_processed.csv data/processed/admet_processed.csv

EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
