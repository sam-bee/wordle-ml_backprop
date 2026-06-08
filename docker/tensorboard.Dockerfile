FROM python:3.12-slim-bookworm

ARG DOCKERCOMPOSE_UID=1001
ARG DOCKERCOMPOSE_GID=1001

ENV DEBIAN_FRONTEND=noninteractive
ENV DOCKERCOMPOSE_UID=${DOCKERCOMPOSE_UID}
ENV DOCKERCOMPOSE_GID=${DOCKERCOMPOSE_GID}
ENV DOCKERCOMPOSE_USER=developer
ENV DOCKERCOMPOSE_GROUP=developer

RUN apt-get update && apt-get install -y --no-install-recommends \
		ca-certificates \
		tini \
	&& groupadd --gid ${DOCKERCOMPOSE_GID} ${DOCKERCOMPOSE_GROUP} \
	&& useradd --uid ${DOCKERCOMPOSE_UID} --gid ${DOCKERCOMPOSE_GID} --create-home --shell /bin/bash ${DOCKERCOMPOSE_USER} \
	&& mkdir -p /workspace /tensorboard/runs /app \
	&& chown -R ${DOCKERCOMPOSE_UID}:${DOCKERCOMPOSE_GID} /workspace /tensorboard /app /home/${DOCKERCOMPOSE_USER} \
	&& rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip \
	&& python -m pip install --no-cache-dir "setuptools<81" tensorboard

COPY docker/tensorboard-entrypoint.py /app/tensorboard-entrypoint.py
RUN chmod 0755 /app/tensorboard-entrypoint.py

USER ${DOCKERCOMPOSE_USER}
WORKDIR /workspace

ENV TENSORBOARD_LOGDIR=/tensorboard/runs
ENV TENSORBOARD_HOST=0.0.0.0
ENV TENSORBOARD_PORT=6006
ENV TENSORBOARD_DUMMY_RUN_DIR=/tensorboard/runs/dummy-backprop-run

ENTRYPOINT ["/usr/bin/tini", "--", "/app/tensorboard-entrypoint.py"]
