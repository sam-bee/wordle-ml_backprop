.PHONY: build test smoke-train full-train resume-full-train tensorboard-up tensorboard-trash ensure-env

ENV_FILE := .env
ENV_EXAMPLE_FILE := .env.example

RUN_LABEL ?= deeper-dense-1x256-2x128
SMOKE_RUN_LABEL ?= smoke-$(RUN_LABEL)

build:
	go build -o /tmp/wordle-backprop-train ./cmd/train
	go build -o /tmp/wordle-backprop-play ./cmd/play

test:
	go test ./...

smoke-train:
	go run ./cmd/train --batch-size 32 --epochs 1 --learning-rate 0.05 --max-train-batches 2 --max-validation-batches 2 --log-every 1 --run-label "$(SMOKE_RUN_LABEL)"

full-train:
	go run ./cmd/train --batch-size 32 --epochs 50 --learning-rate 0.01 --learning-rate-decay --max-train-batches 0 --max-validation-batches 0 --log-every 50 --run-label "$(RUN_LABEL)"

resume-full-train:
	go run ./cmd/train --batch-size 32 --epochs 1 --learning-rate 0.05 --max-train-batches 0 --max-validation-batches 0 --log-every 50 --resume

ensure-env:
	@if [ ! -f "$(ENV_FILE)" ]; then \
		cp "$(ENV_EXAMPLE_FILE)" "$(ENV_FILE)"; \
		echo "Created $(ENV_FILE) from $(ENV_EXAMPLE_FILE)."; \
	fi

tensorboard-up: ensure-env
	docker compose -f docker-compose.tensorboard.yml up --build -d

tensorboard-trash:
	docker compose -f docker-compose.tensorboard.yml down --remove-orphans --volumes
