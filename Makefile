.PHONY: build test smoke-train full-train mini-train resume-full-train evaluate-validation tensorboard-up tensorboard-trash ensure-env

ENV_FILE := .env
ENV_EXAMPLE_FILE := .env.example
MODEL_WEIGHTS ?=
MODEL_METADATA ?=
EVALUATE_ARGS ?=

build:
	go build -o /tmp/wordle-backprop-train ./cmd/train
	go build -o /tmp/wordle-backprop-play ./cmd/play
	go build -o /tmp/wordle-backprop-evaluate ./cmd/evaluate
	go build -o /tmp/wordle-backprop-export-scalars ./cmd/export-scalars

test:
	go test ./...

smoke-train:
	go run ./cmd/train --batch-size 32 --epochs 1 --learning-rate 0.05 --max-train-batches 2 --max-validation-batches 2 --log-every 1

full-train:
	go run ./cmd/train --batch-size 32 --epochs 50 --learning-rate 0.01 --learning-rate-decay --max-train-batches 0 --max-validation-batches 0 --log-every 50

mini-train:
	go run ./cmd/train --batch-size 32 --epochs 300 --learning-rate 0.01 --learning-rate-decay --max-train-batches 0 --max-validation-batches 0 --log-every 50 --train-split mini --validation-split mini

resume-full-train:
	go run ./cmd/train --batch-size 32 --epochs 1 --learning-rate 0.05 --max-train-batches 0 --max-validation-batches 0 --log-every 50 --resume

evaluate-validation:
	@test -n "$(MODEL_WEIGHTS)" || (echo "MODEL_WEIGHTS is required" >&2; exit 2)
	@test -n "$(MODEL_METADATA)" || (echo "MODEL_METADATA is required" >&2; exit 2)
	go run ./cmd/evaluate --model-weights "$(MODEL_WEIGHTS)" --model-metadata "$(MODEL_METADATA)" $(EVALUATE_ARGS)

ensure-env:
	@if [ ! -f "$(ENV_FILE)" ]; then \
		cp "$(ENV_EXAMPLE_FILE)" "$(ENV_FILE)"; \
		echo "Created $(ENV_FILE) from $(ENV_EXAMPLE_FILE)."; \
	fi

tensorboard-up: ensure-env
	docker compose -f docker-compose.tensorboard.yml up --build -d

tensorboard-trash:
	docker compose -f docker-compose.tensorboard.yml down --remove-orphans --volumes
