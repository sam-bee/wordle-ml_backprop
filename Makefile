.PHONY: build test smoke-train

build:
	go build -o /tmp/wordle-backprop-train ./cmd/train

test:
	go test ./...

smoke-train:
	go run ./cmd/train --batch-size 32 --epochs 1 --max-train-batches 2 --max-validation-batches 2 --log-every 1
