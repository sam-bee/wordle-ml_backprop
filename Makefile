.PHONY: build test smoke-train full-train resume-full-train

build:
	go build -o /tmp/wordle-backprop-train ./cmd/train

test:
	go test ./...

smoke-train:
	go run ./cmd/train --batch-size 32 --epochs 1 --learning-rate 0.05 --max-train-batches 2 --max-validation-batches 2 --log-every 1

full-train:
	go run ./cmd/train --batch-size 32 --epochs 1 --learning-rate 0.05 --max-train-batches 0 --max-validation-batches 0 --log-every 50

resume-full-train:
	go run ./cmd/train --batch-size 32 --epochs 1 --learning-rate 0.05 --max-train-batches 0 --max-validation-batches 0 --log-every 50 --resume
