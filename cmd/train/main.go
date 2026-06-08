package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/sam-bee/wordle-ml_backprop/internal/actionspace"
	"github.com/sam-bee/wordle-ml_backprop/internal/data"
	"github.com/sam-bee/wordle-ml_backprop/internal/model"
	"github.com/sam-bee/wordle-ml_backprop/internal/training"
)

const defaultBatchSize = 32

func main() {
	flags := flag.NewFlagSet(filepath.Base(os.Args[0]), flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	batchSize := flags.Int("batch-size", defaultBatchSize, "number of samples per sequential batch")
	flags.Usage = func() {
		fmt.Fprintf(flags.Output(), "usage: %s [--batch-size n] [data-root]\n", flags.Name())
		flags.PrintDefaults()
	}

	if err := flags.Parse(os.Args[1:]); err != nil {
		os.Exit(2)
	}
	if *batchSize <= 0 {
		fmt.Fprintf(os.Stderr, "batch-size must be greater than 0, got %d\n", *batchSize)
		os.Exit(2)
	}

	dataRoot := "data"
	args := flags.Args()
	if len(args) > 1 {
		flags.Usage()
		os.Exit(2)
	}
	if len(args) == 1 {
		dataRoot = args[0]
	}

	fmt.Println("wordle backprop trainer starting")

	var trainSplit *data.Split
	for _, splitName := range data.KnownSplits {
		splitDir := filepath.Join(dataRoot, string(splitName))
		split, err := data.LoadSplit(splitDir)
		if err != nil {
			fmt.Fprintf(os.Stderr, "load %s split: %v\n", splitName, err)
			os.Exit(1)
		}
		if splitName == data.SplitTrain {
			trainSplit = split
		}

		fmt.Printf(
			"%s: samples=%d solutions=%d top_k=%d max_turns=%d guess_vocab=%d record_size=%d\n",
			split.Metadata.Split,
			split.SampleCount(),
			split.Metadata.SolutionCount,
			split.Metadata.TopK,
			split.Metadata.MaxTurns,
			split.Metadata.GuessVocabSize,
			split.Metadata.RecordSizeBytes,
		)
	}

	iterator, err := data.NewBatchIterator(trainSplit, *batchSize)
	if err != nil {
		fmt.Fprintf(os.Stderr, "build training batches: %v\n", err)
		os.Exit(1)
	}
	firstBatch, ok := iterator.Next()
	if !ok {
		fmt.Fprintln(os.Stderr, "build training batches: train split has no samples")
		os.Exit(1)
	}
	printBatchSummary(firstBatch)

	vocab, err := actionspace.Load()
	if err != nil {
		fmt.Fprintf(os.Stderr, "load action space: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf(
		"action space: words=%d fixed_action_features[%d,%d]\n",
		len(vocab.Words),
		len(vocab.Words),
		model.FixedActionFeatureDim,
	)

	result, err := training.RunPolicyStep(firstBatch, vocab)
	if err != nil {
		fmt.Fprintf(os.Stderr, "run GoMLX policy training step: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf(
		"gomlx policy step: action_count=%d initial_loss=%.6f training_loss=%.6f post_update_loss=%.6f update_completed=%t\n",
		result.ActionCount,
		result.InitialLoss,
		result.TrainingLoss,
		result.PostUpdateLoss,
		result.UpdateCompleted,
	)
}

func printBatchSummary(batch data.Batch) {
	dims := batch.Dimensions
	fmt.Printf(
		"first train batch: samples=%d input_dims=turn_depth[%d] previous_guess_words[%d,%d,%d] previous_feedback[%d,%d,%d] shortlist_size_before[%d] target_dims=top_k_guess_words[%d,%d,%d] reduction_ratios[%d,%d] worst_case_sizes[%d,%d]\n",
		batch.Size(),
		dims.BatchSize,
		dims.BatchSize,
		dims.MaxTurns,
		dims.WordLength,
		dims.BatchSize,
		dims.MaxTurns,
		dims.WordLength,
		dims.BatchSize,
		dims.BatchSize,
		dims.TopK,
		dims.WordLength,
		dims.BatchSize,
		dims.TopK,
		dims.BatchSize,
		dims.TopK,
	)
}
