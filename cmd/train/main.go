package main

import (
	"flag"
	"fmt"
	"math"
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
	epochs := flags.Int("epochs", 1, "number of training epochs")
	learningRate := flags.Float64("learning-rate", training.DefaultPolicyLearningRate, "SGD learning rate")
	seed := flags.Int64("seed", training.DefaultPolicySeed, "GoMLX random seed")
	logEvery := flags.Int("log-every", 50, "print training progress every n batches; 0 disables batch progress logs")
	maxTrainBatches := flags.Int("max-train-batches", 0, "maximum training batches per epoch; 0 means all")
	maxValidationBatches := flags.Int("max-validation-batches", 25, "maximum validation batches per evaluation; 0 means all")
	flags.Usage = func() {
		fmt.Fprintf(flags.Output(), "usage: %s [options] [data-root]\n", flags.Name())
		flags.PrintDefaults()
	}

	if err := flags.Parse(os.Args[1:]); err != nil {
		os.Exit(2)
	}
	validatePositiveInt("batch-size", *batchSize)
	validatePositiveInt("epochs", *epochs)
	validateNonNegativeInt("log-every", *logEvery)
	validateNonNegativeInt("max-train-batches", *maxTrainBatches)
	validateNonNegativeInt("max-validation-batches", *maxValidationBatches)
	validatePositiveFiniteFloat("learning-rate", *learningRate)

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

	splits := make(map[data.SplitName]*data.Split, len(data.KnownSplits))
	for _, splitName := range data.KnownSplits {
		splitDir := filepath.Join(dataRoot, string(splitName))
		split, err := data.LoadSplit(splitDir)
		if err != nil {
			fmt.Fprintf(os.Stderr, "load %s split: %v\n", splitName, err)
			os.Exit(1)
		}
		splits[splitName] = split

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
	trainSplit := splits[data.SplitTrain]
	validationSplit := splits[data.SplitValidation]

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

	policyTrainer, err := training.NewPolicyTrainer(vocab, training.PolicyTrainerConfig{
		LearningRate: *learningRate,
		Seed:         *seed,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "build GoMLX policy trainer: %v\n", err)
		os.Exit(1)
	}
	defer policyTrainer.Close()
	fmt.Printf(
		"trainer: action_count=%d backend=%q device=%q learning_rate=%g seed=%d epochs=%d max_train_batches=%d max_validation_batches=%d\n",
		policyTrainer.ActionCount,
		policyTrainer.BackendDescription,
		policyTrainer.DeviceDescription,
		*learningRate,
		*seed,
		*epochs,
		*maxTrainBatches,
		*maxValidationBatches,
	)

	initialValidation, err := evaluateSplit(policyTrainer, validationSplit, *batchSize, *maxValidationBatches)
	if err != nil {
		fmt.Fprintf(os.Stderr, "initial validation: %v\n", err)
		os.Exit(1)
	}
	printLossStats("validation before training", initialValidation)

	lastValidation := initialValidation
	for epoch := 1; epoch <= *epochs; epoch++ {
		trainStats, err := trainEpoch(policyTrainer, trainSplit, *batchSize, *maxTrainBatches, *logEvery, epoch)
		if err != nil {
			fmt.Fprintf(os.Stderr, "epoch %d train: %v\n", epoch, err)
			os.Exit(1)
		}
		printLossStats(fmt.Sprintf("epoch %d train summary", epoch), trainStats)

		lastValidation, err = evaluateSplit(policyTrainer, validationSplit, *batchSize, *maxValidationBatches)
		if err != nil {
			fmt.Fprintf(os.Stderr, "epoch %d validation: %v\n", epoch, err)
			os.Exit(1)
		}
		printLossStats(fmt.Sprintf("epoch %d validation summary", epoch), lastValidation)
		fmt.Printf("epoch %d validation_delta_from_start=%.6f\n", epoch, lastValidation.MeanLoss()-initialValidation.MeanLoss())
	}
}

type lossStats struct {
	Batches int
	Samples int
	SumLoss float64
	First   float64
	Last    float64
}

func (stats *lossStats) Add(batch data.Batch, loss float64) {
	if stats.Batches == 0 {
		stats.First = loss
	}
	stats.Batches++
	stats.Samples += batch.Size()
	stats.SumLoss += loss
	stats.Last = loss
}

func (stats lossStats) MeanLoss() float64 {
	if stats.Batches == 0 {
		return math.NaN()
	}
	return stats.SumLoss / float64(stats.Batches)
}

func trainEpoch(policyTrainer *training.PolicyTrainer, split *data.Split, batchSize, maxBatches, logEvery, epoch int) (lossStats, error) {
	iterator, err := data.NewBatchIterator(split, batchSize)
	if err != nil {
		return lossStats{}, err
	}

	var stats lossStats
	for {
		if maxBatches > 0 && stats.Batches >= maxBatches {
			break
		}
		batch, ok := iterator.Next()
		if !ok {
			break
		}
		loss, err := policyTrainer.TrainBatch(batch)
		if err != nil {
			return lossStats{}, fmt.Errorf("batch %d: %w", stats.Batches+1, err)
		}
		stats.Add(batch, loss)
		if logEvery > 0 && stats.Batches%logEvery == 0 {
			fmt.Printf("epoch %d train progress: batches=%d samples=%d loss=%.6f mean_loss=%.6f\n", epoch, stats.Batches, stats.Samples, stats.Last, stats.MeanLoss())
		}
	}
	if stats.Batches == 0 {
		return lossStats{}, fmt.Errorf("no training batches")
	}
	return stats, nil
}

func evaluateSplit(policyTrainer *training.PolicyTrainer, split *data.Split, batchSize, maxBatches int) (lossStats, error) {
	iterator, err := data.NewBatchIterator(split, batchSize)
	if err != nil {
		return lossStats{}, err
	}

	var stats lossStats
	for {
		if maxBatches > 0 && stats.Batches >= maxBatches {
			break
		}
		batch, ok := iterator.Next()
		if !ok {
			break
		}
		loss, err := policyTrainer.EvalBatch(batch)
		if err != nil {
			return lossStats{}, fmt.Errorf("batch %d: %w", stats.Batches+1, err)
		}
		stats.Add(batch, loss)
	}
	if stats.Batches == 0 {
		return lossStats{}, fmt.Errorf("no evaluation batches")
	}
	return stats, nil
}

func printLossStats(label string, stats lossStats) {
	fmt.Printf("%s: batches=%d samples=%d first_loss=%.6f last_loss=%.6f mean_loss=%.6f\n", label, stats.Batches, stats.Samples, stats.First, stats.Last, stats.MeanLoss())
}

func validatePositiveInt(name string, value int) {
	if value <= 0 {
		fmt.Fprintf(os.Stderr, "%s must be greater than 0, got %d\n", name, value)
		os.Exit(2)
	}
}

func validateNonNegativeInt(name string, value int) {
	if value < 0 {
		fmt.Fprintf(os.Stderr, "%s must be greater than or equal to 0, got %d\n", name, value)
		os.Exit(2)
	}
}

func validatePositiveFiniteFloat(name string, value float64) {
	if math.IsNaN(value) || math.IsInf(value, 0) || value <= 0 {
		fmt.Fprintf(os.Stderr, "%s must be finite and greater than 0, got %g\n", name, value)
		os.Exit(2)
	}
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
