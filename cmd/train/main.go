package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime/debug"
	"time"

	"github.com/sam-bee/wordle-ml_backprop/internal/actionspace"
	"github.com/sam-bee/wordle-ml_backprop/internal/data"
	"github.com/sam-bee/wordle-ml_backprop/internal/model"
	"github.com/sam-bee/wordle-ml_backprop/internal/training"
)

const (
	defaultBatchSize       = 32
	checkpointRootDir      = "checkpoints"
	checkpointGoMLXDir     = "checkpoints/gomlx"
	checkpointManifestPath = "checkpoints/manifest.json"
)

func main() {
	flags := flag.NewFlagSet(filepath.Base(os.Args[0]), flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	batchSize := flags.Int("batch-size", defaultBatchSize, "number of samples per sequential batch")
	epochs := flags.Int("epochs", 1, "number of training epochs")
	learningRate := flags.Float64("learning-rate", training.DefaultPolicyLearningRate, "SGD learning rate")
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

	trainerConfig := training.DefaultPolicyTrainerConfig()
	trainerConfig.LearningRate = *learningRate
	trainerConfig.CheckpointDir = checkpointGoMLXDir
	policyTrainer, err := training.NewPolicyTrainer(vocab, trainerConfig)
	if err != nil {
		fmt.Fprintf(os.Stderr, "build GoMLX policy trainer: %v\n", err)
		os.Exit(1)
	}
	defer policyTrainer.Close()
	fmt.Printf(
		"trainer: action_count=%d backend=%q device=%q learning_rate=%g rng_seed=%d epochs=%d max_train_batches=%d max_validation_batches=%d\n",
		policyTrainer.ActionCount,
		policyTrainer.BackendDescription,
		policyTrainer.DeviceDescription,
		*learningRate,
		policyTrainer.Seed,
		*epochs,
		*maxTrainBatches,
		*maxValidationBatches,
	)
	fmt.Printf(
		"checkpoint: root=%q gomlx_dir=%q manifest=%q keep=%d loaded=%t latest=%q\n",
		checkpointRootDir,
		policyTrainer.CheckpointDir,
		checkpointManifestPath,
		policyTrainer.CheckpointKeep,
		policyTrainer.CheckpointLoaded,
		policyTrainer.LatestCheckpoint,
	)

	runStarted := time.Now()

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
		validationDelta := lastValidation.MeanLoss() - initialValidation.MeanLoss()
		fmt.Printf("epoch %d validation_delta_from_start=%.6f\n", epoch, validationDelta)

		latestCheckpoint, err := policyTrainer.SaveCheckpoint()
		if err != nil {
			fmt.Fprintf(os.Stderr, "epoch %d checkpoint save: %v\n", epoch, err)
			os.Exit(1)
		}
		manifest := buildCheckpointManifest(
			dataRoot,
			splits,
			policyTrainer,
			epoch,
			*batchSize,
			*learningRate,
			*maxTrainBatches,
			*maxValidationBatches,
			initialValidation,
			trainStats,
			lastValidation,
			validationDelta,
			latestCheckpoint,
		)
		if err := writeCheckpointManifest(checkpointManifestPath, manifest); err != nil {
			fmt.Fprintf(os.Stderr, "epoch %d checkpoint manifest: %v\n", epoch, err)
			os.Exit(1)
		}
		fmt.Printf("checkpoint saved: epoch=%d global_step=%d latest=%q manifest=%q\n", epoch, policyTrainer.GlobalStep(), latestCheckpoint, checkpointManifestPath)
	}
	fmt.Printf("training run complete: elapsed=%s\n", formatDuration(time.Since(runStarted)))
}

type lossStats struct {
	Batches int
	Samples int
	SumLoss float64
	First   float64
	Last    float64
	Elapsed time.Duration
}

type checkpointManifest struct {
	Version                  int                     `json:"version"`
	UpdatedAtUTC             string                  `json:"updated_at_utc"`
	CheckpointRoot           string                  `json:"checkpoint_root"`
	GoMLXDir                 string                  `json:"gomlx_dir"`
	LatestGoMLXCheckpoint    string                  `json:"latest_gomlx_checkpoint"`
	GoMLXCheckpointKeep      int                     `json:"gomlx_checkpoint_keep"`
	EpochsCompletedThisRun   int                     `json:"epochs_completed_this_run"`
	GlobalStep               int64                   `json:"global_step"`
	DataRoot                 string                  `json:"data_root"`
	Splits                   map[string]splitSummary `json:"splits"`
	ActionVocabularySource   string                  `json:"action_vocabulary_source"`
	ActionCount              int                     `json:"action_count"`
	FixedActionFeatureDim    int                     `json:"fixed_action_feature_dim"`
	Backend                  string                  `json:"backend"`
	Device                   string                  `json:"device"`
	LearningRate             float64                 `json:"learning_rate"`
	RNGSeed                  int64                   `json:"rng_seed"`
	BatchSize                int                     `json:"batch_size"`
	MaxTrainBatches          int                     `json:"max_train_batches"`
	MaxValidationBatches     int                     `json:"max_validation_batches"`
	InitialValidation        lossSummary             `json:"initial_validation"`
	LastTrain                lossSummary             `json:"last_train"`
	LastValidation           lossSummary             `json:"last_validation"`
	ValidationDeltaFromStart float64                 `json:"validation_delta_from_start"`
	VCS                      map[string]string       `json:"vcs,omitempty"`
}

type splitSummary struct {
	Samples                   int    `json:"samples"`
	Solutions                 uint32 `json:"solutions"`
	TopK                      uint32 `json:"top_k"`
	MaxTurns                  uint32 `json:"max_turns"`
	GuessVocabSize            uint32 `json:"guess_vocab_size"`
	GlobalSolutionVocabSize   uint32 `json:"global_solution_vocab_size"`
	RecordSizeBytes           uint32 `json:"record_size_bytes"`
	WordlistHash              string `json:"wordlist_hash"`
	GeneratorCommit           string `json:"generator_commit"`
	GeneratorWorkingTreeDirty bool   `json:"generator_working_tree_dirty"`
	GeneratedAtUTC            string `json:"generated_at_utc"`
	DatasetSeed               int64  `json:"dataset_seed"`
}

type lossSummary struct {
	Batches          int     `json:"batches"`
	Samples          int     `json:"samples"`
	FirstLoss        float64 `json:"first_loss"`
	LastLoss         float64 `json:"last_loss"`
	MeanLoss         float64 `json:"mean_loss"`
	Elapsed          string  `json:"elapsed"`
	ElapsedMillis    int64   `json:"elapsed_millis"`
	BatchesPerSecond float64 `json:"batches_per_second"`
	SamplesPerSecond float64 `json:"samples_per_second"`
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

func (stats lossStats) BatchesPerSecond() float64 {
	if stats.Elapsed <= 0 {
		return math.NaN()
	}
	return float64(stats.Batches) / stats.Elapsed.Seconds()
}

func (stats lossStats) SamplesPerSecond() float64 {
	if stats.Elapsed <= 0 {
		return math.NaN()
	}
	return float64(stats.Samples) / stats.Elapsed.Seconds()
}

func buildCheckpointManifest(
	dataRoot string,
	splits map[data.SplitName]*data.Split,
	policyTrainer *training.PolicyTrainer,
	epoch int,
	batchSize int,
	learningRate float64,
	maxTrainBatches int,
	maxValidationBatches int,
	initialValidation lossStats,
	lastTrain lossStats,
	lastValidation lossStats,
	validationDelta float64,
	latestCheckpoint string,
) checkpointManifest {
	return checkpointManifest{
		Version:                  1,
		UpdatedAtUTC:             time.Now().UTC().Format(time.RFC3339Nano),
		CheckpointRoot:           checkpointRootDir,
		GoMLXDir:                 policyTrainer.CheckpointDir,
		LatestGoMLXCheckpoint:    latestCheckpoint,
		GoMLXCheckpointKeep:      policyTrainer.CheckpointKeep,
		EpochsCompletedThisRun:   epoch,
		GlobalStep:               policyTrainer.GlobalStep(),
		DataRoot:                 dataRoot,
		Splits:                   summarizeSplits(splits),
		ActionVocabularySource:   "github.com/sam-bee/wordle-ml_game-engine/words.GetActionSpace",
		ActionCount:              policyTrainer.ActionCount,
		FixedActionFeatureDim:    model.FixedActionFeatureDim,
		Backend:                  policyTrainer.BackendDescription,
		Device:                   policyTrainer.DeviceDescription,
		LearningRate:             learningRate,
		RNGSeed:                  policyTrainer.Seed,
		BatchSize:                batchSize,
		MaxTrainBatches:          maxTrainBatches,
		MaxValidationBatches:     maxValidationBatches,
		InitialValidation:        summarizeLoss(initialValidation),
		LastTrain:                summarizeLoss(lastTrain),
		LastValidation:           summarizeLoss(lastValidation),
		ValidationDeltaFromStart: finiteFloat(validationDelta),
		VCS:                      vcsSettings(),
	}
}

func summarizeSplits(splits map[data.SplitName]*data.Split) map[string]splitSummary {
	summaries := make(map[string]splitSummary, len(splits))
	for splitName, split := range splits {
		metadata := split.Metadata
		summaries[string(splitName)] = splitSummary{
			Samples:                   split.SampleCount(),
			Solutions:                 metadata.SolutionCount,
			TopK:                      metadata.TopK,
			MaxTurns:                  metadata.MaxTurns,
			GuessVocabSize:            metadata.GuessVocabSize,
			GlobalSolutionVocabSize:   metadata.GlobalSolutionVocabSize,
			RecordSizeBytes:           metadata.RecordSizeBytes,
			WordlistHash:              metadata.WordlistHash,
			GeneratorCommit:           metadata.GeneratorCommit,
			GeneratorWorkingTreeDirty: metadata.GeneratorWorkingTreeDirty,
			GeneratedAtUTC:            metadata.GeneratedAtUTC,
			DatasetSeed:               metadata.Seed,
		}
	}
	return summaries
}

func summarizeLoss(stats lossStats) lossSummary {
	return lossSummary{
		Batches:          stats.Batches,
		Samples:          stats.Samples,
		FirstLoss:        finiteFloat(stats.First),
		LastLoss:         finiteFloat(stats.Last),
		MeanLoss:         finiteFloat(stats.MeanLoss()),
		Elapsed:          formatDuration(stats.Elapsed),
		ElapsedMillis:    stats.Elapsed.Milliseconds(),
		BatchesPerSecond: finiteFloat(stats.BatchesPerSecond()),
		SamplesPerSecond: finiteFloat(stats.SamplesPerSecond()),
	}
}

func finiteFloat(value float64) float64 {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return 0
	}
	return value
}

func vcsSettings() map[string]string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return nil
	}
	settings := make(map[string]string)
	for _, setting := range info.Settings {
		switch setting.Key {
		case "vcs.revision", "vcs.time", "vcs.modified":
			settings[setting.Key] = setting.Value
		}
	}
	if len(settings) == 0 {
		return nil
	}
	return settings
}

func writeCheckpointManifest(path string, manifest checkpointManifest) error {
	content, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return err
	}
	content = append(content, '\n')

	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o770); err != nil {
		return err
	}
	tmpPath := path + ".tmp"
	if err := os.WriteFile(tmpPath, content, 0o660); err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
}

func trainEpoch(policyTrainer *training.PolicyTrainer, split *data.Split, batchSize, maxBatches, logEvery, epoch int) (lossStats, error) {
	iterator, err := data.NewBatchIterator(split, batchSize)
	if err != nil {
		return lossStats{}, err
	}

	started := time.Now()
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
		stats.Elapsed = time.Since(started)
		if logEvery > 0 && stats.Batches%logEvery == 0 {
			fmt.Printf("epoch %d train progress: batches=%d samples=%d loss=%.6f mean_loss=%.6f elapsed=%s batches_per_sec=%.2f samples_per_sec=%.2f\n", epoch, stats.Batches, stats.Samples, stats.Last, stats.MeanLoss(), formatDuration(stats.Elapsed), stats.BatchesPerSecond(), stats.SamplesPerSecond())
		}
	}
	if stats.Batches == 0 {
		return lossStats{}, fmt.Errorf("no training batches")
	}
	stats.Elapsed = time.Since(started)
	return stats, nil
}

func evaluateSplit(policyTrainer *training.PolicyTrainer, split *data.Split, batchSize, maxBatches int) (lossStats, error) {
	iterator, err := data.NewBatchIterator(split, batchSize)
	if err != nil {
		return lossStats{}, err
	}

	started := time.Now()
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
		stats.Elapsed = time.Since(started)
	}
	if stats.Batches == 0 {
		return lossStats{}, fmt.Errorf("no evaluation batches")
	}
	stats.Elapsed = time.Since(started)
	return stats, nil
}

func printLossStats(label string, stats lossStats) {
	fmt.Printf("%s: batches=%d samples=%d first_loss=%.6f last_loss=%.6f mean_loss=%.6f elapsed=%s batches_per_sec=%.2f samples_per_sec=%.2f\n", label, stats.Batches, stats.Samples, stats.First, stats.Last, stats.MeanLoss(), formatDuration(stats.Elapsed), stats.BatchesPerSecond(), stats.SamplesPerSecond())
}

func formatDuration(duration time.Duration) string {
	return duration.Round(time.Millisecond).String()
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
