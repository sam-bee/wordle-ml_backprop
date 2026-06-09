package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"
	"time"

	"github.com/sam-bee/wordle-ml_backprop/internal/actionspace"
	"github.com/sam-bee/wordle-ml_backprop/internal/data"
	"github.com/sam-bee/wordle-ml_backprop/internal/model"
	"github.com/sam-bee/wordle-ml_backprop/internal/telemetry"
	"github.com/sam-bee/wordle-ml_backprop/internal/training"
)

const (
	defaultBatchSize       = 32
	checkpointRootDir      = "checkpoints"
	checkpointRunsDir      = "runs"
	checkpointGoMLXDir     = "gomlx"
	checkpointTelemetryDir = "tensorboard"
	manifestFileName       = "manifest.json"
	latestRunFileName      = "latest-run.txt"
)

func main() {
	flags := flag.NewFlagSet(filepath.Base(os.Args[0]), flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	batchSize := flags.Int("batch-size", defaultBatchSize, "number of samples per sequential batch")
	epochs := flags.Int("epochs", 1, "number of training epochs")
	learningRate := flags.Float64("learning-rate", training.DefaultPolicyLearningRate, "SGD learning rate")
	learningRateDecay := flags.Bool("learning-rate-decay", false, "enable GoMLX SGD decay: effective learning rate is initial learning rate / sqrt(global_step)")
	logEvery := flags.Int("log-every", 50, "print training progress every n batches; 0 disables batch progress logs")
	maxTrainBatches := flags.Int("max-train-batches", 0, "maximum training batches per epoch; 0 means all")
	maxValidationBatches := flags.Int("max-validation-batches", 25, "maximum validation batches per evaluation; 0 means all")
	resume := flags.Bool("resume", false, "resume from the latest checkpoint run")
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

	checkpoints, err := resolveCheckpointPaths(checkpointRootDir, *resume, time.Now().UTC())
	if err != nil {
		fmt.Fprintf(os.Stderr, "configure checkpoints: %v\n", err)
		os.Exit(1)
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
	trainerConfig.LearningRateDecay = *learningRateDecay
	trainerConfig.CheckpointDir = checkpoints.GoMLXDir
	trainerConfig.CheckpointMustLoad = checkpoints.Resume
	policyTrainer, err := training.NewPolicyTrainer(vocab, trainerConfig)
	if err != nil {
		fmt.Fprintf(os.Stderr, "build GoMLX policy trainer: %v\n", err)
		os.Exit(1)
	}
	defer policyTrainer.Close()

	telemetryWriter, err := telemetry.NewTensorBoardWriter(checkpoints.TensorBoardDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "build tensorboard telemetry writer: %v\n", err)
		os.Exit(1)
	}
	defer func() {
		if err := telemetryWriter.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "close tensorboard telemetry writer: %v\n", err)
		}
	}()
	fmt.Printf(
		"trainer: action_count=%d backend=%q device=%q learning_rate=%g learning_rate_decay=%t next_learning_rate=%g rng_seed=%d epochs=%d max_train_batches=%d max_validation_batches=%d\n",
		policyTrainer.ActionCount,
		policyTrainer.BackendDescription,
		policyTrainer.DeviceDescription,
		*learningRate,
		*learningRateDecay,
		policyTrainer.NextLearningRate(),
		policyTrainer.Seed,
		*epochs,
		*maxTrainBatches,
		*maxValidationBatches,
	)
	fmt.Printf(
		"checkpoint: root=%q run_id=%q run_dir=%q gomlx_dir=%q manifest=%q latest_run_file=%q keep=%d resume=%t loaded=%t latest=%q\n",
		checkpoints.RootDir,
		checkpoints.RunID,
		checkpoints.RunDir,
		policyTrainer.CheckpointDir,
		checkpoints.ManifestPath,
		checkpoints.LatestRunPath,
		policyTrainer.CheckpointKeep,
		checkpoints.Resume,
		policyTrainer.CheckpointLoaded,
		policyTrainer.LatestCheckpoint,
	)
	fmt.Printf("telemetry: tensorboard_dir=%q event_file=%q\n", checkpoints.TensorBoardDir, telemetryWriter.Path())

	runStarted := time.Now()

	initialValidation, err := evaluateSplit(policyTrainer, validationSplit, *batchSize, *maxValidationBatches)
	if err != nil {
		fmt.Fprintf(os.Stderr, "initial validation: %v\n", err)
		os.Exit(1)
	}
	printLossStats("validation before training", initialValidation)
	if err := writeInitialTelemetry(telemetryWriter, policyTrainer, initialValidation); err != nil {
		fmt.Fprintf(os.Stderr, "initial telemetry: %v\n", err)
		os.Exit(1)
	}

	lastValidation := initialValidation
	for epoch := 1; epoch <= *epochs; epoch++ {
		trainStats, err := trainEpoch(policyTrainer, trainSplit, *batchSize, *maxTrainBatches, *logEvery, epoch, telemetryWriter)
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
		if err := writeEpochTelemetry(telemetryWriter, policyTrainer, epoch, trainStats, lastValidation, validationDelta); err != nil {
			fmt.Fprintf(os.Stderr, "epoch %d telemetry: %v\n", epoch, err)
			os.Exit(1)
		}

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
			*learningRateDecay,
			*maxTrainBatches,
			*maxValidationBatches,
			initialValidation,
			trainStats,
			lastValidation,
			validationDelta,
			latestCheckpoint,
			telemetryWriter.Path(),
			checkpoints,
		)
		if err := writeCheckpointManifest(checkpoints.ManifestPath, manifest); err != nil {
			fmt.Fprintf(os.Stderr, "epoch %d checkpoint manifest: %v\n", epoch, err)
			os.Exit(1)
		}
		if err := writeLatestRunID(checkpoints.LatestRunPath, checkpoints.RunID); err != nil {
			fmt.Fprintf(os.Stderr, "epoch %d checkpoint latest-run update: %v\n", epoch, err)
			os.Exit(1)
		}
		fmt.Printf("checkpoint saved: epoch=%d global_step=%d latest=%q manifest=%q\n", epoch, policyTrainer.GlobalStep(), latestCheckpoint, checkpoints.ManifestPath)
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
	RunID                    string                  `json:"run_id"`
	RunDir                   string                  `json:"run_dir"`
	GoMLXDir                 string                  `json:"gomlx_dir"`
	TensorBoardDir           string                  `json:"tensorboard_dir"`
	TensorBoardEventFile     string                  `json:"tensorboard_event_file"`
	LatestGoMLXCheckpoint    string                  `json:"latest_gomlx_checkpoint"`
	LatestRunFile            string                  `json:"latest_run_file"`
	GoMLXCheckpointKeep      int                     `json:"gomlx_checkpoint_keep"`
	Resume                   bool                    `json:"resume"`
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
	LearningRateDecay        bool                    `json:"learning_rate_decay"`
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

type checkpointPaths struct {
	RootDir        string
	RunsDir        string
	RunID          string
	RunDir         string
	GoMLXDir       string
	TensorBoardDir string
	ManifestPath   string
	LatestRunPath  string
	Resume         bool
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
	learningRateDecay bool,
	maxTrainBatches int,
	maxValidationBatches int,
	initialValidation lossStats,
	lastTrain lossStats,
	lastValidation lossStats,
	validationDelta float64,
	latestCheckpoint string,
	tensorboardEventFile string,
	checkpoints checkpointPaths,
) checkpointManifest {
	return checkpointManifest{
		Version:                  1,
		UpdatedAtUTC:             time.Now().UTC().Format(time.RFC3339Nano),
		CheckpointRoot:           checkpoints.RootDir,
		RunID:                    checkpoints.RunID,
		RunDir:                   checkpoints.RunDir,
		GoMLXDir:                 policyTrainer.CheckpointDir,
		TensorBoardDir:           checkpoints.TensorBoardDir,
		TensorBoardEventFile:     tensorboardEventFile,
		LatestGoMLXCheckpoint:    latestCheckpoint,
		LatestRunFile:            checkpoints.LatestRunPath,
		GoMLXCheckpointKeep:      policyTrainer.CheckpointKeep,
		Resume:                   checkpoints.Resume,
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
		LearningRateDecay:        learningRateDecay,
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

func resolveCheckpointPaths(rootDir string, resume bool, now time.Time) (checkpointPaths, error) {
	paths := checkpointPaths{
		RootDir:       rootDir,
		RunsDir:       filepath.Join(rootDir, checkpointRunsDir),
		LatestRunPath: filepath.Join(rootDir, latestRunFileName),
		Resume:        resume,
	}
	if resume {
		content, err := os.ReadFile(paths.LatestRunPath)
		if err != nil {
			return paths, fmt.Errorf("read latest run file %s: %w", paths.LatestRunPath, err)
		}
		paths.RunID = strings.TrimSpace(string(content))
		if err := validateRunID(paths.RunID); err != nil {
			return paths, fmt.Errorf("latest run file %s: %w", paths.LatestRunPath, err)
		}
	} else {
		paths.RunID = fmt.Sprintf("run-%s", now.Format("20060102-150405.000000000"))
	}

	paths.RunDir = filepath.Join(paths.RunsDir, paths.RunID)
	paths.GoMLXDir = filepath.Join(paths.RunDir, checkpointGoMLXDir)
	paths.TensorBoardDir = filepath.Join(paths.RunDir, checkpointTelemetryDir)
	paths.ManifestPath = filepath.Join(paths.RunDir, manifestFileName)
	return paths, nil
}

func validateRunID(runID string) error {
	if runID == "" {
		return fmt.Errorf("run id is empty")
	}
	if runID == "." || runID == ".." {
		return fmt.Errorf("run id %q is invalid", runID)
	}
	if runID != filepath.Base(runID) || strings.ContainsAny(runID, `/\`) {
		return fmt.Errorf("run id %q must be a single path component", runID)
	}
	return nil
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

func writeLatestRunID(path, runID string) error {
	if err := validateRunID(runID); err != nil {
		return err
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o770); err != nil {
		return err
	}
	tmpPath := path + ".tmp"
	if err := os.WriteFile(tmpPath, []byte(runID+"\n"), 0o660); err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
}

func writeInitialTelemetry(writer *telemetry.TensorBoardWriter, policyTrainer *training.PolicyTrainer, validation lossStats) error {
	step := policyTrainer.GlobalStep()
	if err := writeLossStatsTelemetry(writer, "validation", step, validation); err != nil {
		return err
	}
	if err := writeTelemetryScalar(writer, "validation_delta_from_start", step, 0); err != nil {
		return err
	}
	if err := writeTelemetryScalar(writer, "learning_rate", step, policyTrainer.NextLearningRate()); err != nil {
		return err
	}
	return writeTelemetryScalar(writer, "epoch", step, 0)
}

func writeEpochTelemetry(writer *telemetry.TensorBoardWriter, policyTrainer *training.PolicyTrainer, epoch int, trainStats, validationStats lossStats, validationDelta float64) error {
	step := policyTrainer.GlobalStep()
	if err := writeTelemetryScalar(writer, "epoch", step, float64(epoch)); err != nil {
		return err
	}
	if err := writeTelemetryScalar(writer, "learning_rate", step, policyTrainer.NextLearningRate()); err != nil {
		return err
	}
	if err := writeLossStatsTelemetry(writer, "train", step, trainStats); err != nil {
		return err
	}
	if err := writeLossStatsTelemetry(writer, "validation", step, validationStats); err != nil {
		return err
	}
	return writeTelemetryScalar(writer, "validation_delta_from_start", step, validationDelta)
}

func writeTrainProgressTelemetry(writer *telemetry.TensorBoardWriter, policyTrainer *training.PolicyTrainer, epoch int, stats lossStats) error {
	step := policyTrainer.GlobalStep()
	if err := writeTelemetryScalar(writer, "epoch", step, float64(epoch)); err != nil {
		return err
	}
	if err := writeTelemetryScalar(writer, "learning_rate", step, policyTrainer.NextLearningRate()); err != nil {
		return err
	}
	if err := writeTelemetryScalar(writer, "train/progress_loss", step, stats.Last); err != nil {
		return err
	}
	if err := writeTelemetryScalar(writer, "train/progress_mean_loss", step, stats.MeanLoss()); err != nil {
		return err
	}
	if err := writeTelemetryScalar(writer, "train/progress_batches", step, float64(stats.Batches)); err != nil {
		return err
	}
	if err := writeTelemetryScalar(writer, "train/progress_samples", step, float64(stats.Samples)); err != nil {
		return err
	}
	if err := writeTelemetryScalar(writer, "train/progress_batches_per_second", step, stats.BatchesPerSecond()); err != nil {
		return err
	}
	return writeTelemetryScalar(writer, "train/progress_samples_per_second", step, stats.SamplesPerSecond())
}

func writeLossStatsTelemetry(writer *telemetry.TensorBoardWriter, prefix string, step int64, stats lossStats) error {
	if err := writeTelemetryScalar(writer, prefix+"/mean_loss", step, stats.MeanLoss()); err != nil {
		return err
	}
	if err := writeTelemetryScalar(writer, prefix+"/first_loss", step, stats.First); err != nil {
		return err
	}
	if err := writeTelemetryScalar(writer, prefix+"/last_loss", step, stats.Last); err != nil {
		return err
	}
	if err := writeTelemetryScalar(writer, prefix+"/batches", step, float64(stats.Batches)); err != nil {
		return err
	}
	if err := writeTelemetryScalar(writer, prefix+"/samples", step, float64(stats.Samples)); err != nil {
		return err
	}
	if err := writeTelemetryScalar(writer, prefix+"/batches_per_second", step, stats.BatchesPerSecond()); err != nil {
		return err
	}
	return writeTelemetryScalar(writer, prefix+"/samples_per_second", step, stats.SamplesPerSecond())
}

func writeTelemetryScalar(writer *telemetry.TensorBoardWriter, tag string, step int64, value float64) error {
	if err := writer.WriteScalar(tag, step, value); err != nil {
		return fmt.Errorf("write %s at step %d: %w", tag, step, err)
	}
	return nil
}

func trainEpoch(policyTrainer *training.PolicyTrainer, split *data.Split, batchSize, maxBatches, logEvery, epoch int, telemetryWriter *telemetry.TensorBoardWriter) (lossStats, error) {
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
			if telemetryWriter != nil {
				if err := writeTrainProgressTelemetry(telemetryWriter, policyTrainer, epoch, stats); err != nil {
					return lossStats{}, fmt.Errorf("batch %d telemetry: %w", stats.Batches, err)
				}
			}
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
