package training

import (
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/xla"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/checkpoints"
	gomlxtrain "github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"

	"github.com/sam-bee/wordle-ml_backprop/internal/actionspace"
	"github.com/sam-bee/wordle-ml_backprop/internal/data"
	"github.com/sam-bee/wordle-ml_backprop/internal/model"
)

const (
	DefaultPolicyLearningRate = 0.05
	DefaultCheckpointKeep     = 3

	policyBackendConfig       = "cuda"
	policyRequiredDeviceCount = 1
)

type PolicyTrainerConfig struct {
	LearningRate       float64
	Seed               int64
	CheckpointDir      string
	CheckpointKeep     int
	CheckpointMustLoad bool
}

func DefaultPolicyTrainerConfig() PolicyTrainerConfig {
	return PolicyTrainerConfig{
		LearningRate:   DefaultPolicyLearningRate,
		Seed:           NewPolicySeed(),
		CheckpointKeep: DefaultCheckpointKeep,
	}
}

func NewPolicySeed() int64 {
	return time.Now().UnixNano()
}

func (config PolicyTrainerConfig) Validate() error {
	if math.IsNaN(config.LearningRate) || math.IsInf(config.LearningRate, 0) || config.LearningRate <= 0 {
		return fmt.Errorf("learning rate must be finite and > 0, got %v", config.LearningRate)
	}
	if config.CheckpointDir != "" && config.CheckpointKeep < -1 {
		return fmt.Errorf("checkpoint keep must be -1 or >= 0, got %d", config.CheckpointKeep)
	}
	if config.CheckpointMustLoad && config.CheckpointDir == "" {
		return fmt.Errorf("checkpoint dir is required when checkpoint loading is required")
	}
	return nil
}

type PolicyTrainer struct {
	vocab              actionspace.Vocabulary
	backend            backends.Backend
	trainer            *gomlxtrain.Trainer
	checkpoint         *checkpoints.Handler
	ActionCount        int
	BackendDescription string
	DeviceDescription  string
	Seed               int64
	CheckpointDir      string
	CheckpointKeep     int
	CheckpointLoaded   bool
	LatestCheckpoint   string
}

type PolicyStepResult struct {
	ActionCount        int
	BackendDescription string
	DeviceDescription  string
	InitialLoss        float64
	TrainingLoss       float64
	PostUpdateLoss     float64
	UpdateCompleted    bool
	Seed               int64
}

func NewPolicyTrainer(vocab actionspace.Vocabulary, config PolicyTrainerConfig) (*PolicyTrainer, error) {
	if err := validateActionVocabulary(vocab); err != nil {
		return nil, err
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	if config.Seed == 0 {
		config.Seed = NewPolicySeed()
	}
	if config.CheckpointDir != "" && config.CheckpointKeep == 0 {
		config.CheckpointKeep = DefaultCheckpointKeep
	}

	backend, err := newPolicyBackend()
	if err != nil {
		return nil, err
	}

	ctx := context.New()
	if err := ctx.SetRNGStateFromSeed(config.Seed); err != nil {
		backend.Finalize()
		return nil, fmt.Errorf("seed GoMLX context: %w", err)
	}
	if err := model.ConfigureRandomInitialization(ctx, model.DefaultRandomInitializationConfig()); err != nil {
		backend.Finalize()
		return nil, fmt.Errorf("configure random initialization: %w", err)
	}

	var checkpoint *checkpoints.Handler
	checkpointLoaded := false
	latestCheckpoint := ""
	if config.CheckpointDir != "" {
		checkpointConfig := checkpoints.Build(ctx)
		if config.CheckpointMustLoad {
			checkpointConfig = checkpoints.Load(ctx)
		}
		checkpoint, err = checkpointConfig.Dir(config.CheckpointDir).
			Keep(config.CheckpointKeep).
			Done()
		if err != nil {
			backend.Finalize()
			return nil, fmt.Errorf("configure checkpoints: %w", err)
		}
		names, err := checkpoint.ListCheckpoints()
		if err != nil {
			backend.Finalize()
			return nil, fmt.Errorf("list checkpoints: %w", err)
		}
		if len(names) > 0 {
			checkpointLoaded = true
			latestCheckpoint = names[len(names)-1]
		}
	}

	trainerCtx := ctx
	if checkpointLoaded {
		trainerCtx = ctx.Reuse()
	}

	optimizer := optimizers.StochasticGradientDescent().
		WithLearningRate(config.LearningRate).
		WithDecay(false).
		Done()
	gomlxTrainer := gomlxtrain.NewTrainer(
		backend,
		trainerCtx,
		model.PolicyModel,
		PolicyLoss,
		optimizer,
		nil,
		nil,
	)

	return &PolicyTrainer{
		vocab:              vocab,
		backend:            backend,
		trainer:            gomlxTrainer,
		checkpoint:         checkpoint,
		ActionCount:        len(vocab.Words),
		BackendDescription: backend.Description(),
		DeviceDescription:  backend.DeviceDescription(0),
		Seed:               config.Seed,
		CheckpointDir:      config.CheckpointDir,
		CheckpointKeep:     config.CheckpointKeep,
		CheckpointLoaded:   checkpointLoaded,
		LatestCheckpoint:   latestCheckpoint,
	}, nil
}

func (trainer *PolicyTrainer) Close() {
	if trainer == nil || trainer.backend == nil {
		return
	}
	trainer.backend.Finalize()
	trainer.backend = nil
}

func (trainer *PolicyTrainer) TrainBatch(batch data.Batch) (float64, error) {
	if trainer == nil || trainer.trainer == nil {
		return 0, fmt.Errorf("policy trainer is nil")
	}
	return trainPolicyLoss(trainer.trainer, batch, trainer.vocab)
}

func (trainer *PolicyTrainer) EvalBatch(batch data.Batch) (float64, error) {
	if trainer == nil || trainer.trainer == nil {
		return 0, fmt.Errorf("policy trainer is nil")
	}
	return evalPolicyLoss(trainer.trainer, batch, trainer.vocab)
}

func (trainer *PolicyTrainer) GlobalStep() int64 {
	if trainer == nil || trainer.trainer == nil {
		return 0
	}
	return trainer.trainer.GlobalStep()
}

func (trainer *PolicyTrainer) SaveCheckpoint() (string, error) {
	if trainer == nil {
		return "", fmt.Errorf("policy trainer is nil")
	}
	if trainer.checkpoint == nil {
		return "", nil
	}
	if err := trainer.checkpoint.Save(); err != nil {
		return "", err
	}
	names, err := trainer.checkpoint.ListCheckpoints()
	if err != nil {
		return "", err
	}
	if len(names) == 0 {
		return "", fmt.Errorf("checkpoint save completed but no checkpoint files were listed")
	}
	trainer.LatestCheckpoint = names[len(names)-1]
	return trainer.LatestCheckpoint, nil
}

func RunPolicyStep(batch data.Batch, vocab actionspace.Vocabulary) (PolicyStepResult, error) {
	var result PolicyStepResult
	if batch.Size() == 0 {
		return result, fmt.Errorf("batch is empty")
	}

	trainer, err := NewPolicyTrainer(vocab, DefaultPolicyTrainerConfig())
	if err != nil {
		return result, err
	}
	defer trainer.Close()

	initialLoss, err := trainer.EvalBatch(batch)
	if err != nil {
		return result, fmt.Errorf("initial eval: %w", err)
	}
	trainingLoss, err := trainer.TrainBatch(batch)
	if err != nil {
		return result, fmt.Errorf("train step: %w", err)
	}
	postUpdateLoss, err := trainer.EvalBatch(batch)
	if err != nil {
		return result, fmt.Errorf("post-update eval: %w", err)
	}

	result.ActionCount = trainer.ActionCount
	result.BackendDescription = trainer.BackendDescription
	result.DeviceDescription = trainer.DeviceDescription
	result.InitialLoss = initialLoss
	result.TrainingLoss = trainingLoss
	result.PostUpdateLoss = postUpdateLoss
	result.UpdateCompleted = true
	result.Seed = trainer.Seed
	return result, nil
}

func newPolicyBackend() (backends.Backend, error) {
	xla.EnableAutoInstall(false)

	backend, err := xla.New(policyBackendConfig)
	if err != nil {
		return nil, fmt.Errorf("create XLA CUDA backend: %w", err)
	}
	if err := validatePolicyBackend(backend); err != nil {
		backend.Finalize()
		return nil, err
	}
	return backend, nil
}

func validatePolicyBackend(backend backends.Backend) error {
	if backend == nil {
		return fmt.Errorf("backend is nil")
	}
	if backend.Name() != xla.BackendName {
		return fmt.Errorf("backend is %q, expected %q", backend.Name(), xla.BackendName)
	}
	if description := strings.ToLower(backend.Description()); !strings.Contains(description, policyBackendConfig) {
		return fmt.Errorf("backend description %q does not identify the %q plugin", backend.Description(), policyBackendConfig)
	}
	if got := backend.NumDevices(); got != policyRequiredDeviceCount {
		return fmt.Errorf("XLA CUDA backend exposes %d devices, expected exactly %d", got, policyRequiredDeviceCount)
	}
	return nil
}

func BatchToPolicyTensors(batch data.Batch, vocab actionspace.Vocabulary) (inputs, labels []*tensors.Tensor, err error) {
	if batch.Size() == 0 {
		return nil, nil, fmt.Errorf("batch is empty")
	}
	if len(batch.Inputs) != len(batch.Targets) {
		return nil, nil, fmt.Errorf("batch has %d inputs and %d targets", len(batch.Inputs), len(batch.Targets))
	}
	if err := validateActionVocabulary(vocab); err != nil {
		return nil, nil, err
	}

	stateInputs, err := BatchToPolicyStateTensors(batch)
	if err != nil {
		return nil, nil, err
	}

	fixedFeatures, err := model.FixedActionFeatureMatrix(vocab.Words, len(vocab.Words))
	if err != nil {
		finalizeTensors(stateInputs)
		return nil, nil, fmt.Errorf("build fixed action features: %w", err)
	}
	fixedActionFeatures := tensors.FromFlatDataAndDimensions(
		fixedFeatures,
		len(vocab.Words),
		model.FixedActionFeatureDim,
	)

	teacherTopK, err := BatchTeacherTopKIndices(batch, vocab)
	if err != nil {
		finalizeTensors(stateInputs)
		fixedActionFeatures.MustFinalizeAll()
		return nil, nil, err
	}
	teacherTopKTensor := tensors.FromFlatDataAndDimensions(teacherTopK, batch.Size(), data.TopK)

	inputs = append(stateInputs, fixedActionFeatures)
	labels = []*tensors.Tensor{teacherTopKTensor}
	return inputs, labels, nil
}

func BatchTeacherTopKIndices(batch data.Batch, vocab actionspace.Vocabulary) ([]int32, error) {
	if batch.Size() == 0 {
		return nil, fmt.Errorf("batch is empty")
	}
	if len(batch.Inputs) != len(batch.Targets) {
		return nil, fmt.Errorf("batch has %d inputs and %d targets", len(batch.Inputs), len(batch.Targets))
	}
	if err := validateActionVocabulary(vocab); err != nil {
		return nil, err
	}

	indices := make([]int32, 0, batch.Size()*data.TopK)
	for sampleIndex, target := range batch.Targets {
		for rank, word := range target.TopKGuessWords {
			if word.IsEmpty() {
				return nil, fmt.Errorf("sample %d top_k_guess_words[%d] is empty", sampleIndex, rank)
			}
			actionIndex, exists := vocab.Index[word]
			if !exists {
				return nil, fmt.Errorf("sample %d top_k_guess_words[%d] %q is not in action vocabulary", sampleIndex, rank, word)
			}
			indices = append(indices, int32(actionIndex))
		}
	}
	return indices, nil
}

func evalPolicyLoss(trainer *gomlxtrain.Trainer, batch data.Batch, vocab actionspace.Vocabulary) (float64, error) {
	inputs, labels, err := BatchToPolicyTensors(batch, vocab)
	if err != nil {
		return 0, err
	}
	defer finalizeTensors(inputs)
	defer finalizeTensors(labels)

	metrics, err := trainer.EvalStep(nil, inputs, labels)
	if err != nil {
		return 0, err
	}
	defer finalizeTensors(metrics)

	return scalarTensor(metrics[0])
}

func trainPolicyLoss(trainer *gomlxtrain.Trainer, batch data.Batch, vocab actionspace.Vocabulary) (float64, error) {
	inputs, labels, err := BatchToPolicyTensors(batch, vocab)
	if err != nil {
		return 0, err
	}
	defer finalizeTensors(inputs)
	defer finalizeTensors(labels)

	metrics, err := trainer.TrainStep(nil, inputs, labels)
	if err != nil {
		return 0, err
	}
	defer finalizeTensors(metrics)

	return scalarTensor(metrics[0])
}

func validateActionVocabulary(vocab actionspace.Vocabulary) error {
	actionCount := len(vocab.Words)
	if actionCount == 0 {
		return fmt.Errorf("action vocabulary is empty")
	}
	if actionCount > model.ActionCatalogCapacity {
		return fmt.Errorf("action vocabulary has %d words, maximum is %d", actionCount, model.ActionCatalogCapacity)
	}
	if len(vocab.Index) != actionCount {
		return fmt.Errorf("action vocabulary has %d words and %d index entries", actionCount, len(vocab.Index))
	}
	for index, word := range vocab.Words {
		if word.IsEmpty() {
			return fmt.Errorf("action vocabulary word %d is empty", index)
		}
		gotIndex, exists := vocab.Index[word]
		if !exists {
			return fmt.Errorf("action vocabulary word %d %q is missing from index", index, word)
		}
		if gotIndex != index {
			return fmt.Errorf("action vocabulary word %q has index %d, expected %d", word, gotIndex, index)
		}
	}
	return nil
}

func scalarTensor(tensor *tensors.Tensor) (float64, error) {
	switch tensor.DType() {
	case dtypes.Float32:
		return float64(tensors.ToScalar[float32](tensor)), nil
	case dtypes.Float64:
		return tensors.ToScalar[float64](tensor), nil
	default:
		return 0, fmt.Errorf("scalar tensor has unsupported dtype %s", tensor.DType())
	}
}

func finalizeTensors(tensorsToFinalize []*tensors.Tensor) {
	for _, tensor := range tensorsToFinalize {
		if tensor != nil {
			tensor.MustFinalizeAll()
		}
	}
}
