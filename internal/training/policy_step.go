package training

import (
	"fmt"
	"math"
	"strings"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/xla"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	gomlxtrain "github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"

	"github.com/sam-bee/wordle-ml_backprop/internal/actionspace"
	"github.com/sam-bee/wordle-ml_backprop/internal/data"
	"github.com/sam-bee/wordle-ml_backprop/internal/model"
)

const (
	DefaultPolicyLearningRate = 0.05
	DefaultPolicySeed         = int64(1)

	policyBackendConfig       = "cuda"
	policyRequiredDeviceCount = 1
)

type PolicyTrainerConfig struct {
	LearningRate float64
	Seed         int64
}

func DefaultPolicyTrainerConfig() PolicyTrainerConfig {
	return PolicyTrainerConfig{
		LearningRate: DefaultPolicyLearningRate,
		Seed:         DefaultPolicySeed,
	}
}

func (config PolicyTrainerConfig) Validate() error {
	if math.IsNaN(config.LearningRate) || math.IsInf(config.LearningRate, 0) || config.LearningRate <= 0 {
		return fmt.Errorf("learning rate must be finite and > 0, got %v", config.LearningRate)
	}
	return nil
}

type PolicyTrainer struct {
	vocab              actionspace.Vocabulary
	backend            backends.Backend
	trainer            *gomlxtrain.Trainer
	ActionCount        int
	BackendDescription string
	DeviceDescription  string
}

type PolicyStepResult struct {
	ActionCount        int
	BackendDescription string
	DeviceDescription  string
	InitialLoss        float64
	TrainingLoss       float64
	PostUpdateLoss     float64
	UpdateCompleted    bool
}

func NewPolicyTrainer(vocab actionspace.Vocabulary, config PolicyTrainerConfig) (*PolicyTrainer, error) {
	if err := validateActionVocabulary(vocab); err != nil {
		return nil, err
	}
	if err := config.Validate(); err != nil {
		return nil, err
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

	optimizer := optimizers.StochasticGradientDescent().
		WithLearningRate(config.LearningRate).
		WithDecay(false).
		Done()
	gomlxTrainer := gomlxtrain.NewTrainer(
		backend,
		ctx,
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
		ActionCount:        len(vocab.Words),
		BackendDescription: backend.Description(),
		DeviceDescription:  backend.DeviceDescription(0),
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
