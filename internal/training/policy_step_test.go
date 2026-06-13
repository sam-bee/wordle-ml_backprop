package training

import (
	"math"
	"strings"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/tensors"

	"github.com/sam-bee/wordle-ml_backprop/internal/actionspace"
	"github.com/sam-bee/wordle-ml_backprop/internal/data"
	"github.com/sam-bee/wordle-ml_backprop/internal/model"
)

func TestBatchToPolicyTensorsShapes(t *testing.T) {
	batch := trainingFixtureBatch(t, 3)
	vocab := trainingFixtureVocabulary(t)

	inputs, labels, err := BatchToPolicyTensors(batch, vocab)
	if err != nil {
		t.Fatalf("BatchToPolicyTensors() error = %v", err)
	}
	defer finalizeTensors(inputs)
	defer finalizeTensors(labels)

	if len(inputs) != model.PolicyModelInputCount {
		t.Fatalf("len(inputs) = %d, want %d", len(inputs), model.PolicyModelInputCount)
	}
	if len(labels) != 1 {
		t.Fatalf("len(labels) = %d, want 1", len(labels))
	}
	if got := inputs[0].Shape().Dimensions; !equalInts(got, []int{3, data.MaxTurns, model.RawTurnFeatureCount}) {
		t.Fatalf("turn feature dimensions = %v, want [3 %d %d]", got, data.MaxTurns, model.RawTurnFeatureCount)
	}
	if got := inputs[1].Shape().Dimensions; !equalInts(got, []int{3, data.MaxTurns}) {
		t.Fatalf("occupied-turn dimensions = %v, want [3 %d]", got, data.MaxTurns)
	}
	if got := inputs[2].Shape().Dimensions; !equalInts(got, []int{3, 1}) {
		t.Fatalf("virgin-grid dimensions = %v, want [3 1]", got)
	}
	if got := inputs[3].Shape().Dimensions; !equalInts(got, []int{len(vocab.Words), model.FixedActionFeatureDim}) {
		t.Fatalf("fixed action feature dimensions = %v, want [%d %d]", got, len(vocab.Words), model.FixedActionFeatureDim)
	}
	if got := labels[0].Shape().DType; got != dtypes.Int32 {
		t.Fatalf("label dtype = %s, want Int32", got)
	}
	if got := labels[0].Shape().Dimensions; !equalInts(got, []int{3, data.TopK}) {
		t.Fatalf("label dimensions = %v, want [3 %d]", got, data.TopK)
	}

	teacherTopK := tensors.MustCopyFlatData[int32](labels[0])
	for _, index := range teacherTopK {
		if index != 0 {
			t.Fatalf("teacher action index = %d, want 0", index)
		}
	}
}

func TestBatchTeacherTopKIndicesRejectsMissingWord(t *testing.T) {
	batch := trainingFixtureBatch(t, 1)
	vocab := trainingFixtureVocabulary(t)
	batch.Targets[0].TopKGuessWords[0] = trainingFixtureWord(t, "ZZZZZ")

	_, err := BatchTeacherTopKIndices(batch, vocab)
	if err == nil || !strings.Contains(err.Error(), "not in action vocabulary") {
		t.Fatalf("BatchTeacherTopKIndices() error = %v, want missing word error", err)
	}
}

func TestPolicyTrainerConfigValidation(t *testing.T) {
	if err := DefaultPolicyTrainerConfig().Validate(); err != nil {
		t.Fatalf("default config validation failed: %v", err)
	}

	for _, learningRate := range []float64{0, -0.1, math.NaN(), math.Inf(1)} {
		config := DefaultPolicyTrainerConfig()
		config.LearningRate = learningRate
		if err := config.Validate(); err == nil {
			t.Fatalf("Validate() with learning rate %v succeeded, want error", learningRate)
		}
	}
}

func TestPolicyTrainerUsesConfiguredSeed(t *testing.T) {
	config := DefaultPolicyTrainerConfig()
	config.Seed = 12345

	trainer, err := NewPolicyTrainer(trainingFixtureVocabulary(t), config)
	if err != nil {
		t.Fatalf("NewPolicyTrainer() error = %v", err)
	}
	defer trainer.Close()

	if trainer.Seed != config.Seed {
		t.Fatalf("Seed = %d, want %d", trainer.Seed, config.Seed)
	}
}

func TestPolicyLearningRateForStep(t *testing.T) {
	const learningRate = 0.05

	if got := policyLearningRateForStep(learningRate, false, 100); got != learningRate {
		t.Fatalf("fixed learning rate = %g, want %g", got, learningRate)
	}
	if got := policyLearningRateForStep(learningRate, true, 0); got != learningRate {
		t.Fatalf("decayed learning rate at step 0 = %g, want %g", got, learningRate)
	}
	if got := policyLearningRateForStep(learningRate, true, 4); math.Abs(got-0.025) > 1e-12 {
		t.Fatalf("decayed learning rate at step 4 = %g, want 0.025", got)
	}
}

func TestRunPolicyStep(t *testing.T) {
	vocab := trainingFixtureVocabulary(t)
	result, err := RunPolicyStep(trainingFixtureBatch(t, 4), vocab)
	if err != nil {
		t.Fatalf("RunPolicyStep() error = %v", err)
	}
	if result.ActionCount != len(vocab.Words) {
		t.Fatalf("ActionCount = %d, want %d", result.ActionCount, len(vocab.Words))
	}
	if !result.UpdateCompleted {
		t.Fatal("UpdateCompleted = false, want true")
	}
	if !strings.Contains(result.BackendDescription, "cuda") {
		t.Fatalf("BackendDescription = %q, want CUDA backend", result.BackendDescription)
	}
	if result.DeviceDescription == "" {
		t.Fatal("DeviceDescription is empty")
	}
	for name, value := range map[string]float64{
		"InitialLoss":    result.InitialLoss,
		"TrainingLoss":   result.TrainingLoss,
		"PostUpdateLoss": result.PostUpdateLoss,
	} {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			t.Fatalf("%s = %g, want finite", name, value)
		}
	}
}

func trainingFixtureVocabulary(t *testing.T) actionspace.Vocabulary {
	t.Helper()

	words := []data.Word{
		trainingFixtureWord(t, "TRACE"),
		trainingFixtureWord(t, "SLATE"),
		trainingFixtureWord(t, "CRASS"),
		trainingFixtureWord(t, "AARGH"),
	}
	index := make(map[data.Word]int, len(words))
	for i, word := range words {
		index[word] = i
	}
	return actionspace.Vocabulary{Words: words, Index: index}
}
