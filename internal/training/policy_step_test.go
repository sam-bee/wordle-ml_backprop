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
