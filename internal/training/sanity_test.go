package training

import (
	"math"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"

	"github.com/sam-bee/wordle-ml_backprop/internal/data"
	"github.com/sam-bee/wordle-ml_backprop/internal/model"
)

func TestBatchToTensorsShapes(t *testing.T) {
	batch := trainingFixtureBatch(t, 3)

	inputs, labels, err := BatchToTensors(batch)
	if err != nil {
		t.Fatalf("BatchToTensors() error = %v", err)
	}
	defer finalizeTensors(inputs)
	defer finalizeTensors(labels)

	if len(inputs) != 1 {
		t.Fatalf("len(inputs) = %d, want 1", len(inputs))
	}
	if len(labels) != 1 {
		t.Fatalf("len(labels) = %d, want 1", len(labels))
	}
	if got := inputs[0].Shape().DType; got != dtypes.Float32 {
		t.Fatalf("input dtype = %s, want Float32", got)
	}
	if got := inputs[0].Shape().Dimensions; !equalInts(got, []int{3, model.InputFeatureCount}) {
		t.Fatalf("input dimensions = %v, want [3 %d]", got, model.InputFeatureCount)
	}
	if got := labels[0].Shape().Dimensions; !equalInts(got, []int{3, model.OutputCount}) {
		t.Fatalf("label dimensions = %v, want [3 %d]", got, model.OutputCount)
	}
}

func TestRunSanityStep(t *testing.T) {
	result, err := RunSanityStep(trainingFixtureBatch(t, 4))
	if err != nil {
		t.Fatalf("RunSanityStep() error = %v", err)
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

func trainingFixtureBatch(t *testing.T, count int) data.Batch {
	t.Helper()

	samples := make([]data.Sample, count)
	for i := range samples {
		samples[i].TurnDepth = 1
		samples[i].PreviousGuessWords[0] = trainingFixtureWord(t, "SLATE")
		samples[i].PreviousFeedback[0] = [data.WordLength]data.Feedback{
			data.FeedbackGrey,
			data.FeedbackYellow,
			data.FeedbackGreen,
			data.FeedbackGrey,
			data.FeedbackYellow,
		}
		samples[i].ShortlistSizeBefore = uint16(100 + i)
		for rank := range samples[i].TopKReductionRatios {
			samples[i].TopKGuessWords[rank] = trainingFixtureWord(t, "TRACE")
			samples[i].TopKReductionRatios[rank] = float32(rank+1) / data.TopK
			samples[i].TopKWorstCaseSizes[rank] = 5
		}
	}
	return data.BuildBatch(samples)
}

func trainingFixtureWord(t *testing.T, value string) data.Word {
	t.Helper()
	if len(value) != data.WordLength {
		t.Fatalf("fixture word %q has length %d, want %d", value, len(value), data.WordLength)
	}
	var word data.Word
	copy(word[:], value)
	return word
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
