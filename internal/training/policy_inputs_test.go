package training

import (
	"strings"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/tensors"

	"github.com/sam-bee/wordle-ml_backprop/internal/data"
	"github.com/sam-bee/wordle-ml_backprop/internal/model"
)

func TestBuildPolicyStateFeatures(t *testing.T) {
	batch := data.BuildBatch([]data.Sample{
		{TurnDepth: 0},
		policyInputFixtureSample(t),
	})

	features, err := BuildPolicyStateFeatures(batch)
	if err != nil {
		t.Fatalf("BuildPolicyStateFeatures() error = %v", err)
	}
	if features.BatchSize != 2 {
		t.Fatalf("BatchSize = %d, want 2", features.BatchSize)
	}
	if got := features.VirginGrid; !equalFloat32s(got, []float32{1, 0}) {
		t.Fatalf("VirginGrid = %v, want [1 0]", got)
	}
	if got := features.OccupiedTurns; !equalFloat32s(got, []float32{0, 0, 0, 0, 0, 1, 1, 0, 0, 0}) {
		t.Fatalf("OccupiedTurns = %v, want empty sample then two occupied turns", got)
	}

	if got := sumPolicyTurn(features.TurnFeatures, 0, 0); got != 0 {
		t.Fatalf("empty sample turn feature sum = %g, want 0", got)
	}
	if got := sumPolicyTurn(features.TurnFeatures, 1, 2); got != 0 {
		t.Fatalf("unused turn feature sum = %g, want 0", got)
	}
	if got := sumPolicyTurn(features.TurnFeatures, 1, 0); got != 10 {
		t.Fatalf("occupied turn feature sum = %g, want 10", got)
	}

	for _, index := range []int{0, 27, 54, 81, 108, 132, 134, 136, 141, 143} {
		if got := policyTurnFeature(features.TurnFeatures, 1, 0, index); got != 1 {
			t.Fatalf("turn feature[%d] = %g, want 1", index, got)
		}
	}
}

func TestBatchToPolicyStateTensorsShapes(t *testing.T) {
	batch := data.BuildBatch([]data.Sample{
		{TurnDepth: 0},
		policyInputFixtureSample(t),
	})

	inputs, err := BatchToPolicyStateTensors(batch)
	if err != nil {
		t.Fatalf("BatchToPolicyStateTensors() error = %v", err)
	}
	defer finalizeTensors(inputs)

	if len(inputs) != PolicyStateInputCount {
		t.Fatalf("len(inputs) = %d, want %d", len(inputs), PolicyStateInputCount)
	}
	for i, input := range inputs {
		if got := input.Shape().DType; got != dtypes.Float32 {
			t.Fatalf("inputs[%d] dtype = %s, want Float32", i, got)
		}
	}
	if got := inputs[0].Shape().Dimensions; !equalInts(got, []int{2, data.MaxTurns, model.RawTurnFeatureCount}) {
		t.Fatalf("turn feature dimensions = %v, want [2 %d %d]", got, data.MaxTurns, model.RawTurnFeatureCount)
	}
	if got := inputs[1].Shape().Dimensions; !equalInts(got, []int{2, data.MaxTurns}) {
		t.Fatalf("occupied-turn dimensions = %v, want [2 %d]", got, data.MaxTurns)
	}
	if got := inputs[2].Shape().Dimensions; !equalInts(got, []int{2, 1}) {
		t.Fatalf("virgin-grid dimensions = %v, want [2 1]", got)
	}

	virginGrid := tensors.MustCopyFlatData[float32](inputs[2])
	if !equalFloat32s(virginGrid, []float32{1, 0}) {
		t.Fatalf("virgin-grid tensor = %v, want [1 0]", virginGrid)
	}
}

func TestBuildPolicyStateFeaturesRejectsInvalidFeedback(t *testing.T) {
	sample := policyInputFixtureSample(t)
	sample.PreviousFeedback[0][0] = data.Feedback(99)

	_, err := BuildPolicyStateFeatures(data.BuildBatch([]data.Sample{sample}))
	if err == nil || !strings.Contains(err.Error(), "previous_feedback[0][0]") {
		t.Fatalf("BuildPolicyStateFeatures() error = %v, want invalid feedback location", err)
	}
}

func TestBuildPolicyStateFeaturesRejectsTerminalWin(t *testing.T) {
	sample := policyInputFixtureSample(t)
	sample.TurnDepth = 1
	sample.PreviousFeedback[0] = [data.WordLength]data.Feedback{
		data.FeedbackGreen,
		data.FeedbackGreen,
		data.FeedbackGreen,
		data.FeedbackGreen,
		data.FeedbackGreen,
	}

	_, err := BuildPolicyStateFeatures(data.BuildBatch([]data.Sample{sample}))
	if err == nil || !strings.Contains(err.Error(), "terminal all-green") {
		t.Fatalf("BuildPolicyStateFeatures() error = %v, want terminal all-green error", err)
	}
}

func policyInputFixtureSample(t *testing.T) data.Sample {
	t.Helper()

	sample := data.Sample{TurnDepth: 2}
	sample.PreviousGuessWords[0] = trainingFixtureWord(t, "ABCDE")
	sample.PreviousFeedback[0] = [data.WordLength]data.Feedback{
		data.FeedbackGrey,
		data.FeedbackYellow,
		data.FeedbackGreen,
		data.FeedbackGrey,
		data.FeedbackYellow,
	}
	sample.PreviousGuessWords[1] = trainingFixtureWord(t, "CRASS")
	sample.PreviousFeedback[1] = [data.WordLength]data.Feedback{
		data.FeedbackYellow,
		data.FeedbackGrey,
		data.FeedbackYellow,
		data.FeedbackGreen,
		data.FeedbackGrey,
	}
	return sample
}

func policyTurnFeature(features []float32, sampleIndex, turn, index int) float32 {
	offset := ((sampleIndex * data.MaxTurns) + turn) * model.RawTurnFeatureCount
	return features[offset+index]
}

func sumPolicyTurn(features []float32, sampleIndex, turn int) float32 {
	offset := ((sampleIndex * data.MaxTurns) + turn) * model.RawTurnFeatureCount
	var sum float32
	for _, value := range features[offset : offset+model.RawTurnFeatureCount] {
		sum += value
	}
	return sum
}

func equalFloat32s(a, b []float32) bool {
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
