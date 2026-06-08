package training

import (
	"testing"

	"github.com/sam-bee/wordle-ml_backprop/internal/data"
)

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
		for rank := range samples[i].TopKGuessWords {
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
