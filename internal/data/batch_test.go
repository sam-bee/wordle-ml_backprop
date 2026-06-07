package data

import "testing"

func TestBatchIteratorIncludesFinalPartialBatch(t *testing.T) {
	split := &Split{Samples: batchFixtureSamples(t, 5)}
	it, err := NewBatchIterator(split, 2)
	if err != nil {
		t.Fatalf("NewBatchIterator() error = %v", err)
	}

	var sizes []int
	var firstDepths []uint8
	for {
		batch, ok := it.Next()
		if !ok {
			break
		}
		sizes = append(sizes, batch.Size())
		firstDepths = append(firstDepths, batch.Inputs[0].TurnDepth)
		if len(batch.Inputs) != len(batch.Targets) {
			t.Fatalf("len(batch.Inputs) = %d, len(batch.Targets) = %d", len(batch.Inputs), len(batch.Targets))
		}
	}

	if want := []int{2, 2, 1}; !equalInts(sizes, want) {
		t.Fatalf("batch sizes = %v, want %v", sizes, want)
	}
	if want := []uint8{1, 3, 5}; !equalUint8s(firstDepths, want) {
		t.Fatalf("first depths = %v, want %v", firstDepths, want)
	}
}

func TestBuildBatchDimensionsAndTargets(t *testing.T) {
	samples := batchFixtureSamples(t, 3)
	batch := BuildBatch(samples)

	if batch.Size() != 3 {
		t.Fatalf("batch.Size() = %d, want 3", batch.Size())
	}
	if batch.Dimensions != (BatchDimensions{BatchSize: 3, MaxTurns: MaxTurns, WordLength: WordLength, TopK: TopK}) {
		t.Fatalf("batch.Dimensions = %+v", batch.Dimensions)
	}
	if batch.Inputs[0].TurnDepth != samples[0].TurnDepth {
		t.Fatalf("batch.Inputs[0].TurnDepth = %d, want %d", batch.Inputs[0].TurnDepth, samples[0].TurnDepth)
	}
	if batch.Targets[0].TopKGuessWords[0] != samples[0].TopKGuessWords[0] {
		t.Fatalf("batch target word = %q, want %q", batch.Targets[0].TopKGuessWords[0], samples[0].TopKGuessWords[0])
	}
}

func TestNewBatchIteratorRejectsInvalidBatchSize(t *testing.T) {
	_, err := NewBatchIterator(&Split{}, 0)
	if err == nil {
		t.Fatal("NewBatchIterator() error = nil, want invalid batch size error")
	}
}

func batchFixtureSamples(t *testing.T, count int) []Sample {
	t.Helper()

	samples := make([]Sample, count)
	for i := range samples {
		depth := uint8(i + 1)
		if depth > MaxTurns {
			depth = MaxTurns
		}
		samples[i] = Sample{
			TurnDepth:           depth,
			ShortlistSizeBefore: uint16(10 + i),
		}
		samples[i].PreviousGuessWords[0] = fixtureWord(t, "SLATE")
		samples[i].PreviousFeedback[0] = [WordLength]Feedback{FeedbackGrey, FeedbackYellow, FeedbackGreen, FeedbackGrey, FeedbackYellow}
		samples[i].TopKGuessWords[0] = fixtureWord(t, "TRACE")
		samples[i].TopKReductionRatios[0] = 0.5
		samples[i].TopKWorstCaseSizes[0] = 5
	}
	return samples
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

func equalUint8s(a, b []uint8) bool {
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
