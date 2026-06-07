package data

import "fmt"

type BatchInput struct {
	TurnDepth           uint8
	PreviousGuessWords  [MaxTurns]Word
	PreviousFeedback    [MaxTurns][WordLength]Feedback
	ShortlistSizeBefore uint16
}

type BatchTarget struct {
	TopKGuessWords      [TopK]Word
	TopKReductionRatios [TopK]float32
	TopKWorstCaseSizes  [TopK]uint16
}

type BatchDimensions struct {
	BatchSize  int
	MaxTurns   int
	WordLength int
	TopK       int
}

type Batch struct {
	Inputs     []BatchInput
	Targets    []BatchTarget
	Dimensions BatchDimensions
}

func (b Batch) Size() int {
	return len(b.Inputs)
}

type BatchIterator struct {
	samples   []Sample
	batchSize int
	next      int
}

func NewBatchIterator(split *Split, batchSize int) (*BatchIterator, error) {
	if split == nil {
		return nil, fmt.Errorf("split is nil")
	}
	if batchSize <= 0 {
		return nil, fmt.Errorf("batch size must be greater than 0, got %d", batchSize)
	}
	return &BatchIterator{
		samples:   split.Samples,
		batchSize: batchSize,
	}, nil
}

// Next returns sequential batches and includes the final partial batch.
func (it *BatchIterator) Next() (Batch, bool) {
	if it == nil || it.next >= len(it.samples) {
		return Batch{}, false
	}

	end := it.next + it.batchSize
	if end > len(it.samples) {
		end = len(it.samples)
	}

	batch := BuildBatch(it.samples[it.next:end])
	it.next = end
	return batch, true
}

func BuildBatch(samples []Sample) Batch {
	inputs := make([]BatchInput, len(samples))
	targets := make([]BatchTarget, len(samples))
	for i, sample := range samples {
		inputs[i] = BatchInput{
			TurnDepth:           sample.TurnDepth,
			PreviousGuessWords:  sample.PreviousGuessWords,
			PreviousFeedback:    sample.PreviousFeedback,
			ShortlistSizeBefore: sample.ShortlistSizeBefore,
		}
		targets[i] = BatchTarget{
			TopKGuessWords:      sample.TopKGuessWords,
			TopKReductionRatios: sample.TopKReductionRatios,
			TopKWorstCaseSizes:  sample.TopKWorstCaseSizes,
		}
	}

	return Batch{
		Inputs:  inputs,
		Targets: targets,
		Dimensions: BatchDimensions{
			BatchSize:  len(samples),
			MaxTurns:   MaxTurns,
			WordLength: WordLength,
			TopK:       TopK,
		},
	}
}
