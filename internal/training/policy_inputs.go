package training

import (
	"fmt"

	"github.com/gomlx/gomlx/pkg/core/tensors"

	"github.com/sam-bee/wordle-ml_backprop/internal/data"
	"github.com/sam-bee/wordle-ml_backprop/internal/model"
)

const PolicyStateInputCount = 3

type PolicyStateFeatures struct {
	TurnFeatures  []float32
	OccupiedTurns []float32
	VirginGrid    []float32
	BatchSize     int
}

func BatchToPolicyStateTensors(batch data.Batch) ([]*tensors.Tensor, error) {
	features, err := BuildPolicyStateFeatures(batch)
	if err != nil {
		return nil, err
	}

	return []*tensors.Tensor{
		tensors.FromFlatDataAndDimensions(features.TurnFeatures, features.BatchSize, data.MaxTurns, model.RawTurnFeatureCount),
		tensors.FromFlatDataAndDimensions(features.OccupiedTurns, features.BatchSize, data.MaxTurns),
		tensors.FromFlatDataAndDimensions(features.VirginGrid, features.BatchSize, 1),
	}, nil
}

func BuildPolicyStateFeatures(batch data.Batch) (PolicyStateFeatures, error) {
	var features PolicyStateFeatures
	if batch.Size() == 0 {
		return features, fmt.Errorf("batch is empty")
	}

	batchSize := batch.Size()
	features = PolicyStateFeatures{
		TurnFeatures:  make([]float32, batchSize*data.MaxTurns*model.RawTurnFeatureCount),
		OccupiedTurns: make([]float32, batchSize*data.MaxTurns),
		VirginGrid:    make([]float32, batchSize),
		BatchSize:     batchSize,
	}

	for sampleIndex, input := range batch.Inputs {
		if err := appendPolicyStateFeatures(features, sampleIndex, input); err != nil {
			return PolicyStateFeatures{}, fmt.Errorf("sample %d: %w", sampleIndex, err)
		}
	}
	return features, nil
}

func appendPolicyStateFeatures(features PolicyStateFeatures, sampleIndex int, input data.BatchInput) error {
	turnDepth := int(input.TurnDepth)
	if turnDepth > data.MaxTurns {
		return fmt.Errorf("turn_depth is %d, expected 0..%d", input.TurnDepth, data.MaxTurns)
	}
	if turnDepth == 0 {
		features.VirginGrid[sampleIndex] = 1
	}

	for turn := 0; turn < data.MaxTurns; turn++ {
		if turn >= turnDepth {
			if !input.PreviousGuessWords[turn].IsEmpty() {
				return fmt.Errorf("previous_guess_words[%d] is not empty for unused turn", turn)
			}
			continue
		}

		word := input.PreviousGuessWords[turn]
		feedback := input.PreviousFeedback[turn]
		if word.IsEmpty() {
			return fmt.Errorf("previous_guess_words[%d] is empty for used turn", turn)
		}
		if isWinningFeedback(feedback) {
			return fmt.Errorf("previous_feedback[%d] is terminal all-green feedback", turn)
		}

		features.OccupiedTurns[sampleIndex*data.MaxTurns+turn] = 1
		if err := encodePolicyTurn(features.TurnFeatures, sampleIndex, turn, word, feedback); err != nil {
			return err
		}
	}
	return nil
}

func encodePolicyTurn(features []float32, sampleIndex, turn int, word data.Word, feedback [data.WordLength]data.Feedback) error {
	turnOffset := ((sampleIndex * data.MaxTurns) + turn) * model.RawTurnFeatureCount
	feedbackOffset := turnOffset + data.WordLength*model.LetterAlphabetSize

	for pos, b := range word {
		if b < 'A' || b > 'Z' {
			return fmt.Errorf("previous_guess_words[%d][%d] contains non-uppercase ASCII byte %d", turn, pos, b)
		}
		letterIndex := int(b - 'A')
		features[turnOffset+pos*model.LetterAlphabetSize+letterIndex] = 1

		feedbackIndex, err := modelFeedbackIndex(feedback[pos])
		if err != nil {
			return fmt.Errorf("previous_feedback[%d][%d]: %w", turn, pos, err)
		}
		features[feedbackOffset+pos*model.FeedbackAlphabetSize+feedbackIndex] = 1
	}
	return nil
}

func modelFeedbackIndex(feedback data.Feedback) (int, error) {
	switch feedback {
	case data.FeedbackGreen:
		return 0, nil
	case data.FeedbackYellow:
		return 1, nil
	case data.FeedbackGrey:
		return 2, nil
	default:
		return 0, fmt.Errorf("feedback is %d, expected grey/yellow/green", feedback)
	}
}

func isWinningFeedback(feedback [data.WordLength]data.Feedback) bool {
	for _, value := range feedback {
		if value != data.FeedbackGreen {
			return false
		}
	}
	return true
}
