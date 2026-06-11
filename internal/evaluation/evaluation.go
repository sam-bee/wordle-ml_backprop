package evaluation

import (
	"fmt"

	enginegame "github.com/sam-bee/wordle-ml_game-engine/game"
	enginewords "github.com/sam-bee/wordle-ml_game-engine/words"

	"github.com/sam-bee/wordle-ml_backprop/internal/data"
	"github.com/sam-bee/wordle-ml_backprop/internal/inference"
)

const (
	MaxEpisodeScore = 15
	MaxWordleTurns  = 6
)

type Predictor interface {
	Predict(input data.BatchInput) ([]inference.ScoredAction, error)
}

type Turn struct {
	Guess    data.Word
	Feedback [data.WordLength]data.Feedback
}

type EpisodeResult struct {
	Solution  data.Word `json:"solution"`
	Won       bool      `json:"won"`
	TurnsUsed int       `json:"turns_used"`
	Score     int       `json:"score"`
}

type Summary struct {
	Solutions        int     `json:"solutions"`
	RawScore         int     `json:"raw_score"`
	MaxScore         int     `json:"max_score"`
	ScorePercent     float64 `json:"score_percent"`
	Wins             int     `json:"wins"`
	Losses           int     `json:"losses"`
	AverageWinTurns  float64 `json:"average_turns_on_wins"`
	Turns1           int     `json:"turns_1"`
	Turns2           int     `json:"turns_2"`
	Turns3           int     `json:"turns_3"`
	Turns4           int     `json:"turns_4"`
	Turns5           int     `json:"turns_5"`
	Turns6           int     `json:"turns_6"`
	EpisodeMaxScore  int     `json:"episode_max_score"`
	Selection        string  `json:"selection"`
	ValidationSource string  `json:"validation_source,omitempty"`
}

func UniqueValidationSolutions(split *data.Split) ([]data.Word, error) {
	if split == nil {
		return nil, fmt.Errorf("validation split is nil")
	}
	seen := make(map[data.Word]bool)
	solutions := make([]data.Word, 0, split.Metadata.SolutionCount)
	for _, sample := range split.Samples {
		solution := sample.SolutionWord
		if solution.IsEmpty() || seen[solution] {
			continue
		}
		seen[solution] = true
		solutions = append(solutions, solution)
	}
	if len(solutions) == 0 {
		return nil, fmt.Errorf("validation split contains no non-empty solution words")
	}
	return solutions, nil
}

func RunEpisode(solution data.Word, maxTurns int, predictor Predictor) (EpisodeResult, error) {
	if solution.IsEmpty() {
		return EpisodeResult{}, fmt.Errorf("solution word is empty")
	}
	if maxTurns <= 0 || maxTurns > MaxWordleTurns {
		return EpisodeResult{}, fmt.Errorf("max turns must be in 1..%d, got %d", MaxWordleTurns, maxTurns)
	}
	if predictor == nil {
		return EpisodeResult{}, fmt.Errorf("predictor is nil")
	}

	engineSolution, err := enginewords.NewWord(solution.String())
	if err != nil {
		return EpisodeResult{}, fmt.Errorf("solution %s: %w", solution, err)
	}

	var turns []Turn
	guessed := make(map[data.Word]bool)
	for turnIndex := 0; turnIndex < maxTurns; turnIndex++ {
		input, err := BatchInputFromTurns(turns)
		if err != nil {
			return EpisodeResult{}, err
		}
		ranked, err := predictor.Predict(input)
		if err != nil {
			return EpisodeResult{}, fmt.Errorf("turn %d inference: %w", turnIndex+1, err)
		}
		guess, _, err := inference.FirstUnguessed(ranked, guessed)
		if err != nil {
			return EpisodeResult{}, err
		}
		guessed[guess] = true

		engineGuess, err := enginewords.NewWord(guess.String())
		if err != nil {
			return EpisodeResult{}, fmt.Errorf("model selected invalid action %s: %w", guess, err)
		}
		engineFeedback := enginegame.GetFeedback(engineSolution, engineGuess)
		feedback := engineFeedback.String()
		if feedback == "GGGGG" {
			turnsUsed := turnIndex + 1
			return EpisodeResult{
				Solution:  solution,
				Won:       true,
				TurnsUsed: turnsUsed,
				Score:     ScoreEpisode(true, turnsUsed),
			}, nil
		}
		parsedFeedback, err := ParseFeedback(feedback)
		if err != nil {
			return EpisodeResult{}, err
		}
		turns = append(turns, Turn{Guess: guess, Feedback: parsedFeedback})
	}

	return EpisodeResult{
		Solution:  solution,
		Won:       false,
		TurnsUsed: maxTurns,
		Score:     0,
	}, nil
}

func ScoreEpisode(won bool, turnsUsed int) int {
	if !won {
		return 0
	}
	return 10 + (MaxWordleTurns - turnsUsed)
}

func Summarize(results []EpisodeResult, validationSource string) Summary {
	summary := Summary{
		Solutions:        len(results),
		MaxScore:         len(results) * MaxEpisodeScore,
		EpisodeMaxScore:  MaxEpisodeScore,
		Selection:        "highest-scoring unguessed action; no candidate-list filtering",
		ValidationSource: validationSource,
	}
	var winTurnSum int
	var histogram [MaxWordleTurns + 1]int
	for _, result := range results {
		summary.RawScore += result.Score
		if result.Won {
			summary.Wins++
			winTurnSum += result.TurnsUsed
			if result.TurnsUsed >= 1 && result.TurnsUsed <= MaxWordleTurns {
				histogram[result.TurnsUsed]++
			}
		} else {
			summary.Losses++
		}
	}
	if summary.MaxScore > 0 {
		summary.ScorePercent = 100 * float64(summary.RawScore) / float64(summary.MaxScore)
	}
	if summary.Wins > 0 {
		summary.AverageWinTurns = float64(winTurnSum) / float64(summary.Wins)
	}
	summary.Turns1 = histogram[1]
	summary.Turns2 = histogram[2]
	summary.Turns3 = histogram[3]
	summary.Turns4 = histogram[4]
	summary.Turns5 = histogram[5]
	summary.Turns6 = histogram[6]
	return summary
}

func BatchInputFromTurns(turns []Turn) (data.BatchInput, error) {
	var input data.BatchInput
	if len(turns) > data.MaxTurns {
		return input, fmt.Errorf("state has %d previous turns, max supported is %d", len(turns), data.MaxTurns)
	}
	input.TurnDepth = uint8(len(turns))
	for index, turn := range turns {
		input.PreviousGuessWords[index] = turn.Guess
		input.PreviousFeedback[index] = turn.Feedback
	}
	return input, nil
}

func ParseFeedback(value string) ([data.WordLength]data.Feedback, error) {
	var feedback [data.WordLength]data.Feedback
	if len(value) != data.WordLength {
		return feedback, fmt.Errorf("feedback %q has length %d, expected %d", value, len(value), data.WordLength)
	}
	for index := range value {
		switch value[index] {
		case 'G':
			feedback[index] = data.FeedbackGreen
		case 'Y':
			feedback[index] = data.FeedbackYellow
		case '-':
			feedback[index] = data.FeedbackGrey
		default:
			return feedback, fmt.Errorf("feedback %q contains unsupported byte %d at position %d", value, value[index], index)
		}
	}
	return feedback, nil
}
