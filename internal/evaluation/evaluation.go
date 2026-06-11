package evaluation

import (
	"fmt"

	enginegame "github.com/sam-bee/wordle-ml_game-engine/game"
	enginewords "github.com/sam-bee/wordle-ml_game-engine/words"

	"github.com/sam-bee/wordle-ml_backprop/internal/data"
	"github.com/sam-bee/wordle-ml_backprop/internal/inference"
)

const (
	MaxEpisodeScore        = 15
	MaxWordleTurns         = 6
	LossGreenPositionScore = 0.5
	LossYellowLetterScore  = 0.25
	LossGreyLetterScore    = 1.0 / 16.0
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
	Score     float64   `json:"score"`
}

type Summary struct {
	Solutions        int     `json:"solutions"`
	RawScore         float64 `json:"raw_score"`
	MaxScore         float64 `json:"max_score"`
	ScorePercent     float64 `json:"score_percent"`
	WinScore         float64 `json:"win_score"`
	LossCreditScore  float64 `json:"loss_credit_score"`
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
		Score:     LossFeedbackScore(turns),
	}, nil
}

func ScoreEpisode(won bool, turnsUsed int) float64 {
	if !won {
		return 0
	}
	return float64(10 + (MaxWordleTurns - turnsUsed))
}

func LossFeedbackScore(turns []Turn) float64 {
	greenPositions := make(map[int]bool)
	greenLetters := make(map[byte]bool)
	yellowLetters := make(map[byte]bool)
	greyLetters := make(map[byte]bool)

	for _, turn := range turns {
		for position, feedback := range turn.Feedback {
			letter := turn.Guess[position]
			switch feedback {
			case data.FeedbackGreen:
				greenPositions[position] = true
				greenLetters[letter] = true
			case data.FeedbackYellow:
				yellowLetters[letter] = true
			case data.FeedbackGrey:
				greyLetters[letter] = true
			}
		}
	}

	score := float64(len(greenPositions)) * LossGreenPositionScore
	for letter := range yellowLetters {
		if !greenLetters[letter] {
			score += LossYellowLetterScore
		}
	}
	for letter := range greyLetters {
		if !greenLetters[letter] && !yellowLetters[letter] {
			score += LossGreyLetterScore
		}
	}
	return score
}

func Summarize(results []EpisodeResult, validationSource string) Summary {
	summary := Summary{
		Solutions:        len(results),
		MaxScore:         float64(len(results) * MaxEpisodeScore),
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
			summary.WinScore += result.Score
			winTurnSum += result.TurnsUsed
			if result.TurnsUsed >= 1 && result.TurnsUsed <= MaxWordleTurns {
				histogram[result.TurnsUsed]++
			}
		} else {
			summary.Losses++
			summary.LossCreditScore += result.Score
		}
	}
	if summary.MaxScore > 0 {
		summary.ScorePercent = 100 * summary.RawScore / summary.MaxScore
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
