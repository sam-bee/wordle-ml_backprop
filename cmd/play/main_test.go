package main

import (
	"testing"

	"github.com/sam-bee/wordle-ml_backprop/internal/data"
)

func TestParseFeedback(t *testing.T) {
	feedback, err := parseFeedback("GY--G")
	if err != nil {
		t.Fatalf("parseFeedback() error = %v", err)
	}

	want := [data.WordLength]data.Feedback{
		data.FeedbackGreen,
		data.FeedbackYellow,
		data.FeedbackGrey,
		data.FeedbackGrey,
		data.FeedbackGreen,
	}
	if feedback != want {
		t.Fatalf("feedback = %v, want %v", feedback, want)
	}
}

func TestParseSolutionUppercases(t *testing.T) {
	solution, err := parseSolution("chant")
	if err != nil {
		t.Fatalf("parseSolution() error = %v", err)
	}
	if got := string(solution); got != "CHANT" {
		t.Fatalf("solution = %q, want CHANT", got)
	}
}

func TestRankActionsSortsDescending(t *testing.T) {
	words := []data.Word{
		mustWord(t, "AAAAA"),
		mustWord(t, "BBBBB"),
		mustWord(t, "CCCCC"),
	}

	ranked := rankActions(words, []float32{-1, 3, 1})
	if got := ranked[0].Word.String(); got != "BBBBB" {
		t.Fatalf("ranked[0] = %q, want BBBBB", got)
	}
	if got := ranked[1].Word.String(); got != "CCCCC" {
		t.Fatalf("ranked[1] = %q, want CCCCC", got)
	}
	if got := ranked[2].Word.String(); got != "AAAAA" {
		t.Fatalf("ranked[2] = %q, want AAAAA", got)
	}
	if ranked[0].Probability <= ranked[1].Probability || ranked[1].Probability <= ranked[2].Probability {
		t.Fatalf("probabilities are not descending: %v", ranked)
	}
}

func TestBatchInputFromTurns(t *testing.T) {
	turn := gameTurn{
		Guess: mustWord(t, "CRATE"),
		Feedback: [data.WordLength]data.Feedback{
			data.FeedbackGreen,
			data.FeedbackGrey,
			data.FeedbackGreen,
			data.FeedbackYellow,
			data.FeedbackGrey,
		},
	}

	input, err := batchInputFromTurns([]gameTurn{turn})
	if err != nil {
		t.Fatalf("batchInputFromTurns() error = %v", err)
	}
	if input.TurnDepth != 1 {
		t.Fatalf("TurnDepth = %d, want 1", input.TurnDepth)
	}
	if input.PreviousGuessWords[0] != turn.Guess {
		t.Fatalf("PreviousGuessWords[0] = %v, want %v", input.PreviousGuessWords[0], turn.Guess)
	}
	if input.PreviousFeedback[0] != turn.Feedback {
		t.Fatalf("PreviousFeedback[0] = %v, want %v", input.PreviousFeedback[0], turn.Feedback)
	}
}

func mustWord(t *testing.T, value string) data.Word {
	t.Helper()

	var word data.Word
	if len(value) != data.WordLength {
		t.Fatalf("word %q has invalid length", value)
	}
	copy(word[:], value)
	return word
}
