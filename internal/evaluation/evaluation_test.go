package evaluation

import (
	"testing"

	"github.com/sam-bee/wordle-ml_backprop/internal/data"
)

func TestScoreEpisode(t *testing.T) {
	tests := []struct {
		won       bool
		turnsUsed int
		want      float64
	}{
		{won: false, turnsUsed: 6, want: 0},
		{won: true, turnsUsed: 6, want: 10},
		{won: true, turnsUsed: 5, want: 11},
		{won: true, turnsUsed: 1, want: 15},
	}

	for _, tt := range tests {
		if got := ScoreEpisode(tt.won, tt.turnsUsed); got != tt.want {
			t.Fatalf("ScoreEpisode(%t, %d) = %g, want %g", tt.won, tt.turnsUsed, got, tt.want)
		}
	}
}

func TestLossFeedbackScore(t *testing.T) {
	turns := []Turn{
		{
			Guess: fixtureWord("ABCDE"),
			Feedback: [5]data.Feedback{
				data.FeedbackGreen,
				data.FeedbackYellow,
				data.FeedbackGrey,
				data.FeedbackGrey,
				data.FeedbackGrey,
			},
		},
		{
			Guess: fixtureWord("XBYYY"),
			Feedback: [5]data.Feedback{
				data.FeedbackGrey,
				data.FeedbackGreen,
				data.FeedbackYellow,
				data.FeedbackGrey,
				data.FeedbackGrey,
			},
		},
	}

	got := LossFeedbackScore(turns)
	const want = 1.5
	if got != want {
		t.Fatalf("LossFeedbackScore() = %g, want %g", got, want)
	}
}

func TestSummarize(t *testing.T) {
	summary := Summarize([]EpisodeResult{
		{Won: true, TurnsUsed: 3, Score: 13},
		{Won: true, TurnsUsed: 6, Score: 10},
		{Won: false, TurnsUsed: 6, Score: 1.5},
	}, "validation")

	if summary.RawScore != 24.5 {
		t.Fatalf("RawScore = %g, want 24.5", summary.RawScore)
	}
	if summary.MaxScore != 45 {
		t.Fatalf("MaxScore = %g, want 45", summary.MaxScore)
	}
	if summary.WinScore != 23 || summary.LossCreditScore != 1.5 {
		t.Fatalf("win/loss credit score = %g/%g, want 23/1.5", summary.WinScore, summary.LossCreditScore)
	}
	if summary.Wins != 2 || summary.Losses != 1 {
		t.Fatalf("wins/losses = %d/%d, want 2/1", summary.Wins, summary.Losses)
	}
	if summary.Turns3 != 1 || summary.Turns6 != 1 {
		t.Fatalf("turn histogram 3/6 = %d/%d, want 1/1", summary.Turns3, summary.Turns6)
	}
}

func fixtureWord(value string) data.Word {
	var word data.Word
	copy(word[:], value)
	return word
}
