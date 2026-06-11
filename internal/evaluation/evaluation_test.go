package evaluation

import "testing"

func TestScoreEpisode(t *testing.T) {
	tests := []struct {
		won       bool
		turnsUsed int
		want      int
	}{
		{won: false, turnsUsed: 6, want: 0},
		{won: true, turnsUsed: 6, want: 10},
		{won: true, turnsUsed: 5, want: 11},
		{won: true, turnsUsed: 1, want: 15},
	}

	for _, tt := range tests {
		if got := ScoreEpisode(tt.won, tt.turnsUsed); got != tt.want {
			t.Fatalf("ScoreEpisode(%t, %d) = %d, want %d", tt.won, tt.turnsUsed, got, tt.want)
		}
	}
}

func TestSummarize(t *testing.T) {
	summary := Summarize([]EpisodeResult{
		{Won: true, TurnsUsed: 3, Score: 13},
		{Won: true, TurnsUsed: 6, Score: 10},
		{Won: false, TurnsUsed: 6, Score: 0},
	}, "validation")

	if summary.RawScore != 23 {
		t.Fatalf("RawScore = %d, want 23", summary.RawScore)
	}
	if summary.MaxScore != 45 {
		t.Fatalf("MaxScore = %d, want 45", summary.MaxScore)
	}
	if summary.Wins != 2 || summary.Losses != 1 {
		t.Fatalf("wins/losses = %d/%d, want 2/1", summary.Wins, summary.Losses)
	}
	if summary.Turns3 != 1 || summary.Turns6 != 1 {
		t.Fatalf("turn histogram 3/6 = %d/%d, want 1/1", summary.Turns3, summary.Turns6)
	}
}
