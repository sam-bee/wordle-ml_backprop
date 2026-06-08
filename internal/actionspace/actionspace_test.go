package actionspace

import (
	"testing"

	enginewords "github.com/sam-bee/wordle-ml_game-engine/words"

	"github.com/sam-bee/wordle-ml_backprop/internal/data"
)

func TestLoad(t *testing.T) {
	vocab, err := Load()
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if len(vocab.Words) != data.GuessVocabSize {
		t.Fatalf("word count = %d, want %d", len(vocab.Words), data.GuessVocabSize)
	}
	if vocab.Words[0].String() != "AARGH" {
		t.Fatalf("first word = %q, want AARGH", vocab.Words[0])
	}
	if got, exists := vocab.Index[vocab.Words[0]]; !exists || got != 0 {
		t.Fatalf("first word index = %d, %t; want 0, true", got, exists)
	}
}

func TestNewVocabularyRejectsWrongCount(t *testing.T) {
	if _, err := NewVocabulary([]enginewords.Word{"AARGH"}); err == nil {
		t.Fatal("NewVocabulary() succeeded with wrong count, want error")
	}
}
