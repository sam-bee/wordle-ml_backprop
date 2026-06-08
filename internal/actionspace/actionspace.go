package actionspace

import (
	"fmt"

	enginewords "github.com/sam-bee/wordle-ml_game-engine/words"

	"github.com/sam-bee/wordle-ml_backprop/internal/data"
)

type Vocabulary struct {
	Words []data.Word
	Index map[data.Word]int
}

func Load() (Vocabulary, error) {
	words, err := enginewords.GetActionSpace()
	if err != nil {
		return Vocabulary{}, fmt.Errorf("get game-engine action space: %w", err)
	}
	return NewVocabulary(words)
}

func NewVocabulary(engineWords []enginewords.Word) (Vocabulary, error) {
	if len(engineWords) != data.GuessVocabSize {
		return Vocabulary{}, fmt.Errorf("action space has %d words, expected %d", len(engineWords), data.GuessVocabSize)
	}

	vocab := Vocabulary{
		Words: make([]data.Word, len(engineWords)),
		Index: make(map[data.Word]int, len(engineWords)),
	}
	for index, engineWord := range engineWords {
		word, err := convertWord(engineWord)
		if err != nil {
			return Vocabulary{}, fmt.Errorf("action word %d: %w", index, err)
		}
		if previousIndex, exists := vocab.Index[word]; exists {
			return Vocabulary{}, fmt.Errorf("action word %q appears at indexes %d and %d", word, previousIndex, index)
		}
		vocab.Words[index] = word
		vocab.Index[word] = index
	}
	return vocab, nil
}

func convertWord(engineWord enginewords.Word) (data.Word, error) {
	var word data.Word
	value := string(engineWord)
	if len(value) != data.WordLength {
		return word, fmt.Errorf("word %q has length %d, expected %d", value, len(value), data.WordLength)
	}
	for pos := range value {
		b := value[pos]
		if b < 'A' || b > 'Z' {
			return word, fmt.Errorf("word %q contains non-uppercase ASCII byte %d at position %d", value, b, pos)
		}
		word[pos] = b
	}
	return word, nil
}
