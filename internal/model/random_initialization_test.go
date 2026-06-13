package model

import (
	"math"
	"strings"
	"testing"

	"github.com/gomlx/gomlx/pkg/ml/context"

	"github.com/sam-bee/wordle-ml_backprop/internal/data"
)

func TestRandomInitializationConfigValidation(t *testing.T) {
	if err := DefaultRandomInitializationConfig().Validate(); err != nil {
		t.Fatalf("default config validation failed: %v", err)
	}

	tests := []struct {
		name   string
		config RandomInitializationConfig
	}{
		{
			name:   "zero dense gain",
			config: RandomInitializationConfig{DenseWeightGain: 0, OutputEmbeddingTailStddev: DefaultOutputEmbeddingTailStddev},
		},
		{
			name:   "negative dense gain",
			config: RandomInitializationConfig{DenseWeightGain: -1, OutputEmbeddingTailStddev: DefaultOutputEmbeddingTailStddev},
		},
		{
			name:   "nan dense gain",
			config: RandomInitializationConfig{DenseWeightGain: math.NaN(), OutputEmbeddingTailStddev: DefaultOutputEmbeddingTailStddev},
		},
		{
			name:   "negative tail stddev",
			config: RandomInitializationConfig{DenseWeightGain: DefaultDenseWeightGain, OutputEmbeddingTailStddev: -0.1},
		},
		{
			name:   "infinite tail stddev",
			config: RandomInitializationConfig{DenseWeightGain: DefaultDenseWeightGain, OutputEmbeddingTailStddev: math.Inf(1)},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.config.Validate(); err == nil {
				t.Fatal("Validate() succeeded, want error")
			}
		})
	}
}

func TestConfigureRandomInitialization(t *testing.T) {
	ctx := context.New()
	config := RandomInitializationConfig{
		DenseWeightGain:           1.5,
		OutputEmbeddingTailStddev: 0.25,
	}
	if err := ConfigureRandomInitialization(ctx, config); err != nil {
		t.Fatalf("ConfigureRandomInitialization() error = %v", err)
	}

	got := RandomInitializationConfigFromContext(ctx)
	if got != config {
		t.Fatalf("config from context = %+v, want %+v", got, config)
	}

	if err := ConfigureRandomInitialization(nil, DefaultRandomInitializationConfig()); err == nil {
		t.Fatal("ConfigureRandomInitialization(nil) succeeded, want error")
	}
}

func TestValidateActiveActionCount(t *testing.T) {
	for _, actionCount := range []int{1, DefaultActiveActionCount, ActionCatalogCapacity} {
		if err := ValidateActiveActionCount(actionCount); err != nil {
			t.Fatalf("ValidateActiveActionCount(%d) error = %v", actionCount, err)
		}
	}

	for _, actionCount := range []int{0, -1, ActionCatalogCapacity + 1} {
		if err := ValidateActiveActionCount(actionCount); err == nil {
			t.Fatalf("ValidateActiveActionCount(%d) succeeded, want error", actionCount)
		}
	}
}

func TestDenseWeightStddev(t *testing.T) {
	tests := []struct {
		fanIn int
		want  float64
	}{
		{fanIn: 145, want: 0.1174440439},
		{fanIn: 128, want: 0.1250000000},
		{fanIn: 321, want: 0.0789337038},
		{fanIn: 256, want: 0.0883883476},
		{fanIn: 64, want: 0.1767766953},
	}

	for _, tt := range tests {
		got, err := DenseWeightStddev(tt.fanIn, DefaultDenseWeightGain)
		if err != nil {
			t.Fatalf("DenseWeightStddev(%d) error = %v", tt.fanIn, err)
		}
		if math.Abs(got-tt.want) > 1e-10 {
			t.Fatalf("DenseWeightStddev(%d) = %.10f, want %.10f", tt.fanIn, got, tt.want)
		}
	}
}

func TestFixedActionFeatures(t *testing.T) {
	features, err := FixedActionFeatures(fixtureWord(t, "CRASS"))
	if err != nil {
		t.Fatalf("FixedActionFeatures() error = %v", err)
	}

	if features['A'-'A'] != 1 {
		t.Fatalf("A feature = %v, want 1", features['A'-'A'])
	}
	if features['B'-'A'] != -1 {
		t.Fatalf("B feature = %v, want -1", features['B'-'A'])
	}
	if features['C'-'A'] != 1 {
		t.Fatalf("C feature = %v, want 1", features['C'-'A'])
	}
	if features['R'-'A'] != 1 {
		t.Fatalf("R feature = %v, want 1", features['R'-'A'])
	}
	if features['S'-'A'] != 2 {
		t.Fatalf("S feature = %v, want 2", features['S'-'A'])
	}
}

func TestFixedActionFeatureMatrix(t *testing.T) {
	words := []data.Word{
		fixtureWord(t, "CRASS"),
		fixtureWord(t, "ABCDE"),
	}

	features, err := FixedActionFeatureMatrix(words, 2)
	if err != nil {
		t.Fatalf("FixedActionFeatureMatrix() error = %v", err)
	}
	if len(features) != 2*FixedActionFeatureDim {
		t.Fatalf("feature length = %d, want %d", len(features), 2*FixedActionFeatureDim)
	}
	if features[FixedActionFeatureDim+int('E'-'A')] != 1 {
		t.Fatalf("second word E feature = %v, want 1", features[FixedActionFeatureDim+int('E'-'A')])
	}

	if _, err := FixedActionFeatureMatrix(words, 3); err == nil {
		t.Fatal("FixedActionFeatureMatrix() succeeded with too many active actions, want error")
	}
	if _, err := FixedActionFeatureMatrix(words, 0); err == nil {
		t.Fatal("FixedActionFeatureMatrix() succeeded with zero active actions, want error")
	}
}

func fixtureWord(t *testing.T, value string) data.Word {
	t.Helper()

	if len(value) != data.WordLength {
		t.Fatalf("fixture word %q has length %d, want %d", value, len(value), data.WordLength)
	}
	if strings.ToUpper(value) != value {
		t.Fatalf("fixture word %q is not uppercase", value)
	}

	var word data.Word
	copy(word[:], value)
	return word
}
