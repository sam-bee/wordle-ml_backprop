package model

import (
	"fmt"
	"math"

	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/initializers"

	"github.com/sam-bee/wordle-ml_backprop/internal/data"
)

const (
	ActionCatalogCapacity            = data.GuessVocabSize
	DefaultActiveActionCount         = 20
	DefaultDenseWeightGain           = 1.0
	DefaultOutputEmbeddingTailStddev = 0.05
	ParamDenseWeightGain             = "wordle_policy_dense_weight_gain"
	ParamOutputEmbeddingTailStddev   = "wordle_policy_output_embedding_tail_stddev"
)

type RandomInitializationConfig struct {
	DenseWeightGain           float64
	OutputEmbeddingTailStddev float64
}

func DefaultRandomInitializationConfig() RandomInitializationConfig {
	return RandomInitializationConfig{
		DenseWeightGain:           DefaultDenseWeightGain,
		OutputEmbeddingTailStddev: DefaultOutputEmbeddingTailStddev,
	}
}

func (config RandomInitializationConfig) Validate() error {
	if !isFinite(config.DenseWeightGain) || config.DenseWeightGain <= 0 {
		return fmt.Errorf("dense weight gain must be finite and > 0, got %v", config.DenseWeightGain)
	}
	if !isFinite(config.OutputEmbeddingTailStddev) || config.OutputEmbeddingTailStddev < 0 {
		return fmt.Errorf("output embedding tail stddev must be finite and >= 0, got %v", config.OutputEmbeddingTailStddev)
	}
	return nil
}

func ConfigureRandomInitialization(ctx *context.Context, config RandomInitializationConfig) error {
	if ctx == nil {
		return fmt.Errorf("context is nil")
	}
	if err := config.Validate(); err != nil {
		return err
	}
	ctx.SetParam(ParamDenseWeightGain, config.DenseWeightGain)
	ctx.SetParam(ParamOutputEmbeddingTailStddev, config.OutputEmbeddingTailStddev)
	return nil
}

func RandomInitializationConfigFromContext(ctx *context.Context) RandomInitializationConfig {
	if ctx == nil {
		return DefaultRandomInitializationConfig()
	}
	return RandomInitializationConfig{
		DenseWeightGain:           context.GetParamOr(ctx, ParamDenseWeightGain, DefaultDenseWeightGain),
		OutputEmbeddingTailStddev: context.GetParamOr(ctx, ParamOutputEmbeddingTailStddev, DefaultOutputEmbeddingTailStddev),
	}
}

func ValidateActiveActionCount(actionCount int) error {
	if actionCount < 1 || actionCount > ActionCatalogCapacity {
		return fmt.Errorf("active action count must be in 1..%d, got %d", ActionCatalogCapacity, actionCount)
	}
	return nil
}

func DenseWeightStddev(fanIn int, denseWeightGain float64) (float64, error) {
	if fanIn <= 0 {
		return 0, fmt.Errorf("fan-in must be > 0, got %d", fanIn)
	}
	if !isFinite(denseWeightGain) || denseWeightGain <= 0 {
		return 0, fmt.Errorf("dense weight gain must be finite and > 0, got %v", denseWeightGain)
	}
	return denseWeightGain * math.Sqrt(2/float64(fanIn)), nil
}

func FixedActionFeatures(word data.Word) ([FixedActionFeatureDim]float32, error) {
	var features [FixedActionFeatureDim]float32
	if word.IsEmpty() {
		return features, fmt.Errorf("action word is empty")
	}

	for index := range features {
		features[index] = -1
	}

	var counts [LetterAlphabetSize]int
	for pos, b := range word {
		if b < 'A' || b > 'Z' {
			return features, fmt.Errorf("action word byte %d contains non-uppercase ASCII byte %d", pos, b)
		}
		counts[int(b-'A')]++
	}
	for letter, count := range counts {
		if count > 0 {
			features[letter] = float32(count)
		}
	}
	return features, nil
}

func FixedActionFeatureMatrix(words []data.Word, activeActionCount int) ([]float32, error) {
	if err := ValidateActiveActionCount(activeActionCount); err != nil {
		return nil, err
	}
	if activeActionCount > len(words) {
		return nil, fmt.Errorf("active action count %d exceeds action word count %d", activeActionCount, len(words))
	}

	features := make([]float32, activeActionCount*FixedActionFeatureDim)
	for wordIndex := 0; wordIndex < activeActionCount; wordIndex++ {
		wordFeatures, err := FixedActionFeatures(words[wordIndex])
		if err != nil {
			return nil, fmt.Errorf("action word %d: %w", wordIndex, err)
		}
		copy(features[wordIndex*FixedActionFeatureDim:], wordFeatures[:])
	}
	return features, nil
}

func denseInitializer(ctx *context.Context) initializers.VariableInitializer {
	config := RandomInitializationConfigFromContext(ctx)
	if err := config.Validate(); err != nil {
		panic(fmt.Sprintf("invalid random initialization config: %v", err))
	}
	return heWithGainFn(ctx, config.DenseWeightGain)
}

func outputTailInitializer(ctx *context.Context) initializers.VariableInitializer {
	config := RandomInitializationConfigFromContext(ctx)
	if err := config.Validate(); err != nil {
		panic(fmt.Sprintf("invalid random initialization config: %v", err))
	}
	return medianNormalizedOutputTailFn(ctx, config.OutputEmbeddingTailStddev)
}

func heWithGainFn(ctx *context.Context, denseWeightGain float64) initializers.VariableInitializer {
	return func(g *graph.Graph, shape shapes.Shape) *graph.Node {
		if !shape.DType.IsFloat() && !shape.DType.IsComplex() {
			return graph.Zeros(g, shape)
		}
		if shape.Rank() <= 1 {
			return graph.Zeros(g, shape)
		}

		stddev, err := DenseWeightStddev(shape.Dimensions[0], denseWeightGain)
		if err != nil {
			panic(err)
		}
		values := ctx.RandomNormal(g, shape)
		return graph.MulScalar(values, stddev)
	}
}

func randomNormalFn(ctx *context.Context, stddev float64) initializers.VariableInitializer {
	return func(g *graph.Graph, shape shapes.Shape) *graph.Node {
		if !shape.DType.IsFloat() && !shape.DType.IsComplex() {
			return graph.Zeros(g, shape)
		}
		if stddev == 0 {
			return graph.Zeros(g, shape)
		}
		values := ctx.RandomNormal(g, shape)
		return graph.MulScalar(values, stddev)
	}
}

func medianNormalizedOutputTailFn(ctx *context.Context, stddev float64) initializers.VariableInitializer {
	return func(g *graph.Graph, shape shapes.Shape) *graph.Node {
		tail := randomNormalFn(ctx, stddev)(g, shape)
		if !shape.DType.IsFloat() {
			return tail
		}
		if shape.Rank() != 2 {
			panic(fmt.Sprintf("output embedding tail must have rank 2, got shape %s", shape))
		}
		return normalizeRowsToMedianNorm(tail)
	}
}

func normalizeRowsToMedianNorm(rows *graph.Node) *graph.Node {
	if rows.Shape().Rank() != 2 {
		panic(fmt.Sprintf("rows must have rank 2, got shape %s", rows.Shape()))
	}

	rowCount := rows.Shape().Dim(0)
	rowNorms := graph.Reshape(graph.L2Norm(rows, -1), rowCount)
	sortedNorms := graph.Sort(rowNorms, 0, true)

	lower := graph.Slice(sortedNorms, graph.AxisElem((rowCount-1)/2))
	upper := graph.Slice(sortedNorms, graph.AxisElem(rowCount/2))
	medianNorm := graph.Reshape(graph.MulScalar(graph.Add(lower, upper), 0.5))

	return graph.Mul(graph.L2Normalize(rows, -1), medianNorm)
}

func isFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}
