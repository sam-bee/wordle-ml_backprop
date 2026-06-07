package model

import (
	"fmt"

	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"

	"github.com/sam-bee/wordle-ml_backprop/internal/data"
)

const (
	InputFeatureCount = 1 + 1 + data.MaxTurns*data.WordLength + data.MaxTurns*data.WordLength
	OutputCount       = data.TopK
)

func SanityModel(ctx *context.Context, _ any, inputs []*graph.Node) []*graph.Node {
	if len(inputs) != 1 {
		panic(fmt.Sprintf("sanity model expects 1 input tensor, got %d", len(inputs)))
	}

	input := inputs[0]
	shape := input.Shape()
	if shape.Rank() < 2 {
		panic(fmt.Sprintf("sanity model input rank is %d, expected at least 2", shape.Rank()))
	}

	batchSize := shape.Dim(0)
	flat := graph.Reshape(input, batchSize, -1)
	if featureCount := flat.Shape().Dim(1); featureCount != InputFeatureCount {
		panic(fmt.Sprintf("sanity model input feature count is %d, expected %d", featureCount, InputFeatureCount))
	}

	output := layers.Dense(ctx.In("sanity_linear"), flat, true, OutputCount)
	return []*graph.Node{output}
}
