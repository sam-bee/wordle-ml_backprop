package model

import (
	"fmt"

	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention"

	"github.com/sam-bee/wordle-ml_backprop/internal/data"
)

const (
	PolicyModelInputCount = 4

	LetterAlphabetSize    = 26
	FeedbackAlphabetSize  = 3
	RawTurnFeatureCount   = data.WordLength*LetterAlphabetSize + data.WordLength*FeedbackAlphabetSize
	InputEncoderHidden    = 128
	EncodedTurnFeatureDim = 64

	DenseTrunkInputDim = 1 + data.MaxTurns*EncodedTurnFeatureDim
	DenseTrunkHidden0  = 256
	DenseTrunkHidden1  = 128
	PolicyVectorDim    = 64

	TransformerTokenCount       = 1 + data.MaxTurns
	TransformerModelDim         = 64
	TransformerHeadCount        = 4
	TransformerHeadDim          = 16
	TransformerFeedForwardDim   = 128
	TransformerPolicyHeadHidden = 128

	FixedActionFeatureDim     = 26
	TrainableActionFeatureDim = PolicyVectorDim - FixedActionFeatureDim
)

// PolicyModel implements the Wordle policy architecture specified in
// docs/model-architecture.md.
//
// It expects four tensors:
//   - raw turn features: [batch, 5, 145]
//   - occupied turn mask: [batch, 5], with 1.0 for occupied slots and 0.0 otherwise
//   - virgin-grid flag: [batch, 1], with 1.0 only for an empty grid
//   - fixed action features: [action_count, 26]
//
// It returns one tensor of action logits with shape [batch, action_count].
func PolicyModel(ctx *context.Context, _ any, inputs []*graph.Node) []*graph.Node {
	if len(inputs) != PolicyModelInputCount {
		panic(fmt.Sprintf("policy model expects %d input tensors, got %d", PolicyModelInputCount, len(inputs)))
	}

	turnFeatures := inputs[0]
	occupiedTurns := inputs[1]
	virginGrid := inputs[2]
	fixedActionFeatures := inputs[3]

	validatePolicyInputs(turnFeatures, occupiedTurns, virginGrid, fixedActionFeatures)

	policy := PolicyVector(ctx.In("policy_model"), turnFeatures, occupiedTurns, virginGrid)
	logits := ActionLogits(ctx.In("output_embeddings"), policy, fixedActionFeatures)
	return []*graph.Node{logits}
}

// PolicyVector maps a Wordle decision state to the 64-dimensional policy vector.
func PolicyVector(ctx *context.Context, turnFeatures, occupiedTurns, virginGrid *graph.Node) *graph.Node {
	validateDecisionStateInputs(turnFeatures, occupiedTurns, virginGrid)

	stateToken := TransformerStateToken(ctx.In("state_transformer"), turnFeatures, occupiedTurns)
	modelInput := graph.Concatenate([]*graph.Node{virginGrid, stateToken}, -1)

	head := ctx.In("policy_head")
	hidden := activations.Relu(dense(head.In("input_to_hidden"), modelInput, TransformerPolicyHeadHidden))
	return dense(head.In("hidden_to_output"), hidden, PolicyVectorDim)
}

// TransformerStateToken maps a Wordle decision state to a single contextual state vector.
func TransformerStateToken(ctx *context.Context, turnFeatures, occupiedTurns *graph.Node) *graph.Node {
	validateTurnEncoderInputs(turnFeatures, occupiedTurns)

	batchSize := turnFeatures.Shape().Dim(0)
	g := turnFeatures.Graph()

	flatTurns := graph.Reshape(turnFeatures, batchSize*data.MaxTurns, RawTurnFeatureCount)
	turnTokens := dense(ctx.In("turn_projection"), flatTurns, TransformerModelDim)
	turnTokens = graph.Reshape(turnTokens, batchSize, data.MaxTurns, TransformerModelDim)

	stateToken := ctx.WithInitializer(outputTailInitializer(ctx)).
		VariableWithShape("state_token", shapes.Make(turnFeatures.DType(), 1, 1, TransformerModelDim)).
		ValueGraph(g)
	stateToken = graph.BroadcastToDims(stateToken, batchSize, 1, TransformerModelDim)

	tokens := graph.Concatenate([]*graph.Node{stateToken, turnTokens}, 1)
	positionEmbeddings := ctx.WithInitializer(outputTailInitializer(ctx)).
		VariableWithShape("position_embeddings", shapes.Make(turnFeatures.DType(), 1, TransformerTokenCount, TransformerModelDim)).
		ValueGraph(g)
	positionEmbeddings = graph.BroadcastToDims(positionEmbeddings, batchSize, TransformerTokenCount, TransformerModelDim)
	tokens = graph.Add(tokens, positionEmbeddings)

	keyMask := transformerKeyMask(g, occupiedTurns, batchSize)
	tokens = transformerBlock(ctx.In("block0"), tokens, keyMask)

	state := graph.Slice(tokens, graph.AxisRange(), graph.AxisElem(0), graph.AxisRange())
	return graph.Reshape(state, batchSize, TransformerModelDim)
}

func transformerBlock(ctx *context.Context, tokens, keyMask *graph.Node) *graph.Node {
	attnInput := layers.LayerNormalization(ctx.In("attn_norm"), tokens, -1).Done()
	attnOutput, _ := attention.SelfAttention(ctx.In("self_attention"), attnInput, TransformerHeadCount, TransformerHeadDim).
		WithKeyMask(keyMask).
		WithOutputDim(TransformerModelDim).
		DoneWithCoefficients()
	tokens = graph.Add(tokens, attnOutput)

	ffnInput := layers.LayerNormalization(ctx.In("ffn_norm"), tokens, -1).Done()
	ffnHidden := activations.Relu(dense(ctx.In("ffn").In("input_to_hidden"), ffnInput, TransformerFeedForwardDim))
	ffnOutput := dense(ctx.In("ffn").In("hidden_to_output"), ffnHidden, TransformerModelDim)
	return graph.Add(tokens, ffnOutput)
}

func transformerKeyMask(g *graph.Graph, occupiedTurns *graph.Node, batchSize int) *graph.Node {
	stateMask := graph.BroadcastToDims(graph.Const(g, true), batchSize, 1)
	occupiedMask := graph.GreaterThan(occupiedTurns, graph.ConstAsDType(g, occupiedTurns.DType(), 0.5))
	return graph.Concatenate([]*graph.Node{stateMask, occupiedMask}, -1)
}

// EncodeTurns applies the shared per-turn encoder and zeroes unoccupied slots.
func EncodeTurns(ctx *context.Context, turnFeatures, occupiedTurns *graph.Node) *graph.Node {
	validateTurnEncoderInputs(turnFeatures, occupiedTurns)

	batchSize := turnFeatures.Shape().Dim(0)
	flatTurns := graph.Reshape(turnFeatures, batchSize*data.MaxTurns, RawTurnFeatureCount)

	hidden := activations.Relu(dense(ctx.In("input_to_hidden"), flatTurns, InputEncoderHidden))
	encoded := dense(ctx.In("hidden_to_output"), hidden, EncodedTurnFeatureDim)
	encoded = graph.Reshape(encoded, batchSize, data.MaxTurns, EncodedTurnFeatureDim)

	mask := graph.InsertAxes(occupiedTurns, -1)
	mask = graph.BroadcastToDims(mask, batchSize, data.MaxTurns, EncodedTurnFeatureDim)
	return graph.Mul(encoded, mask)
}

// ActionLogits scores each active action word by dot product against its output embedding.
func ActionLogits(ctx *context.Context, policyVector, fixedActionFeatures *graph.Node) *graph.Node {
	validateActionScoringInputs(policyVector, fixedActionFeatures)

	actionCount := fixedActionFeatures.Shape().Dim(0)
	g := policyVector.Graph()

	if fixedActionFeatures.DType() != policyVector.DType() {
		fixedActionFeatures = graph.ConvertDType(fixedActionFeatures, policyVector.DType())
	}

	policyFixed := graph.Slice(policyVector, graph.AxisRange(), graph.AxisRange(0, FixedActionFeatureDim))
	policyTail := graph.Slice(policyVector, graph.AxisRange(), graph.AxisRange(FixedActionFeatureDim, PolicyVectorDim))

	tail := ctx.WithInitializer(outputTailInitializer(ctx)).
		VariableWithShape("trainable_tail", shapes.Make(policyVector.DType(), actionCount, TrainableActionFeatureDim)).
		ValueGraph(g)

	fixedLogits := graph.MatMul(policyFixed, graph.Transpose(fixedActionFeatures, 0, 1))
	tailLogits := graph.MatMul(policyTail, graph.Transpose(tail, 0, 1))
	return graph.Add(fixedLogits, tailLogits)
}

func dense(ctx *context.Context, input *graph.Node, outputDim int) *graph.Node {
	return layers.Dense(ctx.WithInitializer(denseInitializer(ctx)), input, true, outputDim)
}

func validatePolicyInputs(turnFeatures, occupiedTurns, virginGrid, fixedActionFeatures *graph.Node) {
	validateDecisionStateInputs(turnFeatures, occupiedTurns, virginGrid)
	validateFixedActionFeatures(fixedActionFeatures)
}

func validateDecisionStateInputs(turnFeatures, occupiedTurns, virginGrid *graph.Node) {
	validateTurnEncoderInputs(turnFeatures, occupiedTurns)

	batchSize := turnFeatures.Shape().Dim(0)
	requireRank("virgin-grid flag", virginGrid, 2)
	requireDim("virgin-grid flag", virginGrid, 0, batchSize)
	requireDim("virgin-grid flag", virginGrid, 1, 1)
}

func validateTurnEncoderInputs(turnFeatures, occupiedTurns *graph.Node) {
	requireRank("raw turn features", turnFeatures, 3)
	batchSize := turnFeatures.Shape().Dim(0)

	requireDim("raw turn features", turnFeatures, 1, data.MaxTurns)
	requireDim("raw turn features", turnFeatures, 2, RawTurnFeatureCount)

	requireRank("occupied turn mask", occupiedTurns, 2)
	requireDim("occupied turn mask", occupiedTurns, 0, batchSize)
	requireDim("occupied turn mask", occupiedTurns, 1, data.MaxTurns)
}

func validateActionScoringInputs(policyVector, fixedActionFeatures *graph.Node) {
	requireRank("policy vector", policyVector, 2)
	requireDim("policy vector", policyVector, 1, PolicyVectorDim)

	validateFixedActionFeatures(fixedActionFeatures)
}

func validateFixedActionFeatures(fixedActionFeatures *graph.Node) {
	requireRank("fixed action features", fixedActionFeatures, 2)
	requireDim("fixed action features", fixedActionFeatures, 1, FixedActionFeatureDim)
}

func requireRank(name string, node *graph.Node, want int) {
	if got := node.Shape().Rank(); got != want {
		panic(fmt.Sprintf("%s rank is %d, expected %d", name, got, want))
	}
}

func requireDim(name string, node *graph.Node, axis, want int) {
	if got := node.Shape().Dim(axis); got != want {
		panic(fmt.Sprintf("%s dimension %d is %d, expected %d", name, axis, got, want))
	}
}
