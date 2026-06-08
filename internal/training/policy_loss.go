package training

import (
	"fmt"
	"math"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"

	"github.com/sam-bee/wordle-ml_backprop/internal/data"
)

const (
	PolicyMainLossWeight      = 0.8
	PolicyAuxiliaryLossWeight = 0.2
)

// PolicyLoss implements the supervised policy loss described in
// docs/todo/loss-shaping.md.
//
// labels[0] must be the teacher top-k action indices with shape [batch, 16].
// logits[0] must be full-vocabulary action logits with shape [batch, action_count].
func PolicyLoss(labels, logits []*graph.Node) *graph.Node {
	if len(labels) != 1 {
		panic(fmt.Sprintf("policy loss expects 1 label tensor, got %d", len(labels)))
	}
	if len(logits) != 1 {
		panic(fmt.Sprintf("policy loss expects 1 logits tensor, got %d", len(logits)))
	}

	teacherTopK := labels[0]
	actionLogits := logits[0]
	validatePolicyLossInputs(teacherTopK, actionLogits)

	targetDistribution := PolicyTargetDistribution(teacherTopK, actionLogits.Shape().Dim(1), actionLogits.DType())
	logConfidence := graph.LogSoftmax(actionLogits)
	perExampleLoss := graph.ReduceSum(graph.Neg(graph.Mul(targetDistribution, logConfidence)), -1)
	return graph.ReduceAllMean(perExampleLoss)
}

// PolicyTargetDistribution builds the dense target distribution used by
// PolicyLoss. It places alpha mass on the teacher's best guess and beta*q_j
// mass on every teacher top-k guess.
func PolicyTargetDistribution(teacherTopK *graph.Node, actionCount int, dtype dtypes.DType) *graph.Node {
	validateTeacherTopK(teacherTopK)
	if actionCount <= 0 {
		panic(fmt.Sprintf("action count is %d, expected positive", actionCount))
	}
	if !dtype.IsFloat() {
		panic(fmt.Sprintf("target distribution dtype is %s, expected floating point", dtype))
	}

	g := teacherTopK.Graph()
	oneHotTopK := graph.OneHot(teacherTopK, actionCount, dtype)

	rankWeights := graph.ConstAsDType(g, dtype, PolicyAuxiliaryRankWeights())
	rankWeights = graph.Reshape(rankWeights, 1, data.TopK, 1)
	auxiliaryTarget := graph.ReduceSum(graph.Mul(oneHotTopK, rankWeights), 1)

	top1 := graph.Slice(teacherTopK, graph.AxisRange(), graph.AxisRange(0, 1))
	mainTarget := graph.ReduceSum(graph.OneHot(top1, actionCount, dtype), 1)

	mainTarget = graph.MulScalar(mainTarget, PolicyMainLossWeight)
	auxiliaryTarget = graph.MulScalar(auxiliaryTarget, PolicyAuxiliaryLossWeight)
	return graph.Add(mainTarget, auxiliaryTarget)
}

// PolicyAuxiliaryRankWeights returns q_j = 2^-j / Z for j in 1..16.
func PolicyAuxiliaryRankWeights() []float32 {
	weights := make([]float32, data.TopK)
	var total float64
	for rank := range weights {
		weight := math.Pow(0.5, float64(rank+1))
		weights[rank] = float32(weight)
		total += weight
	}
	for rank := range weights {
		weights[rank] /= float32(total)
	}
	return weights
}

func validatePolicyLossInputs(teacherTopK, actionLogits *graph.Node) {
	validateTeacherTopK(teacherTopK)

	requireLossRank("action logits", actionLogits, 2)
	requireLossDim("action logits", actionLogits, 0, teacherTopK.Shape().Dim(0))
	if actionLogits.Shape().Dim(1) <= 0 {
		panic(fmt.Sprintf("action logits dimension 1 is %d, expected positive", actionLogits.Shape().Dim(1)))
	}
	if !actionLogits.DType().IsFloat() {
		panic(fmt.Sprintf("action logits dtype is %s, expected floating point", actionLogits.DType()))
	}
}

func validateTeacherTopK(teacherTopK *graph.Node) {
	requireLossRank("teacher top-k indices", teacherTopK, 2)
	requireLossDim("teacher top-k indices", teacherTopK, 1, data.TopK)
	if !teacherTopK.DType().IsInt() {
		panic(fmt.Sprintf("teacher top-k indices dtype is %s, expected integer", teacherTopK.DType()))
	}
}

func requireLossRank(name string, node *graph.Node, want int) {
	if got := node.Shape().Rank(); got != want {
		panic(fmt.Sprintf("%s rank is %d, expected %d", name, got, want))
	}
}

func requireLossDim(name string, node *graph.Node, axis, want int) {
	if got := node.Shape().Dim(axis); got != want {
		panic(fmt.Sprintf("%s dimension %d is %d, expected %d", name, axis, got, want))
	}
}
