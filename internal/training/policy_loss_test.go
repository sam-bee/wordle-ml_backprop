package training

import (
	"math"
	"testing"

	"github.com/gomlx/gomlx/backends/simplego"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"

	"github.com/sam-bee/wordle-ml_backprop/internal/data"
)

func TestPolicyAuxiliaryRankWeights(t *testing.T) {
	weights := PolicyAuxiliaryRankWeights()
	if len(weights) != data.TopK {
		t.Fatalf("len(weights) = %d, want %d", len(weights), data.TopK)
	}

	var sum float32
	for i, weight := range weights {
		if weight <= 0 {
			t.Fatalf("weights[%d] = %g, want positive", i, weight)
		}
		if i > 0 && weight >= weights[i-1] {
			t.Fatalf("weights[%d] = %g, want less than previous %g", i, weight, weights[i-1])
		}
		sum += weight
	}
	if math.Abs(float64(sum-1)) > 1e-6 {
		t.Fatalf("sum(weights) = %g, want 1", sum)
	}
}

func TestPolicyTargetDistribution(t *testing.T) {
	teacher := []int32{3, 1, 0, 2, 3, 1, 0, 2, 3, 1, 0, 2, 3, 1, 0, 2}
	want := expectedTargetDistribution(teacher, 4)

	output := execPolicyLossGraph(t, func(g *graph.Graph) *graph.Node {
		teacherTopK := graph.Const(g, [][]int32{teacher})
		return PolicyTargetDistribution(teacherTopK, 4, dtypes.Float32)
	})
	defer output.MustFinalizeAll()

	got := tensors.MustCopyFlatData[float32](output)
	assertFloat32SlicesClose(t, got, want, 1e-6)
}

func TestPolicyLoss(t *testing.T) {
	teacher := []int32{3, 1, 0, 2, 3, 1, 0, 2, 3, 1, 0, 2, 3, 1, 0, 2}
	logits := []float64{0, 1, 2, 3}
	want := expectedPolicyLoss(teacher, logits)

	output := execPolicyLossGraph(t, func(g *graph.Graph) *graph.Node {
		teacherTopK := graph.Const(g, [][]int32{teacher})
		actionLogits := graph.Const(g, [][]float32{{0, 1, 2, 3}})
		return PolicyLoss([]*graph.Node{teacherTopK}, []*graph.Node{actionLogits})
	})
	defer output.MustFinalizeAll()

	got := tensors.ToScalar[float32](output)
	if math.Abs(float64(got)-want) > 1e-6 {
		t.Fatalf("policy loss = %g, want %g", got, want)
	}
}

func execPolicyLossGraph(t *testing.T, fn func(g *graph.Graph) *graph.Node) *tensors.Tensor {
	t.Helper()

	backend, err := simplego.New("")
	if err != nil {
		t.Fatalf("create SimpleGo backend: %v", err)
	}
	t.Cleanup(backend.Finalize)

	outputs, err := graph.ExecOnceN(backend, func(g *graph.Graph) []*graph.Node {
		return []*graph.Node{fn(g)}
	})
	if err != nil {
		t.Fatalf("execute graph: %v", err)
	}
	if len(outputs) != 1 {
		t.Fatalf("len(outputs) = %d, want 1", len(outputs))
	}
	return outputs[0]
}

func assertFloat32SlicesClose(t *testing.T, got, want []float32, tolerance float64) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("len(got) = %d, len(want) = %d", len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > tolerance {
			t.Fatalf("got[%d] = %g, want %g", i, got[i], want[i])
		}
	}
}

func expectedTargetDistribution(teacher []int32, actionCount int) []float32 {
	target := make([]float32, actionCount)
	weights := PolicyAuxiliaryRankWeights()
	target[teacher[0]] += PolicyMainLossWeight
	for rank, actionIndex := range teacher {
		target[actionIndex] += PolicyAuxiliaryLossWeight * weights[rank]
	}
	return target
}

func expectedPolicyLoss(teacher []int32, logits []float64) float64 {
	logConfidence := logSoftmax(logits)
	loss := PolicyMainLossWeight * -logConfidence[teacher[0]]

	weights := PolicyAuxiliaryRankWeights()
	for rank, actionIndex := range teacher {
		loss += PolicyAuxiliaryLossWeight * float64(weights[rank]) * -logConfidence[actionIndex]
	}
	return loss
}

func logSoftmax(logits []float64) []float64 {
	maxLogit := logits[0]
	for _, logit := range logits[1:] {
		maxLogit = max(maxLogit, logit)
	}

	var expSum float64
	for _, logit := range logits {
		expSum += math.Exp(logit - maxLogit)
	}

	logDenominator := maxLogit + math.Log(expSum)
	values := make([]float64, len(logits))
	for i, logit := range logits {
		values[i] = logit - logDenominator
	}
	return values
}
