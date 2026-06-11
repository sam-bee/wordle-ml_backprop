package inference

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/xla"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/checkpoints"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"

	"github.com/sam-bee/wordle-ml_backprop/internal/actionspace"
	"github.com/sam-bee/wordle-ml_backprop/internal/data"
	"github.com/sam-bee/wordle-ml_backprop/internal/model"
	"github.com/sam-bee/wordle-ml_backprop/internal/training"
)

const backendConfig = "cuda"

type Player struct {
	backend            backends.Backend
	exec               *context.Exec
	fixedActionTensor  *tensors.Tensor
	vocab              actionspace.Vocabulary
	BackendDescription string
	DeviceDescription  string
	TrunkHiddenDims    []int
}

type ScoredAction struct {
	Word        data.Word
	ActionIndex int
	Logit       float32
	Probability float64
}

type checkpointMetadata struct {
	Variables []checkpointVariable `json:"Variables"`
}

type checkpointVariable struct {
	ParameterName string `json:"ParameterName"`
	Dimensions    []int  `json:"Dimensions"`
	DType         string `json:"DType"`
}

func NewPlayer(weightsPath, metadataPath string, vocab actionspace.Vocabulary) (*Player, error) {
	metadata, err := os.ReadFile(metadataPath)
	if err != nil {
		return nil, fmt.Errorf("read metadata %s: %w", metadataPath, err)
	}
	trunkHiddenDims, err := TrunkHiddenDimsFromMetadata(metadataPath, metadata)
	if err != nil {
		return nil, err
	}

	weights, err := os.ReadFile(weightsPath)
	if err != nil {
		return nil, fmt.Errorf("read weights %s: %w", weightsPath, err)
	}

	backend, err := newBackend()
	if err != nil {
		return nil, err
	}

	ctx := context.New()
	if _, err := checkpoints.Build(ctx).FromEmbed(string(metadata), weights).Done(); err != nil {
		backend.Finalize()
		return nil, fmt.Errorf("load GoMLX checkpoint files: %w", err)
	}

	exec, err := context.NewExec(
		backend,
		ctx.Reuse(),
		func(ctx *context.Context, turnFeatures, occupiedTurns, virginGrid, fixedActionFeatures *graph.Node) *graph.Node {
			policy := policyVector(ctx.In("policy_model"), turnFeatures, occupiedTurns, virginGrid, trunkHiddenDims)
			return model.ActionLogits(ctx.In("output_embeddings"), policy, fixedActionFeatures)
		},
	)
	if err != nil {
		backend.Finalize()
		return nil, fmt.Errorf("build policy inference executor: %w", err)
	}

	fixedActionFeatures, err := model.FixedActionFeatureMatrix(vocab.Words, len(vocab.Words))
	if err != nil {
		exec.Finalize()
		backend.Finalize()
		return nil, fmt.Errorf("build fixed action features: %w", err)
	}
	fixedActionTensor := tensors.FromFlatDataAndDimensions(
		fixedActionFeatures,
		len(vocab.Words),
		model.FixedActionFeatureDim,
	)

	return &Player{
		backend:            backend,
		exec:               exec,
		fixedActionTensor:  fixedActionTensor,
		vocab:              vocab,
		BackendDescription: backend.Description(),
		DeviceDescription:  backend.DeviceDescription(0),
		TrunkHiddenDims:    append([]int(nil), trunkHiddenDims...),
	}, nil
}

func (player *Player) Close() {
	if player == nil {
		return
	}
	if player.fixedActionTensor != nil {
		player.fixedActionTensor.MustFinalizeAll()
		player.fixedActionTensor = nil
	}
	if player.exec != nil {
		player.exec.Finalize()
		player.exec = nil
	}
	if player.backend != nil {
		player.backend.Finalize()
		player.backend = nil
	}
}

func (player *Player) Predict(input data.BatchInput) ([]ScoredAction, error) {
	batch := data.Batch{Inputs: []data.BatchInput{input}}
	stateInputs, err := training.BatchToPolicyStateTensors(batch)
	if err != nil {
		return nil, err
	}
	defer finalizeTensors(stateInputs)

	logitsTensor, err := player.exec.Exec1(stateInputs[0], stateInputs[1], stateInputs[2], player.fixedActionTensor)
	if err != nil {
		return nil, err
	}
	defer logitsTensor.MustFinalizeAll()

	logits := tensors.MustCopyFlatData[float32](logitsTensor)
	if len(logits) != len(player.vocab.Words) {
		return nil, fmt.Errorf("model returned %d logits, expected %d", len(logits), len(player.vocab.Words))
	}
	return RankActions(player.vocab.Words, logits), nil
}

func RankActions(words []data.Word, logits []float32) []ScoredAction {
	maxLogit := float32(math.Inf(-1))
	for _, logit := range logits {
		if logit > maxLogit {
			maxLogit = logit
		}
	}

	expSum := 0.0
	probabilities := make([]float64, len(logits))
	for index, logit := range logits {
		value := math.Exp(float64(logit - maxLogit))
		probabilities[index] = value
		expSum += value
	}

	ranked := make([]ScoredAction, len(logits))
	for index, logit := range logits {
		ranked[index] = ScoredAction{
			Word:        words[index],
			ActionIndex: index,
			Logit:       logit,
			Probability: probabilities[index] / expSum,
		}
	}
	sort.Slice(ranked, func(i, j int) bool {
		if ranked[i].Logit == ranked[j].Logit {
			return ranked[i].ActionIndex < ranked[j].ActionIndex
		}
		return ranked[i].Logit > ranked[j].Logit
	})
	return ranked
}

func FirstUnguessed(ranked []ScoredAction, guessed map[data.Word]bool) (data.Word, int, error) {
	for index, action := range ranked {
		if !guessed[action.Word] {
			return action.Word, index, nil
		}
	}
	return data.Word{}, 0, fmt.Errorf("model action space has no unguessed words")
}

func TrunkHiddenDimsFromMetadata(path string, content []byte) ([]int, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(content, &raw); err != nil {
		return nil, fmt.Errorf("parse checkpoint metadata %s: %w", path, err)
	}
	if _, ok := raw["latest_gomlx_checkpoint"]; ok {
		return nil, fmt.Errorf("%s looks like a project manifest; pass the GoMLX checkpoint .json file instead", path)
	}
	if _, ok := raw["Variables"]; !ok {
		return nil, fmt.Errorf("%s does not look like a GoMLX checkpoint metadata file: missing Variables", path)
	}

	var metadata checkpointMetadata
	if err := json.Unmarshal(content, &metadata); err != nil {
		return nil, fmt.Errorf("parse checkpoint metadata %s: %w", path, err)
	}
	layersByName := denseTrunkWeightLayers(metadata.Variables)
	return trunkHiddenDims(layersByName)
}

func denseTrunkWeightLayers(variables []checkpointVariable) map[string][]int {
	const prefix = "var:/policy_model/dense_trunk/"
	const suffix = "/dense/weights"

	layersByName := make(map[string][]int)
	for _, variable := range variables {
		name := variable.ParameterName
		if !strings.HasPrefix(name, prefix) || !strings.HasSuffix(name, suffix) {
			continue
		}
		layerName := strings.TrimSuffix(strings.TrimPrefix(name, prefix), suffix)
		layersByName[layerName] = variable.Dimensions
	}
	return layersByName
}

func trunkHiddenDims(layersByName map[string][]int) ([]int, error) {
	dims, ok := layersByName["input_to_hidden0"]
	if !ok {
		return nil, fmt.Errorf("checkpoint metadata is missing dense_trunk.input_to_hidden0 weights")
	}
	if err := requireWeightShape("input_to_hidden0", dims, model.DenseTrunkInputDim, 0); err != nil {
		return nil, err
	}

	hiddenDims := []int{dims[1]}
	previousWidth := dims[1]
	for hiddenIndex := 0; hiddenIndex < 32; hiddenIndex++ {
		outputLayer := fmt.Sprintf("hidden%d_to_output", hiddenIndex)
		if dims, ok := layersByName[outputLayer]; ok {
			return hiddenDims, requireWeightShape(outputLayer, dims, previousWidth, model.PolicyVectorDim)
		}

		nextLayer := fmt.Sprintf("hidden%d_to_hidden%d", hiddenIndex, hiddenIndex+1)
		dims, ok := layersByName[nextLayer]
		if !ok {
			return nil, fmt.Errorf("checkpoint metadata is missing dense_trunk.%s or dense_trunk.%s weights", outputLayer, nextLayer)
		}
		if err := requireWeightShape(nextLayer, dims, previousWidth, 0); err != nil {
			return nil, err
		}
		hiddenDims = append(hiddenDims, dims[1])
		previousWidth = dims[1]
	}
	return nil, fmt.Errorf("checkpoint metadata has too many dense trunk hidden layers")
}

func requireWeightShape(layerName string, dims []int, wantInput, wantOutput int) error {
	if len(dims) != 2 {
		return fmt.Errorf("dense_trunk.%s weights have dimensions %v, expected rank 2", layerName, dims)
	}
	if dims[0] != wantInput {
		return fmt.Errorf("dense_trunk.%s input width is %d, expected %d", layerName, dims[0], wantInput)
	}
	if wantOutput > 0 && dims[1] != wantOutput {
		return fmt.Errorf("dense_trunk.%s output width is %d, expected %d", layerName, dims[1], wantOutput)
	}
	return nil
}

func policyVector(ctx *context.Context, turnFeatures, occupiedTurns, virginGrid *graph.Node, trunkHiddenDims []int) *graph.Node {
	encodedTurns := model.EncodeTurns(ctx.In("input_encoder"), turnFeatures, occupiedTurns)
	batchSize := encodedTurns.Shape().Dim(0)

	flatTurns := graph.Reshape(encodedTurns, batchSize, data.MaxTurns*model.EncodedTurnFeatureDim)
	hidden := graph.Concatenate([]*graph.Node{virginGrid, flatTurns}, -1)

	trunk := ctx.In("dense_trunk")
	for index, outputDim := range trunkHiddenDims {
		scope := "input_to_hidden0"
		if index > 0 {
			scope = fmt.Sprintf("hidden%d_to_hidden%d", index-1, index)
		}
		hidden = activations.Relu(layers.Dense(trunk.In(scope), hidden, true, outputDim))
	}
	return layers.Dense(trunk.In(fmt.Sprintf("hidden%d_to_output", len(trunkHiddenDims)-1)), hidden, true, model.PolicyVectorDim)
}

func newBackend() (backends.Backend, error) {
	xla.EnableAutoInstall(false)

	backend, err := xla.New(backendConfig)
	if err != nil {
		return nil, fmt.Errorf("create XLA CUDA backend: %w", err)
	}
	if got := backend.NumDevices(); got != 1 {
		backend.Finalize()
		return nil, fmt.Errorf("XLA CUDA backend exposes %d devices, expected exactly 1", got)
	}
	return backend, nil
}

func finalizeTensors(values []*tensors.Tensor) {
	for _, value := range values {
		if value != nil {
			value.MustFinalizeAll()
		}
	}
}
