package inference

import (
	"fmt"
	"math"
	"os"
	"sort"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/xla"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/checkpoints"

	"github.com/sam-bee/wordle-ml_backprop/internal/actionspace"
	"github.com/sam-bee/wordle-ml_backprop/internal/data"
	"github.com/sam-bee/wordle-ml_backprop/internal/model"
	"github.com/sam-bee/wordle-ml_backprop/internal/training"
)

const backendConfig = "cuda"

type Player struct {
	backend             backends.Backend
	exec                *context.Exec
	fixedActionTensor   *tensors.Tensor
	vocab               actionspace.Vocabulary
	BackendDescription  string
	DeviceDescription   string
	TrunkHiddenDims     []int
	TrunkOutputDim      int
	PolicyVectorDim     int
	TrainableTailDim    int
	HasPolicyOutputHead bool
}

type ScoredAction struct {
	Word        data.Word
	ActionIndex int
	Logit       float32
	Probability float64
}

func NewPlayer(weightsPath, metadataPath string, vocab actionspace.Vocabulary) (*Player, error) {
	metadata, err := os.ReadFile(metadataPath)
	if err != nil {
		return nil, fmt.Errorf("read metadata %s: %w", metadataPath, err)
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
			policy := model.PolicyVector(ctx.In("policy_model"), turnFeatures, occupiedTurns, virginGrid)
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
		backend:             backend,
		exec:                exec,
		fixedActionTensor:   fixedActionTensor,
		vocab:               vocab,
		BackendDescription:  backend.Description(),
		DeviceDescription:   backend.DeviceDescription(0),
		TrunkHiddenDims:     []int{model.DenseTrunkHidden0, model.DenseTrunkHidden1, model.DenseTrunkHidden2},
		TrunkOutputDim:      model.DenseTrunkOutputDim,
		PolicyVectorDim:     model.PolicyVectorDim,
		TrainableTailDim:    model.TrainableActionFeatureDim,
		HasPolicyOutputHead: true,
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
