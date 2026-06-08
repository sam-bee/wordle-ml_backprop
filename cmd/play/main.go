package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/xla"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/checkpoints"
	enginegame "github.com/sam-bee/wordle-ml_game-engine/game"
	enginewords "github.com/sam-bee/wordle-ml_game-engine/words"

	"github.com/sam-bee/wordle-ml_backprop/internal/actionspace"
	"github.com/sam-bee/wordle-ml_backprop/internal/data"
	"github.com/sam-bee/wordle-ml_backprop/internal/model"
	"github.com/sam-bee/wordle-ml_backprop/internal/training"
)

const (
	defaultTopN     = 10
	defaultMaxTurns = 6
	backendConfig   = "cuda"
)

func main() {
	flags := flag.NewFlagSet(filepath.Base(os.Args[0]), flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	modelWeights := flags.String("model-weights", "", "path to GoMLX checkpoint .bin tensor payload")
	modelMetadata := flags.String("model-metadata", "", "path to GoMLX checkpoint .json metadata")
	solutionFlag := flags.String("solution", "", "hidden Wordle solution word")
	topN := flags.Int("top-n", defaultTopN, "number of top model guesses to print per turn")
	maxTurns := flags.Int("max-turns", defaultMaxTurns, "maximum game turns to play")
	flags.Usage = func() {
		fmt.Fprintf(flags.Output(), "usage: %s --model-weights checkpoint.bin --model-metadata checkpoint.json --solution CHANT [options]\n", flags.Name())
		flags.PrintDefaults()
	}

	if err := flags.Parse(os.Args[1:]); err != nil {
		os.Exit(2)
	}
	if *modelWeights == "" || *modelMetadata == "" || *solutionFlag == "" {
		flags.Usage()
		os.Exit(2)
	}
	if *topN <= 0 {
		fmt.Fprintf(os.Stderr, "top-n must be greater than 0, got %d\n", *topN)
		os.Exit(2)
	}
	if *maxTurns <= 0 || *maxTurns > data.MaxTurns+1 {
		fmt.Fprintf(os.Stderr, "max-turns must be in 1..%d, got %d\n", data.MaxTurns+1, *maxTurns)
		os.Exit(2)
	}

	solution, err := parseSolution(*solutionFlag)
	if err != nil {
		fmt.Fprintf(os.Stderr, "solution: %v\n", err)
		os.Exit(2)
	}
	if err := validateKnownSolution(solution); err != nil {
		fmt.Fprintf(os.Stderr, "solution: %v\n", err)
		os.Exit(2)
	}

	vocab, err := actionspace.Load()
	if err != nil {
		fmt.Fprintf(os.Stderr, "load action space: %v\n", err)
		os.Exit(1)
	}

	player, err := newPolicyPlayer(*modelWeights, *modelMetadata, vocab)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load model: %v\n", err)
		os.Exit(1)
	}
	defer player.Close()

	fmt.Println("wordle backprop player starting")
	fmt.Printf("model_weights=%q\n", *modelWeights)
	fmt.Printf("model_metadata=%q\n", *modelMetadata)
	fmt.Printf("solution=%s max_turns=%d top_n=%d\n", solution, *maxTurns, *topN)
	fmt.Printf("backend=%q device=%q action_count=%d\n", player.BackendDescription, player.DeviceDescription, len(vocab.Words))
	fmt.Println("selection=highest-scoring action not guessed before; no candidate-list filtering")

	if err := play(solution, *maxTurns, *topN, player); err != nil {
		fmt.Fprintf(os.Stderr, "play: %v\n", err)
		os.Exit(1)
	}
}

type policyPlayer struct {
	backend            backends.Backend
	exec               *context.Exec
	fixedActionTensor  *tensors.Tensor
	vocab              actionspace.Vocabulary
	BackendDescription string
	DeviceDescription  string
}

type scoredAction struct {
	Word        data.Word
	ActionIndex int
	Logit       float32
	Probability float64
}

type gameTurn struct {
	Guess    data.Word
	Feedback [data.WordLength]data.Feedback
}

func newPolicyPlayer(weightsPath, metadataPath string, vocab actionspace.Vocabulary) (*policyPlayer, error) {
	metadata, err := os.ReadFile(metadataPath)
	if err != nil {
		return nil, fmt.Errorf("read metadata %s: %w", metadataPath, err)
	}
	if err := validateCheckpointMetadata(metadataPath, metadata); err != nil {
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
			return model.PolicyModel(ctx, nil, []*graph.Node{turnFeatures, occupiedTurns, virginGrid, fixedActionFeatures})[0]
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

	return &policyPlayer{
		backend:            backend,
		exec:               exec,
		fixedActionTensor:  fixedActionTensor,
		vocab:              vocab,
		BackendDescription: backend.Description(),
		DeviceDescription:  backend.DeviceDescription(0),
	}, nil
}

func (player *policyPlayer) Close() {
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

func (player *policyPlayer) Predict(input data.BatchInput) ([]scoredAction, error) {
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
	return rankActions(player.vocab.Words, logits), nil
}

func play(solution enginewords.Word, maxTurns, topN int, player *policyPlayer) error {
	var turns []gameTurn
	guessed := make(map[data.Word]bool)

	for turnIndex := 0; turnIndex < maxTurns; turnIndex++ {
		input, err := batchInputFromTurns(turns)
		if err != nil {
			return err
		}

		ranked, err := player.Predict(input)
		if err != nil {
			return fmt.Errorf("turn %d inference: %w", turnIndex+1, err)
		}

		guess, selected, err := firstUnguessed(ranked, guessed)
		if err != nil {
			return err
		}

		fmt.Printf("\nturn %d\n", turnIndex+1)
		printTopActions(ranked, guessed, topN)

		engineGuess := enginewords.Word(guess.String())
		engineFeedback := enginegame.GetFeedback(solution, engineGuess)
		feedback := engineFeedback.String()
		fmt.Printf("selected: rank=%d guess=%s logit=%.6f probability=%.6f feedback=%s\n", selected+1, guess, ranked[selected].Logit, ranked[selected].Probability, feedback)

		if feedback == "GGGGG" {
			fmt.Printf("\nsolved in %d turns\n", turnIndex+1)
			return nil
		}

		parsedFeedback, err := parseFeedback(feedback)
		if err != nil {
			return err
		}
		turns = append(turns, gameTurn{Guess: guess, Feedback: parsedFeedback})
		guessed[guess] = true
	}

	fmt.Printf("\nfailed to solve within %d turns; solution=%s\n", maxTurns, solution)
	return nil
}

func batchInputFromTurns(turns []gameTurn) (data.BatchInput, error) {
	var input data.BatchInput
	if len(turns) > data.MaxTurns {
		return input, fmt.Errorf("state has %d previous turns, max supported is %d", len(turns), data.MaxTurns)
	}
	input.TurnDepth = uint8(len(turns))
	for index, turn := range turns {
		input.PreviousGuessWords[index] = turn.Guess
		input.PreviousFeedback[index] = turn.Feedback
	}
	return input, nil
}

func firstUnguessed(ranked []scoredAction, guessed map[data.Word]bool) (data.Word, int, error) {
	for index, action := range ranked {
		if !guessed[action.Word] {
			return action.Word, index, nil
		}
	}
	return data.Word{}, 0, fmt.Errorf("model action space has no unguessed words")
}

func printTopActions(ranked []scoredAction, guessed map[data.Word]bool, topN int) {
	if topN > len(ranked) {
		topN = len(ranked)
	}
	fmt.Printf("top %d actions:\n", topN)
	for index := 0; index < topN; index++ {
		action := ranked[index]
		marker := ""
		if guessed[action.Word] {
			marker = " already_guessed"
		}
		fmt.Printf("  %2d. %s logit=% .6f probability=%.6f%s\n", index+1, action.Word, action.Logit, action.Probability, marker)
	}
}

func rankActions(words []data.Word, logits []float32) []scoredAction {
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

	ranked := make([]scoredAction, len(logits))
	for index, logit := range logits {
		ranked[index] = scoredAction{
			Word:        words[index],
			ActionIndex: index,
			Logit:       logit,
			Probability: probabilities[index] / expSum,
		}
	}
	sort.Slice(ranked, func(i, j int) bool {
		return ranked[i].Logit > ranked[j].Logit
	})
	return ranked
}

func parseSolution(value string) (enginewords.Word, error) {
	value = strings.ToUpper(strings.TrimSpace(value))
	if len(value) != data.WordLength {
		return "", fmt.Errorf("solution %q has length %d, expected %d", value, len(value), data.WordLength)
	}
	for index := range value {
		if value[index] < 'A' || value[index] > 'Z' {
			return "", fmt.Errorf("solution %q contains non-uppercase ASCII byte %d at position %d", value, value[index], index)
		}
	}
	word, err := enginewords.NewWord(value)
	if err != nil {
		return "", err
	}
	return word, nil
}

func validateKnownSolution(solution enginewords.Word) error {
	solutions, err := enginewords.GetValidSolutions()
	if err != nil {
		return fmt.Errorf("load valid solutions: %w", err)
	}
	for _, candidate := range solutions {
		if candidate == solution {
			return nil
		}
	}
	return fmt.Errorf("%s is not in the game-engine valid solution list", solution)
}

func parseFeedback(value string) ([data.WordLength]data.Feedback, error) {
	var feedback [data.WordLength]data.Feedback
	if len(value) != data.WordLength {
		return feedback, fmt.Errorf("feedback %q has length %d, expected %d", value, len(value), data.WordLength)
	}
	for index := range value {
		switch value[index] {
		case 'G':
			feedback[index] = data.FeedbackGreen
		case 'Y':
			feedback[index] = data.FeedbackYellow
		case '-':
			feedback[index] = data.FeedbackGrey
		default:
			return feedback, fmt.Errorf("feedback %q contains unsupported byte %d at position %d", value, value[index], index)
		}
	}
	return feedback, nil
}

func validateCheckpointMetadata(path string, content []byte) error {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(content, &raw); err != nil {
		return fmt.Errorf("parse checkpoint metadata %s: %w", path, err)
	}
	if _, ok := raw["latest_gomlx_checkpoint"]; ok {
		return fmt.Errorf("%s looks like a project manifest; pass the GoMLX checkpoint .json file instead", path)
	}
	if _, ok := raw["Variables"]; !ok {
		return fmt.Errorf("%s does not look like a GoMLX checkpoint metadata file: missing Variables", path)
	}
	return nil
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
