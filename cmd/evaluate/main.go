package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sam-bee/wordle-ml_backprop/internal/actionspace"
	"github.com/sam-bee/wordle-ml_backprop/internal/data"
	"github.com/sam-bee/wordle-ml_backprop/internal/evaluation"
	"github.com/sam-bee/wordle-ml_backprop/internal/inference"
)

func main() {
	flags := flag.NewFlagSet(filepath.Base(os.Args[0]), flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	modelWeights := flags.String("model-weights", "", "path to GoMLX checkpoint .bin tensor payload")
	modelMetadata := flags.String("model-metadata", "", "path to GoMLX checkpoint .json metadata")
	dataRoot := flags.String("data-root", "data", "training data root containing the validation split")
	limit := flags.Int("limit", 0, "maximum validation solutions to evaluate; 0 means all")
	jsonOutput := flags.Bool("json", false, "write machine-readable JSON output")
	progressEvery := flags.Int("progress-every", 25, "write progress to stderr every n solutions; 0 disables progress")
	flags.Usage = func() {
		fmt.Fprintf(flags.Output(), "usage: %s --model-weights checkpoint.bin --model-metadata checkpoint.json [options]\n", flags.Name())
		flags.PrintDefaults()
	}

	if err := flags.Parse(os.Args[1:]); err != nil {
		os.Exit(2)
	}
	if *modelWeights == "" || *modelMetadata == "" {
		flags.Usage()
		os.Exit(2)
	}
	if *limit < 0 {
		fmt.Fprintf(os.Stderr, "limit must be >= 0, got %d\n", *limit)
		os.Exit(2)
	}
	if *progressEvery < 0 {
		fmt.Fprintf(os.Stderr, "progress-every must be >= 0, got %d\n", *progressEvery)
		os.Exit(2)
	}

	if err := run(*modelWeights, *modelMetadata, *dataRoot, *limit, *progressEvery, *jsonOutput); err != nil {
		fmt.Fprintf(os.Stderr, "evaluate: %v\n", err)
		os.Exit(1)
	}
}

func run(weightsPath, metadataPath, dataRoot string, limit, progressEvery int, jsonOutput bool) error {
	validationDir := filepath.Join(dataRoot, string(data.SplitValidation))
	validationSplit, err := data.LoadSplit(validationDir)
	if err != nil {
		return fmt.Errorf("load validation split: %w", err)
	}
	solutions, err := evaluation.UniqueValidationSolutions(validationSplit)
	if err != nil {
		return err
	}
	if limit > 0 && limit < len(solutions) {
		solutions = solutions[:limit]
	}

	vocab, err := actionspace.Load()
	if err != nil {
		return fmt.Errorf("load action space: %w", err)
	}
	player, err := inference.NewPlayer(weightsPath, metadataPath, vocab)
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}
	defer player.Close()

	started := time.Now()
	results := make([]evaluation.EpisodeResult, 0, len(solutions))
	for index, solution := range solutions {
		result, err := evaluation.RunEpisode(solution, evaluation.MaxWordleTurns, player)
		if err != nil {
			return fmt.Errorf("solution %d %s: %w", index+1, solution, err)
		}
		results = append(results, result)
		if progressEvery > 0 && (index+1)%progressEvery == 0 {
			fmt.Fprintf(os.Stderr, "evaluated %d/%d validation solutions\n", index+1, len(solutions))
		}
	}

	summary := evaluation.Summarize(results, validationDir)
	if jsonOutput {
		return writeJSON(summary, weightsPath, metadataPath, player, time.Since(started))
	}
	writeText(summary, weightsPath, metadataPath, player, time.Since(started))
	return nil
}

func writeText(summary evaluation.Summary, weightsPath, metadataPath string, player *inference.Player, elapsed time.Duration) {
	fmt.Println("wordle validation evaluator")
	fmt.Printf("model_weights=%q\n", weightsPath)
	fmt.Printf("model_metadata=%q\n", metadataPath)
	fmt.Printf("validation_source=%q\n", summary.ValidationSource)
	fmt.Printf(
		"backend=%q device=%q trunk_hidden_dims=%s trunk_output_dim=%d policy_vector_dim=%d trainable_tail_dim=%d policy_output_head=%t\n",
		player.BackendDescription,
		player.DeviceDescription,
		formatInts(player.TrunkHiddenDims),
		player.TrunkOutputDim,
		player.PolicyVectorDim,
		player.TrainableTailDim,
		player.HasPolicyOutputHead,
	)
	fmt.Printf("selection=%q\n", summary.Selection)
	fmt.Printf("validation_score_percent=%.4f\n", summary.ScorePercent)
	fmt.Printf("raw_score=%.4f\n", summary.RawScore)
	fmt.Printf("max_score=%.4f\n", summary.MaxScore)
	fmt.Printf("win_score=%.4f\n", summary.WinScore)
	fmt.Printf("loss_credit_score=%.4f\n", summary.LossCreditScore)
	fmt.Printf("solutions=%d\n", summary.Solutions)
	fmt.Printf("wins=%d\n", summary.Wins)
	fmt.Printf("losses=%d\n", summary.Losses)
	fmt.Printf("average_turns_on_wins=%.4f\n", summary.AverageWinTurns)
	fmt.Printf("turns_1=%d turns_2=%d turns_3=%d turns_4=%d turns_5=%d turns_6=%d\n", summary.Turns1, summary.Turns2, summary.Turns3, summary.Turns4, summary.Turns5, summary.Turns6)
	fmt.Printf("elapsed=%s\n", elapsed.Round(time.Millisecond))
}

func writeJSON(summary evaluation.Summary, weightsPath, metadataPath string, player *inference.Player, elapsed time.Duration) error {
	output := struct {
		evaluation.Summary
		ModelWeights        string `json:"model_weights"`
		ModelMetadata       string `json:"model_metadata"`
		TrunkHiddenDims     []int  `json:"trunk_hidden_dims"`
		TrunkOutputDim      int    `json:"trunk_output_dim"`
		PolicyVectorDim     int    `json:"policy_vector_dim"`
		TrainableTailDim    int    `json:"trainable_tail_dim"`
		HasPolicyOutputHead bool   `json:"has_policy_output_head"`
		ElapsedMillis       int64  `json:"elapsed_millis"`
	}{
		Summary:             summary,
		ModelWeights:        weightsPath,
		ModelMetadata:       metadataPath,
		TrunkHiddenDims:     player.TrunkHiddenDims,
		TrunkOutputDim:      player.TrunkOutputDim,
		PolicyVectorDim:     player.PolicyVectorDim,
		TrainableTailDim:    player.TrainableTailDim,
		HasPolicyOutputHead: player.HasPolicyOutputHead,
		ElapsedMillis:       elapsed.Milliseconds(),
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	return encoder.Encode(output)
}

func formatInts(values []int) string {
	parts := make([]string, len(values))
	for index, value := range values {
		parts[index] = fmt.Sprintf("%d", value)
	}
	return "[" + strings.Join(parts, ",") + "]"
}
