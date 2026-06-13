package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/sam-bee/wordle-ml_backprop/internal/telemetry"
)

const defaultTags = "train/mean_loss,validation_delta_from_start"

func main() {
	flags := flag.NewFlagSet(filepath.Base(os.Args[0]), flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	runsRoot := flags.String("runs-root", filepath.Join("checkpoints", "runs"), "checkpoint runs root")
	latestRunFile := flags.String("latest-run-file", filepath.Join("checkpoints", "latest-run.txt"), "file containing the latest run id")
	runID := flags.String("run", "latest", "run id to read, or latest")
	tagsRaw := flags.String("tags", defaultTags, "comma-separated scalar tags to export")
	flags.Usage = func() {
		fmt.Fprintf(flags.Output(), "usage: %s [options]\n", flags.Name())
		flags.PrintDefaults()
	}

	if err := flags.Parse(os.Args[1:]); err != nil {
		os.Exit(2)
	}
	if flags.NArg() != 0 {
		flags.Usage()
		os.Exit(2)
	}

	resolvedRunID, err := resolveRunID(*runID, *latestRunFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "resolve run id: %v\n", err)
		os.Exit(1)
	}
	tags, err := parseTags(*tagsRaw)
	if err != nil {
		fmt.Fprintf(os.Stderr, "parse tags: %v\n", err)
		os.Exit(2)
	}

	eventDir := filepath.Join(*runsRoot, resolvedRunID, "tensorboard")
	points, err := telemetry.ReadScalarPoints(eventDir, tags)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read scalars: %v\n", err)
		os.Exit(1)
	}
	if err := requireTags(points, tags); err != nil {
		fmt.Fprintf(os.Stderr, "read scalars: %v\n", err)
		os.Exit(1)
	}
	if err := writeCSV(os.Stdout, resolvedRunID, points); err != nil {
		fmt.Fprintf(os.Stderr, "write csv: %v\n", err)
		os.Exit(1)
	}
}

func resolveRunID(runID, latestRunFile string) (string, error) {
	if runID == "latest" {
		content, err := os.ReadFile(latestRunFile)
		if err != nil {
			return "", fmt.Errorf("read latest run file %s: %w", latestRunFile, err)
		}
		runID = strings.TrimSpace(string(content))
	}
	if runID == "" {
		return "", fmt.Errorf("run id is empty")
	}
	if runID != filepath.Base(runID) || strings.ContainsAny(runID, `/\`) {
		return "", fmt.Errorf("run id %q must be a single path component", runID)
	}
	return runID, nil
}

func parseTags(raw string) ([]string, error) {
	parts := strings.Split(raw, ",")
	tags := make([]string, 0, len(parts))
	seen := make(map[string]bool, len(parts))
	for _, part := range parts {
		tag := strings.TrimSpace(part)
		if tag == "" {
			continue
		}
		if seen[tag] {
			continue
		}
		seen[tag] = true
		tags = append(tags, tag)
	}
	if len(tags) == 0 {
		return nil, fmt.Errorf("at least one tag is required")
	}
	return tags, nil
}

func requireTags(points []telemetry.ScalarPoint, tags []string) error {
	seen := make(map[string]bool, len(tags))
	for _, point := range points {
		seen[point.Tag] = true
	}
	var missing []string
	for _, tag := range tags {
		if !seen[tag] {
			missing = append(missing, tag)
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("no points found for tag(s): %s", strings.Join(missing, ", "))
	}
	return nil
}

func writeCSV(output *os.File, runID string, points []telemetry.ScalarPoint) error {
	writer := csv.NewWriter(output)
	if err := writer.Write([]string{"run_id", "series", "step", "wall_time", "value"}); err != nil {
		return err
	}
	for _, point := range points {
		if err := writer.Write([]string{
			runID,
			point.Tag,
			strconv.FormatInt(point.Step, 10),
			strconv.FormatFloat(point.WallTime, 'f', 9, 64),
			strconv.FormatFloat(point.Value, 'g', -1, 64),
		}); err != nil {
			return err
		}
	}
	writer.Flush()
	return writer.Error()
}
