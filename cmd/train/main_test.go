package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/sam-bee/wordle-ml_backprop/internal/data"
)

func TestResolveCheckpointPathsStartsFreshRunByDefault(t *testing.T) {
	root := t.TempDir()
	now := time.Date(2026, 6, 8, 5, 30, 15, 123456789, time.UTC)

	paths, err := resolveCheckpointPaths(root, false, now)
	if err != nil {
		t.Fatalf("resolveCheckpointPaths() error = %v", err)
	}
	if paths.Resume {
		t.Fatal("Resume = true, want false")
	}
	if paths.RunID != "run-20260608-053015.123456789" {
		t.Fatalf("RunID = %q", paths.RunID)
	}
	if paths.GoMLXDir != filepath.Join(root, checkpointRunsDir, paths.RunID, checkpointGoMLXDir) {
		t.Fatalf("GoMLXDir = %q", paths.GoMLXDir)
	}
	if paths.TensorBoardDir != filepath.Join(root, checkpointRunsDir, paths.RunID, checkpointTelemetryDir) {
		t.Fatalf("TensorBoardDir = %q", paths.TensorBoardDir)
	}
	if paths.ManifestPath != filepath.Join(root, checkpointRunsDir, paths.RunID, manifestFileName) {
		t.Fatalf("ManifestPath = %q", paths.ManifestPath)
	}
}

func TestResolveCheckpointPathsResumesLatestRun(t *testing.T) {
	root := t.TempDir()
	latestPath := filepath.Join(root, latestRunFileName)
	if err := os.WriteFile(latestPath, []byte("run-existing\n"), 0o600); err != nil {
		t.Fatalf("write latest run: %v", err)
	}

	paths, err := resolveCheckpointPaths(root, true, time.Time{})
	if err != nil {
		t.Fatalf("resolveCheckpointPaths() error = %v", err)
	}
	if !paths.Resume {
		t.Fatal("Resume = false, want true")
	}
	if paths.RunID != "run-existing" {
		t.Fatalf("RunID = %q", paths.RunID)
	}
	if paths.GoMLXDir != filepath.Join(root, checkpointRunsDir, "run-existing", checkpointGoMLXDir) {
		t.Fatalf("GoMLXDir = %q", paths.GoMLXDir)
	}
	if paths.TensorBoardDir != filepath.Join(root, checkpointRunsDir, "run-existing", checkpointTelemetryDir) {
		t.Fatalf("TensorBoardDir = %q", paths.TensorBoardDir)
	}
}

func TestResolveCheckpointPathsRejectsUnsafeLatestRun(t *testing.T) {
	root := t.TempDir()
	latestPath := filepath.Join(root, latestRunFileName)
	if err := os.WriteFile(latestPath, []byte("../outside\n"), 0o600); err != nil {
		t.Fatalf("write latest run: %v", err)
	}

	if _, err := resolveCheckpointPaths(root, true, time.Time{}); err == nil {
		t.Fatal("resolveCheckpointPaths() succeeded, want error")
	}
}

func TestSplitNamesForTrainingConfigDefaultsToCanonicalSplits(t *testing.T) {
	got := splitNamesForTrainingConfig(data.SplitTrain, data.SplitValidation)
	want := data.KnownSplits[:]

	if len(got) != len(want) {
		t.Fatalf("splitNamesForTrainingConfig() returned %d splits, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("splitNamesForTrainingConfig()[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestSplitNamesForTrainingConfigMiniUsesMiniOnce(t *testing.T) {
	got := splitNamesForTrainingConfig(data.SplitMini, data.SplitMini)
	want := []data.SplitName{data.SplitMini}

	if len(got) != len(want) {
		t.Fatalf("splitNamesForTrainingConfig() returned %d splits, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("splitNamesForTrainingConfig()[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}
