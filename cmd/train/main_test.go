package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestResolveCheckpointPathsStartsFreshRunByDefault(t *testing.T) {
	root := t.TempDir()
	now := time.Date(2026, 6, 8, 5, 30, 15, 123456789, time.UTC)

	paths, err := resolveCheckpointPaths(root, false, "", now)
	if err != nil {
		t.Fatalf("resolveCheckpointPaths() error = %v", err)
	}
	if paths.Resume {
		t.Fatal("Resume = true, want false")
	}
	if paths.RunID != "run-20260608-053015.123456789" {
		t.Fatalf("RunID = %q", paths.RunID)
	}
	if paths.RunLabel != "" {
		t.Fatalf("RunLabel = %q, want empty", paths.RunLabel)
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

func TestResolveCheckpointPathsAppendsRunLabel(t *testing.T) {
	root := t.TempDir()
	now := time.Date(2026, 6, 9, 18, 10, 31, 987654321, time.UTC)

	paths, err := resolveCheckpointPaths(root, false, "  Deeper dense: 1 x 256 + 1 x 128  ", now)
	if err != nil {
		t.Fatalf("resolveCheckpointPaths() error = %v", err)
	}
	if paths.RunLabel != "deeper-dense-1-x-256-1-x-128" {
		t.Fatalf("RunLabel = %q", paths.RunLabel)
	}
	if paths.RunID != "run-20260609-181031.987654321-deeper-dense-1-x-256-1-x-128" {
		t.Fatalf("RunID = %q", paths.RunID)
	}
	if paths.TensorBoardDir != filepath.Join(root, checkpointRunsDir, paths.RunID, checkpointTelemetryDir) {
		t.Fatalf("TensorBoardDir = %q", paths.TensorBoardDir)
	}
}

func TestResolveCheckpointPathsResumesLatestRun(t *testing.T) {
	root := t.TempDir()
	latestPath := filepath.Join(root, latestRunFileName)
	if err := os.WriteFile(latestPath, []byte("run-20260609-181031.987654321-small-trunk\n"), 0o600); err != nil {
		t.Fatalf("write latest run: %v", err)
	}

	paths, err := resolveCheckpointPaths(root, true, "", time.Time{})
	if err != nil {
		t.Fatalf("resolveCheckpointPaths() error = %v", err)
	}
	if !paths.Resume {
		t.Fatal("Resume = false, want true")
	}
	if paths.RunID != "run-20260609-181031.987654321-small-trunk" {
		t.Fatalf("RunID = %q", paths.RunID)
	}
	if paths.RunLabel != "small-trunk" {
		t.Fatalf("RunLabel = %q", paths.RunLabel)
	}
	if paths.GoMLXDir != filepath.Join(root, checkpointRunsDir, paths.RunID, checkpointGoMLXDir) {
		t.Fatalf("GoMLXDir = %q", paths.GoMLXDir)
	}
	if paths.TensorBoardDir != filepath.Join(root, checkpointRunsDir, paths.RunID, checkpointTelemetryDir) {
		t.Fatalf("TensorBoardDir = %q", paths.TensorBoardDir)
	}
}

func TestResolveCheckpointPathsRejectsRunLabelWithResume(t *testing.T) {
	root := t.TempDir()
	latestPath := filepath.Join(root, latestRunFileName)
	if err := os.WriteFile(latestPath, []byte("run-existing\n"), 0o600); err != nil {
		t.Fatalf("write latest run: %v", err)
	}

	if _, err := resolveCheckpointPaths(root, true, "new-label", time.Time{}); err == nil {
		t.Fatal("resolveCheckpointPaths() succeeded with resume and run label, want error")
	}
}

func TestResolveCheckpointPathsRejectsEmptySanitizedRunLabel(t *testing.T) {
	root := t.TempDir()

	if _, err := resolveCheckpointPaths(root, false, "!!!", time.Time{}); err == nil {
		t.Fatal("resolveCheckpointPaths() succeeded with empty sanitized run label, want error")
	}
}

func TestResolveCheckpointPathsRejectsUnsafeLatestRun(t *testing.T) {
	root := t.TempDir()
	latestPath := filepath.Join(root, latestRunFileName)
	if err := os.WriteFile(latestPath, []byte("../outside\n"), 0o600); err != nil {
		t.Fatalf("write latest run: %v", err)
	}

	if _, err := resolveCheckpointPaths(root, true, "", time.Time{}); err == nil {
		t.Fatal("resolveCheckpointPaths() succeeded, want error")
	}
}
