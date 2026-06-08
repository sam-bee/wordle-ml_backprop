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
