package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestResolveRunIDReadsLatestRun(t *testing.T) {
	path := filepath.Join(t.TempDir(), "latest-run.txt")
	if err := os.WriteFile(path, []byte("run-20260613-165820.441895744\n"), 0o600); err != nil {
		t.Fatalf("write latest run: %v", err)
	}

	runID, err := resolveRunID("latest", path)
	if err != nil {
		t.Fatalf("resolveRunID() error = %v", err)
	}
	if runID != "run-20260613-165820.441895744" {
		t.Fatalf("resolveRunID() = %q", runID)
	}
}

func TestResolveRunIDRejectsPathTraversal(t *testing.T) {
	if _, err := resolveRunID("../outside", "unused"); err == nil {
		t.Fatal("resolveRunID() error = nil, want path traversal error")
	}
}

func TestParseTagsTrimsAndDeduplicates(t *testing.T) {
	tags, err := parseTags(" train/mean_loss, validation_delta_from_start,train/mean_loss ")
	if err != nil {
		t.Fatalf("parseTags() error = %v", err)
	}
	want := []string{"train/mean_loss", "validation_delta_from_start"}
	if len(tags) != len(want) {
		t.Fatalf("parseTags() returned %d tags, want %d", len(tags), len(want))
	}
	for i := range want {
		if tags[i] != want[i] {
			t.Fatalf("parseTags()[%d] = %q, want %q", i, tags[i], want[i])
		}
	}
}
