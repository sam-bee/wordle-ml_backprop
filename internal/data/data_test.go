package data

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadSplitParsesFixture(t *testing.T) {
	dir := t.TempDir()
	metadata := fixtureMetadata()
	records := fixtureRecords(t, int(metadata.RecordCount))
	writeFixture(t, dir, metadata, records)

	split, err := LoadSplit(dir)
	if err != nil {
		t.Fatalf("LoadSplit() error = %v", err)
	}

	if split.Metadata.Split != SplitValidation {
		t.Fatalf("split.Metadata.Split = %q, want %q", split.Metadata.Split, SplitValidation)
	}
	if split.SampleCount() != int(metadata.RecordCount) {
		t.Fatalf("split.SampleCount() = %d, want %d", split.SampleCount(), metadata.RecordCount)
	}

	first := split.Samples[0]
	if first.SolutionWord.String() != "CRANE" {
		t.Fatalf("first.SolutionWord = %q, want CRANE", first.SolutionWord)
	}
	if first.TurnDepth != 1 {
		t.Fatalf("first.TurnDepth = %d, want 1", first.TurnDepth)
	}
	if first.PreviousGuessWords[0].String() != "SLATE" {
		t.Fatalf("first.PreviousGuessWords[0] = %q, want SLATE", first.PreviousGuessWords[0])
	}
	if first.PreviousFeedback[0] != [WordLength]Feedback{FeedbackGrey, FeedbackYellow, FeedbackGreen, FeedbackGrey, FeedbackYellow} {
		t.Fatalf("first.PreviousFeedback[0] = %v", first.PreviousFeedback[0])
	}
	if first.TopKGuessWords[0].String() != "TRACE" {
		t.Fatalf("first.TopKGuessWords[0] = %q, want TRACE", first.TopKGuessWords[0])
	}
	if first.TopKReductionRatios[0] != 0.5 {
		t.Fatalf("first.TopKReductionRatios[0] = %g, want 0.5", first.TopKReductionRatios[0])
	}
	if first.TopKWorstCaseSizes[0] != 5 {
		t.Fatalf("first.TopKWorstCaseSizes[0] = %d, want 5", first.TopKWorstCaseSizes[0])
	}
}

func TestLoadSplitParsesMiniSplit(t *testing.T) {
	dir := t.TempDir()
	metadata := fixtureMetadata()
	metadata.Split = SplitMini
	metadata.SplitID = 4
	metadata.BinaryFile = "wordle-mini.bin"
	records := fixtureRecords(t, int(metadata.RecordCount))
	writeFixture(t, dir, metadata, records)

	split, err := LoadSplit(dir)
	if err != nil {
		t.Fatalf("LoadSplit() error = %v", err)
	}

	if split.Metadata.Split != SplitMini {
		t.Fatalf("split.Metadata.Split = %q, want %q", split.Metadata.Split, SplitMini)
	}
}

func TestLoadSplitRejectsUnsupportedMetadataVersion(t *testing.T) {
	dir := t.TempDir()
	metadata := fixtureMetadata()
	metadata.Version = 99
	writeMetadata(t, dir, metadata)

	_, err := LoadSplit(dir)
	if err == nil {
		t.Fatal("LoadSplit() error = nil, want unsupported version error")
	}
	if !strings.Contains(err.Error(), "unsupported metadata version 99") {
		t.Fatalf("LoadSplit() error = %v, want unsupported version", err)
	}
}

func TestLoadSplitRejectsWrongBinarySize(t *testing.T) {
	dir := t.TempDir()
	metadata := fixtureMetadata()
	records := fixtureRecords(t, int(metadata.RecordCount))
	writeFixture(t, dir, metadata, records)

	binaryPath := filepath.Join(dir, metadata.BinaryFile)
	stat, err := os.Stat(binaryPath)
	if err != nil {
		t.Fatalf("stat fixture binary: %v", err)
	}
	if err := os.Truncate(binaryPath, stat.Size()-1); err != nil {
		t.Fatalf("truncate fixture binary: %v", err)
	}

	_, err = LoadSplit(dir)
	if err == nil {
		t.Fatal("LoadSplit() error = nil, want size error")
	}
	if !strings.Contains(err.Error(), "binary file size") {
		t.Fatalf("LoadSplit() error = %v, want binary file size", err)
	}
}

func fixtureMetadata() Metadata {
	return Metadata{
		Version:                 FormatVersion,
		Split:                   SplitValidation,
		SplitID:                 2,
		BinaryFile:              "wordle-validation.bin",
		RecordCount:             RecordsPerSolution,
		HeaderSizeBytes:         HeaderSizeBytes,
		RecordSizeBytes:         RecordSizeBytes,
		TopK:                    TopK,
		MaxTurns:                MaxTurns,
		GuessVocabSize:          GuessVocabSize,
		GlobalSolutionVocabSize: GlobalSolutionVocabSize,
		SolutionCount:           1,
		SolutionIDs:             []uint32{48},
		RecordsPerSolution:      RecordsPerSolution,
		RecordsPerDepth:         RecordsPerDepth,
		IncludesOpeningState:    false,
		PaddingFeedbackValue:    PaddingFeedbackValue,
		WordlistHash:            "fixture",
		Seed:                    1,
		TeacherName:             "fixture_teacher",
	}
}

func fixtureRecords(t *testing.T, count int) []Sample {
	t.Helper()

	records := make([]Sample, count)
	for i := range records {
		var sample Sample
		sample.SolutionWord = fixtureWord(t, "CRANE")
		sample.TurnDepth = 1
		sample.PreviousGuessWords[0] = fixtureWord(t, "SLATE")
		sample.PreviousFeedback[0] = [WordLength]Feedback{FeedbackGrey, FeedbackYellow, FeedbackGreen, FeedbackGrey, FeedbackYellow}
		for turn := 1; turn < MaxTurns; turn++ {
			for pos := 0; pos < WordLength; pos++ {
				sample.PreviousFeedback[turn][pos] = PaddingFeedbackValue
			}
		}
		sample.ShortlistSizeBefore = 10
		for rank := 0; rank < TopK; rank++ {
			sample.TopKGuessWords[rank] = fixtureWord(t, "TRACE")
			sample.TopKReductionRatios[rank] = 0.5
			sample.TopKWorstCaseSizes[rank] = 5
		}
		records[i] = sample
	}
	return records
}

func fixtureWord(t *testing.T, value string) Word {
	t.Helper()
	if len(value) != WordLength {
		t.Fatalf("fixture word %q has length %d, want %d", value, len(value), WordLength)
	}
	var word Word
	copy(word[:], value)
	return word
}

func writeFixture(t *testing.T, dir string, metadata Metadata, records []Sample) {
	t.Helper()
	writeMetadata(t, dir, metadata)
	writeBinary(t, filepath.Join(dir, metadata.BinaryFile), metadata, records)
}

func writeMetadata(t *testing.T, dir string, metadata Metadata) {
	t.Helper()
	content, err := json.MarshalIndent(metadata, "", "  ")
	if err != nil {
		t.Fatalf("marshal metadata: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "wordle-validation.json"), content, 0o644); err != nil {
		t.Fatalf("write metadata: %v", err)
	}
}

func writeBinary(t *testing.T, path string, metadata Metadata, records []Sample) {
	t.Helper()
	var buf bytes.Buffer
	buf.WriteString(binaryMagic)
	writeUint32(t, &buf, metadata.Version)
	writeUint32(t, &buf, uint32(len(records)))
	writeUint32(t, &buf, metadata.TopK)
	writeUint32(t, &buf, metadata.MaxTurns)
	writeUint32(t, &buf, metadata.GuessVocabSize)
	writeUint32(t, &buf, metadata.SolutionCount)
	writeUint32(t, &buf, metadata.SplitID)
	buf.Write(make([]byte, HeaderSizeBytes-32))

	for _, record := range records {
		buf.Write(marshalRecord(t, record))
	}
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatalf("write binary: %v", err)
	}
}

func marshalRecord(t *testing.T, sample Sample) []byte {
	t.Helper()
	var buf bytes.Buffer
	buf.Write(sample.SolutionWord[:])
	buf.WriteByte(sample.TurnDepth)
	for _, word := range sample.PreviousGuessWords {
		buf.Write(word[:])
	}
	for _, turn := range sample.PreviousFeedback {
		for _, feedback := range turn {
			buf.WriteByte(byte(feedback))
		}
	}
	writeUint16(t, &buf, sample.ShortlistSizeBefore)
	for _, word := range sample.TopKGuessWords {
		buf.Write(word[:])
	}
	for _, ratio := range sample.TopKReductionRatios {
		writeUint32(t, &buf, math.Float32bits(ratio))
	}
	for _, worstCaseSize := range sample.TopKWorstCaseSizes {
		writeUint16(t, &buf, worstCaseSize)
	}
	if buf.Len() != RecordSizeBytes {
		t.Fatalf("marshaled record is %d bytes, want %d", buf.Len(), RecordSizeBytes)
	}
	return buf.Bytes()
}

func writeUint16(t *testing.T, buf *bytes.Buffer, value uint16) {
	t.Helper()
	if err := binary.Write(buf, binary.LittleEndian, value); err != nil {
		t.Fatalf("write uint16: %v", err)
	}
}

func writeUint32(t *testing.T, buf *bytes.Buffer, value uint32) {
	t.Helper()
	if err := binary.Write(buf, binary.LittleEndian, value); err != nil {
		t.Fatalf("write uint32: %v", err)
	}
}
