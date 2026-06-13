package data

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
)

const (
	FormatVersion uint32 = 2

	HeaderSizeBytes = 64
	RecordSizeBytes = 234

	WordLength = 5
	MaxTurns   = 5
	TopK       = 16

	GuessVocabSize          = 4739
	GlobalSolutionVocabSize = 2309
	RecordsPerSolution      = 25
	RecordsPerDepth         = 5

	PaddingFeedbackValue Feedback = 255
)

const (
	binaryMagic        = "WDIT"
	computedRecordSize = WordLength + 1 + MaxTurns*WordLength + MaxTurns*WordLength + 2 + TopK*WordLength + TopK*4 + TopK*2
)

type SplitName string

const (
	SplitTrain      SplitName = "train"
	SplitValidation SplitName = "validation"
	SplitTest       SplitName = "test"
	SplitMini       SplitName = "mini"
)

var KnownSplits = [...]SplitName{SplitTrain, SplitValidation, SplitTest}

type Metadata struct {
	Version                   uint32    `json:"version"`
	Split                     SplitName `json:"split"`
	SplitID                   uint32    `json:"split_id"`
	BinaryFile                string    `json:"binary_file"`
	RecordCount               uint32    `json:"record_count"`
	HeaderSizeBytes           uint32    `json:"header_size_bytes"`
	RecordSizeBytes           uint32    `json:"record_size_bytes"`
	TopK                      uint32    `json:"top_k"`
	MaxTurns                  uint32    `json:"max_turns"`
	GuessVocabSize            uint32    `json:"guess_vocab_size"`
	GlobalSolutionVocabSize   uint32    `json:"global_solution_vocab_size"`
	SolutionCount             uint32    `json:"solution_count"`
	SolutionIDs               []uint32  `json:"solution_ids"`
	RecordsPerSolution        uint32    `json:"records_per_solution"`
	RecordsPerDepth           uint32    `json:"records_per_depth"`
	IncludesOpeningState      bool      `json:"includes_opening_state"`
	OpeningSolutionWord       string    `json:"opening_solution_word"`
	PaddingWord               string    `json:"padding_word"`
	PaddingFeedbackValue      Feedback  `json:"padding_feedback_value"`
	WordlistHash              string    `json:"wordlist_hash"`
	GeneratorCommit           string    `json:"generator_commit"`
	GeneratorWorkingTreeDirty bool      `json:"generator_working_tree_dirty"`
	GeneratedAtUTC            string    `json:"generated_at_utc"`
	Seed                      int64     `json:"seed"`
	TeacherName               string    `json:"teacher_name"`
	ScoreMeaning              string    `json:"score_meaning"`
	WordEncoding              string    `json:"word_encoding"`
	FeedbackConvention        string    `json:"feedback_convention"`
}

type Header struct {
	Version        uint32
	RecordCount    uint32
	TopK           uint32
	MaxTurns       uint32
	GuessVocabSize uint32
	SolutionCount  uint32
	SplitID        uint32
}

type Split struct {
	Dir      string
	Metadata Metadata
	Header   Header
	Samples  []Sample
}

type Word [WordLength]byte

func (w Word) IsEmpty() bool {
	for _, b := range w {
		if b != 0 {
			return false
		}
	}
	return true
}

func (w Word) String() string {
	if w.IsEmpty() {
		return ""
	}
	return string(w[:])
}

type Feedback uint8

const (
	FeedbackGrey   Feedback = 0
	FeedbackYellow Feedback = 1
	FeedbackGreen  Feedback = 2
)

type Sample struct {
	SolutionWord        Word
	TurnDepth           uint8
	PreviousGuessWords  [MaxTurns]Word
	PreviousFeedback    [MaxTurns][WordLength]Feedback
	ShortlistSizeBefore uint16
	TopKGuessWords      [TopK]Word
	TopKReductionRatios [TopK]float32
	TopKWorstCaseSizes  [TopK]uint16
}

func LoadSplit(dir string) (*Split, error) {
	metadataPath, err := findMetadataFile(dir)
	if err != nil {
		return nil, err
	}

	metadata, err := loadMetadata(metadataPath)
	if err != nil {
		return nil, err
	}
	if err := validateMetadata(metadata); err != nil {
		return nil, fmt.Errorf("%s: %w", metadataPath, err)
	}

	binaryPath, err := binaryPathForMetadata(dir, metadata)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", metadataPath, err)
	}

	header, samples, err := loadBinary(binaryPath, metadata)
	if err != nil {
		return nil, err
	}

	return &Split{
		Dir:      dir,
		Metadata: metadata,
		Header:   header,
		Samples:  samples,
	}, nil
}

func (s *Split) SampleCount() int {
	if s == nil {
		return 0
	}
	return len(s.Samples)
}

func findMetadataFile(dir string) (string, error) {
	matches, err := filepath.Glob(filepath.Join(dir, "*.json"))
	if err != nil {
		return "", fmt.Errorf("find metadata in %s: %w", dir, err)
	}
	sort.Strings(matches)
	switch len(matches) {
	case 0:
		return "", fmt.Errorf("no metadata JSON file found in %s", dir)
	case 1:
		return matches[0], nil
	default:
		return "", fmt.Errorf("expected one metadata JSON file in %s, found %d", dir, len(matches))
	}
}

func loadMetadata(path string) (Metadata, error) {
	var metadata Metadata
	content, err := os.ReadFile(path)
	if err != nil {
		return metadata, fmt.Errorf("read metadata %s: %w", path, err)
	}
	if err := json.Unmarshal(content, &metadata); err != nil {
		return metadata, fmt.Errorf("parse metadata %s: %w", path, err)
	}
	return metadata, nil
}

func validateMetadata(metadata Metadata) error {
	if metadata.Version != FormatVersion {
		return fmt.Errorf("unsupported metadata version %d, expected %d", metadata.Version, FormatVersion)
	}
	expectedSplitID, ok := expectedSplitID(metadata.Split)
	if !ok {
		return fmt.Errorf("unsupported split %q", metadata.Split)
	}
	if metadata.SplitID != expectedSplitID {
		return fmt.Errorf("split %q has split_id %d, expected %d", metadata.Split, metadata.SplitID, expectedSplitID)
	}
	if metadata.BinaryFile == "" {
		return errors.New("binary_file is required")
	}
	if metadata.HeaderSizeBytes != HeaderSizeBytes {
		return fmt.Errorf("header_size_bytes is %d, expected %d", metadata.HeaderSizeBytes, HeaderSizeBytes)
	}
	if metadata.RecordSizeBytes != RecordSizeBytes {
		return fmt.Errorf("record_size_bytes is %d, expected %d", metadata.RecordSizeBytes, RecordSizeBytes)
	}
	if computedRecordSize != RecordSizeBytes {
		return fmt.Errorf("compiled record layout is %d bytes, expected %d", computedRecordSize, RecordSizeBytes)
	}
	if metadata.TopK != TopK {
		return fmt.Errorf("top_k is %d, expected %d", metadata.TopK, TopK)
	}
	if metadata.MaxTurns != MaxTurns {
		return fmt.Errorf("max_turns is %d, expected %d", metadata.MaxTurns, MaxTurns)
	}
	if metadata.GuessVocabSize != GuessVocabSize {
		return fmt.Errorf("guess_vocab_size is %d, expected %d", metadata.GuessVocabSize, GuessVocabSize)
	}
	if metadata.GlobalSolutionVocabSize != GlobalSolutionVocabSize {
		return fmt.Errorf("global_solution_vocab_size is %d, expected %d", metadata.GlobalSolutionVocabSize, GlobalSolutionVocabSize)
	}
	if metadata.RecordsPerSolution != RecordsPerSolution {
		return fmt.Errorf("records_per_solution is %d, expected %d", metadata.RecordsPerSolution, RecordsPerSolution)
	}
	if metadata.RecordsPerDepth != RecordsPerDepth {
		return fmt.Errorf("records_per_depth is %d, expected %d", metadata.RecordsPerDepth, RecordsPerDepth)
	}
	if metadata.PaddingFeedbackValue != PaddingFeedbackValue {
		return fmt.Errorf("padding_feedback_value is %d, expected %d", metadata.PaddingFeedbackValue, PaddingFeedbackValue)
	}
	if int(metadata.SolutionCount) != len(metadata.SolutionIDs) {
		return fmt.Errorf("solution_count is %d, but solution_ids has %d entries", metadata.SolutionCount, len(metadata.SolutionIDs))
	}
	expectedRecords := metadata.SolutionCount * metadata.RecordsPerSolution
	if metadata.IncludesOpeningState {
		expectedRecords++
	}
	if metadata.RecordCount != expectedRecords {
		return fmt.Errorf("record_count is %d, expected %d from solution_count and records_per_solution", metadata.RecordCount, expectedRecords)
	}
	return nil
}

func expectedSplitID(split SplitName) (uint32, bool) {
	switch split {
	case SplitTrain:
		return 1, true
	case SplitValidation:
		return 2, true
	case SplitTest:
		return 3, true
	case SplitMini:
		return 4, true
	default:
		return 0, false
	}
}

func binaryPathForMetadata(dir string, metadata Metadata) (string, error) {
	if filepath.Base(metadata.BinaryFile) != metadata.BinaryFile {
		return "", fmt.Errorf("binary_file must be a basename, got %q", metadata.BinaryFile)
	}
	return filepath.Join(dir, metadata.BinaryFile), nil
}

func loadBinary(path string, metadata Metadata) (Header, []Sample, error) {
	var header Header

	file, err := os.Open(path)
	if err != nil {
		return header, nil, fmt.Errorf("open binary %s: %w", path, err)
	}
	defer file.Close()

	stat, err := file.Stat()
	if err != nil {
		return header, nil, fmt.Errorf("stat binary %s: %w", path, err)
	}
	expectedSize := int64(HeaderSizeBytes) + int64(metadata.RecordCount)*int64(RecordSizeBytes)
	if stat.Size() != expectedSize {
		return header, nil, fmt.Errorf("binary file size for %s is %d bytes, expected %d", path, stat.Size(), expectedSize)
	}

	headerBytes := make([]byte, HeaderSizeBytes)
	if _, err := io.ReadFull(file, headerBytes); err != nil {
		return header, nil, fmt.Errorf("read binary header %s: %w", path, err)
	}

	header, err = parseHeader(headerBytes)
	if err != nil {
		return header, nil, fmt.Errorf("parse binary header %s: %w", path, err)
	}
	if err := validateHeader(header, metadata); err != nil {
		return header, nil, fmt.Errorf("validate binary header %s: %w", path, err)
	}

	samples := make([]Sample, 0, int(metadata.RecordCount))
	reader := bufio.NewReader(file)
	record := make([]byte, RecordSizeBytes)
	for recordIndex := uint32(0); recordIndex < header.RecordCount; recordIndex++ {
		if _, err := io.ReadFull(reader, record); err != nil {
			return header, nil, fmt.Errorf("read record %d from %s: %w", recordIndex, path, err)
		}
		sample, err := parseRecord(record)
		if err != nil {
			return header, nil, fmt.Errorf("parse record %d from %s: %w", recordIndex, path, err)
		}
		if err := validateSample(sample, recordIndex, metadata); err != nil {
			return header, nil, fmt.Errorf("validate record %d from %s: %w", recordIndex, path, err)
		}
		samples = append(samples, sample)
	}

	return header, samples, nil
}

func parseHeader(headerBytes []byte) (Header, error) {
	var header Header
	if len(headerBytes) != HeaderSizeBytes {
		return header, fmt.Errorf("header is %d bytes, expected %d", len(headerBytes), HeaderSizeBytes)
	}
	if string(headerBytes[0:4]) != binaryMagic {
		return header, fmt.Errorf("magic is %q, expected %q", string(headerBytes[0:4]), binaryMagic)
	}
	header = Header{
		Version:        binary.LittleEndian.Uint32(headerBytes[4:8]),
		RecordCount:    binary.LittleEndian.Uint32(headerBytes[8:12]),
		TopK:           binary.LittleEndian.Uint32(headerBytes[12:16]),
		MaxTurns:       binary.LittleEndian.Uint32(headerBytes[16:20]),
		GuessVocabSize: binary.LittleEndian.Uint32(headerBytes[20:24]),
		SolutionCount:  binary.LittleEndian.Uint32(headerBytes[24:28]),
		SplitID:        binary.LittleEndian.Uint32(headerBytes[28:32]),
	}
	for offset, b := range headerBytes[32:HeaderSizeBytes] {
		if b != 0 {
			return header, fmt.Errorf("reserved header byte %d is %d, expected 0", 32+offset, b)
		}
	}
	return header, nil
}

func validateHeader(header Header, metadata Metadata) error {
	if header.Version != FormatVersion {
		return fmt.Errorf("unsupported binary version %d, expected %d", header.Version, FormatVersion)
	}
	if header.Version != metadata.Version {
		return fmt.Errorf("version %d does not match metadata version %d", header.Version, metadata.Version)
	}
	if header.RecordCount != metadata.RecordCount {
		return fmt.Errorf("record_count %d does not match metadata record_count %d", header.RecordCount, metadata.RecordCount)
	}
	if header.TopK != metadata.TopK {
		return fmt.Errorf("top_k %d does not match metadata top_k %d", header.TopK, metadata.TopK)
	}
	if header.MaxTurns != metadata.MaxTurns {
		return fmt.Errorf("max_turns %d does not match metadata max_turns %d", header.MaxTurns, metadata.MaxTurns)
	}
	if header.GuessVocabSize != metadata.GuessVocabSize {
		return fmt.Errorf("guess_vocab_size %d does not match metadata guess_vocab_size %d", header.GuessVocabSize, metadata.GuessVocabSize)
	}
	if header.SolutionCount != metadata.SolutionCount {
		return fmt.Errorf("solution_count %d does not match metadata solution_count %d", header.SolutionCount, metadata.SolutionCount)
	}
	if header.SplitID != metadata.SplitID {
		return fmt.Errorf("split_id %d does not match metadata split_id %d", header.SplitID, metadata.SplitID)
	}
	return nil
}

func parseRecord(record []byte) (Sample, error) {
	var sample Sample
	if len(record) != RecordSizeBytes {
		return sample, fmt.Errorf("record is %d bytes, expected %d", len(record), RecordSizeBytes)
	}

	offset := 0
	var err error
	if sample.SolutionWord, err = parseWord(record[offset : offset+WordLength]); err != nil {
		return sample, fmt.Errorf("solution_word: %w", err)
	}
	offset += WordLength

	sample.TurnDepth = record[offset]
	offset++

	for i := range sample.PreviousGuessWords {
		if sample.PreviousGuessWords[i], err = parseWord(record[offset : offset+WordLength]); err != nil {
			return sample, fmt.Errorf("previous_guess_words[%d]: %w", i, err)
		}
		offset += WordLength
	}

	for turn := range sample.PreviousFeedback {
		for pos := range sample.PreviousFeedback[turn] {
			sample.PreviousFeedback[turn][pos] = Feedback(record[offset])
			offset++
		}
	}

	sample.ShortlistSizeBefore = binary.LittleEndian.Uint16(record[offset : offset+2])
	offset += 2

	for i := range sample.TopKGuessWords {
		if sample.TopKGuessWords[i], err = parseWord(record[offset : offset+WordLength]); err != nil {
			return sample, fmt.Errorf("top_k_guess_words[%d]: %w", i, err)
		}
		offset += WordLength
	}

	for i := range sample.TopKReductionRatios {
		sample.TopKReductionRatios[i] = math.Float32frombits(binary.LittleEndian.Uint32(record[offset : offset+4]))
		offset += 4
	}

	for i := range sample.TopKWorstCaseSizes {
		sample.TopKWorstCaseSizes[i] = binary.LittleEndian.Uint16(record[offset : offset+2])
		offset += 2
	}

	if offset != RecordSizeBytes {
		return sample, fmt.Errorf("parsed %d bytes, expected %d", offset, RecordSizeBytes)
	}
	return sample, nil
}

func parseWord(raw []byte) (Word, error) {
	var word Word
	if len(raw) != WordLength {
		return word, fmt.Errorf("word is %d bytes, expected %d", len(raw), WordLength)
	}
	copy(word[:], raw)

	if word.IsEmpty() {
		return word, nil
	}
	for _, b := range word {
		if b < 'A' || b > 'Z' {
			return word, fmt.Errorf("contains non-uppercase ASCII byte %d", b)
		}
	}
	return word, nil
}

func validateSample(sample Sample, recordIndex uint32, metadata Metadata) error {
	if sample.TurnDepth > MaxTurns {
		return fmt.Errorf("turn_depth is %d, expected 0..%d", sample.TurnDepth, MaxTurns)
	}
	if sample.TurnDepth == 0 {
		if !metadata.IncludesOpeningState || recordIndex != 0 {
			return errors.New("turn_depth 0 is only supported for the first opening-state record")
		}
		if !sample.SolutionWord.IsEmpty() {
			return errors.New("opening-state solution_word must be empty")
		}
	} else if sample.SolutionWord.IsEmpty() {
		return errors.New("non-opening solution_word must not be empty")
	}

	for turn := 0; turn < MaxTurns; turn++ {
		usedTurn := turn < int(sample.TurnDepth)
		if usedTurn {
			if sample.PreviousGuessWords[turn].IsEmpty() {
				return fmt.Errorf("previous_guess_words[%d] is empty for used turn", turn)
			}
		} else if !sample.PreviousGuessWords[turn].IsEmpty() {
			return fmt.Errorf("previous_guess_words[%d] is not empty for unused turn", turn)
		}

		for pos, feedback := range sample.PreviousFeedback[turn] {
			if usedTurn {
				if feedback != FeedbackGrey && feedback != FeedbackYellow && feedback != FeedbackGreen {
					return fmt.Errorf("previous_feedback[%d][%d] is %d for used turn", turn, pos, feedback)
				}
				continue
			}
			if feedback != PaddingFeedbackValue {
				return fmt.Errorf("previous_feedback[%d][%d] is %d for unused turn, expected %d", turn, pos, feedback, PaddingFeedbackValue)
			}
		}
	}

	if sample.ShortlistSizeBefore == 0 {
		return errors.New("shortlist_size_before must be greater than 0")
	}
	for i, word := range sample.TopKGuessWords {
		if word.IsEmpty() {
			return fmt.Errorf("top_k_guess_words[%d] must not be empty", i)
		}
	}
	for i, ratio := range sample.TopKReductionRatios {
		if math.IsNaN(float64(ratio)) {
			return fmt.Errorf("top_k_reduction_ratios[%d] is NaN", i)
		}
		if ratio < 0 || ratio > 1 {
			return fmt.Errorf("top_k_reduction_ratios[%d] is %g, expected 0..1", i, ratio)
		}
	}
	for i, worstCaseSize := range sample.TopKWorstCaseSizes {
		if worstCaseSize > sample.ShortlistSizeBefore {
			return fmt.Errorf("top_k_worst_case_sizes[%d] is %d, greater than shortlist_size_before %d", i, worstCaseSize, sample.ShortlistSizeBefore)
		}
	}
	return nil
}
