package telemetry

import (
	"bytes"
	"encoding/binary"
	"hash/crc32"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestTensorBoardWriterWritesScalarEventFile(t *testing.T) {
	dir := t.TempDir()
	writer, err := NewTensorBoardWriter(dir)
	if err != nil {
		t.Fatalf("NewTensorBoardWriter() error = %v", err)
	}
	path := writer.Path()
	if filepath.Dir(path) != dir {
		t.Fatalf("writer path dir = %q, want %q", filepath.Dir(path), dir)
	}
	if err := writer.WriteScalar("validation_delta_from_start", 1443, -1.25); err != nil {
		t.Fatalf("WriteScalar() error = %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir() error = %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("event file count = %d, want 1", len(entries))
	}
	if !bytes.HasPrefix([]byte(entries[0].Name()), []byte(eventFilePrefix+".")) {
		t.Fatalf("event file name = %q, want prefix %q", entries[0].Name(), eventFilePrefix+".")
	}

	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	records, err := readTestRecords(content)
	if err != nil {
		t.Fatalf("readTestRecords() error = %v", err)
	}
	if len(records) != 2 {
		t.Fatalf("record count = %d, want 2", len(records))
	}
	if !bytes.Contains(records[0], []byte(fileVersion)) {
		t.Fatalf("first record does not contain file version %q", fileVersion)
	}
	if !bytes.Contains(records[1], []byte("validation_delta_from_start")) {
		t.Fatalf("scalar record does not contain tag")
	}

	var wantValue [4]byte
	binary.LittleEndian.PutUint32(wantValue[:], math.Float32bits(float32(-1.25)))
	if !bytes.Contains(records[1], wantValue[:]) {
		t.Fatalf("scalar record does not contain encoded scalar value")
	}
}

func TestTensorBoardWriterRejectsInvalidScalar(t *testing.T) {
	writer, err := NewTensorBoardWriter(t.TempDir())
	if err != nil {
		t.Fatalf("NewTensorBoardWriter() error = %v", err)
	}
	defer writer.Close()

	if err := writer.WriteScalar("", 0, 1); err == nil {
		t.Fatal("WriteScalar() with empty tag succeeded, want error")
	}
	if err := writer.WriteScalar("loss", -1, 1); err == nil {
		t.Fatal("WriteScalar() with negative step succeeded, want error")
	}
	if err := writer.WriteScalar("loss", 0, math.NaN()); err == nil {
		t.Fatal("WriteScalar() with NaN succeeded, want error")
	}
}

func readTestRecords(content []byte) ([][]byte, error) {
	var records [][]byte
	for len(content) > 0 {
		if len(content) < 12 {
			return nil, os.ErrInvalid
		}
		lengthBytes := content[:8]
		lengthCRC := binary.LittleEndian.Uint32(content[8:12])
		if got := crc32.Checksum(lengthBytes, crc32cTable); maskTestCRC32C(got) != lengthCRC {
			return nil, os.ErrInvalid
		}

		length := binary.LittleEndian.Uint64(lengthBytes)
		recordEnd := 12 + int(length)
		checksumEnd := recordEnd + 4
		if length > uint64(len(content)) || checksumEnd > len(content) {
			return nil, os.ErrInvalid
		}
		payload := content[12:recordEnd]
		payloadCRC := binary.LittleEndian.Uint32(content[recordEnd:checksumEnd])
		if got := crc32.Checksum(payload, crc32cTable); maskTestCRC32C(got) != payloadCRC {
			return nil, os.ErrInvalid
		}

		records = append(records, payload)
		content = content[checksumEnd:]
	}
	return records, nil
}

func maskTestCRC32C(crc uint32) uint32 {
	return ((crc >> 15) | (crc << 17)) + 0xa282ead8
}
