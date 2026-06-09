package telemetry

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	eventFilePrefix = "events.out.tfevents"
	fileVersion     = "brain.Event:2"
)

var crc32cTable = crc32.MakeTable(crc32.Castagnoli)

// TensorBoardWriter writes the minimal TensorBoard event-file format needed for
// scalar training telemetry.
type TensorBoardWriter struct {
	file *os.File
	path string
}

func NewTensorBoardWriter(dir string) (*TensorBoardWriter, error) {
	if dir == "" {
		return nil, fmt.Errorf("tensorboard directory is required")
	}
	if err := os.MkdirAll(dir, 0o770); err != nil {
		return nil, fmt.Errorf("create tensorboard directory %s: %w", dir, err)
	}

	now := time.Now()
	file, path, err := createEventFile(dir, now)
	if err != nil {
		return nil, err
	}
	writer := &TensorBoardWriter{
		file: file,
		path: path,
	}
	if err := writer.writeRecord(marshalFileVersionEvent(now, fileVersion)); err != nil {
		_ = writer.Close()
		return nil, fmt.Errorf("write tensorboard file version: %w", err)
	}
	return writer, nil
}

func (writer *TensorBoardWriter) Path() string {
	if writer == nil {
		return ""
	}
	return writer.path
}

func (writer *TensorBoardWriter) WriteScalar(tag string, step int64, value float64) error {
	if writer == nil {
		return fmt.Errorf("tensorboard writer is nil")
	}
	if tag == "" {
		return fmt.Errorf("tensorboard scalar tag is required")
	}
	if strings.ContainsAny(tag, "\x00\r\n") {
		return fmt.Errorf("tensorboard scalar tag %q contains invalid control characters", tag)
	}
	if step < 0 {
		return fmt.Errorf("tensorboard scalar step must be >= 0, got %d", step)
	}
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return fmt.Errorf("tensorboard scalar %s is not finite: %g", tag, value)
	}
	return writer.writeRecord(marshalScalarEvent(time.Now(), tag, step, float32(value)))
}

func (writer *TensorBoardWriter) Close() error {
	if writer == nil || writer.file == nil {
		return nil
	}
	err := writer.file.Close()
	writer.file = nil
	return err
}

func createEventFile(dir string, now time.Time) (*os.File, string, error) {
	host, err := os.Hostname()
	if err != nil || host == "" {
		host = "unknown-host"
	}
	host = sanitizeEventFileComponent(host)
	pid := os.Getpid()
	unixSeconds := now.Unix()

	for suffix := 0; suffix < 1000; suffix++ {
		name := fmt.Sprintf("%s.%d.%s.%d.%d", eventFilePrefix, unixSeconds, host, pid, suffix)
		path := filepath.Join(dir, name)
		file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o660)
		if err == nil {
			return file, path, nil
		}
		if !os.IsExist(err) {
			return nil, "", fmt.Errorf("create tensorboard event file %s: %w", path, err)
		}
	}
	return nil, "", fmt.Errorf("create tensorboard event file in %s: too many filename collisions", dir)
}

func sanitizeEventFileComponent(value string) string {
	return strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z':
			return r
		case r >= 'A' && r <= 'Z':
			return r
		case r >= '0' && r <= '9':
			return r
		case r == '.', r == '_', r == '-':
			return r
		default:
			return '_'
		}
	}, value)
}

func (writer *TensorBoardWriter) writeRecord(payload []byte) error {
	if writer.file == nil {
		return fmt.Errorf("tensorboard event file is closed")
	}

	var length [8]byte
	binary.LittleEndian.PutUint64(length[:], uint64(len(payload)))
	var checksum [4]byte

	if _, err := writer.file.Write(length[:]); err != nil {
		return err
	}
	binary.LittleEndian.PutUint32(checksum[:], maskedCRC32C(length[:]))
	if _, err := writer.file.Write(checksum[:]); err != nil {
		return err
	}
	if _, err := writer.file.Write(payload); err != nil {
		return err
	}
	binary.LittleEndian.PutUint32(checksum[:], maskedCRC32C(payload))
	if _, err := writer.file.Write(checksum[:]); err != nil {
		return err
	}
	return nil
}

func maskedCRC32C(data []byte) uint32 {
	crc := crc32.Checksum(data, crc32cTable)
	return ((crc >> 15) | (crc << 17)) + 0xa282ead8
}

func marshalFileVersionEvent(at time.Time, version string) []byte {
	var event []byte
	event = appendDoubleField(event, 1, wallTime(at))
	event = appendStringField(event, 3, version)
	return event
}

func marshalScalarEvent(at time.Time, tag string, step int64, value float32) []byte {
	var scalarValue []byte
	scalarValue = appendStringField(scalarValue, 1, tag)
	scalarValue = appendFloatField(scalarValue, 2, value)

	var summary []byte
	summary = appendMessageField(summary, 1, scalarValue)

	var event []byte
	event = appendDoubleField(event, 1, wallTime(at))
	event = appendVarintField(event, 2, uint64(step))
	event = appendMessageField(event, 5, summary)
	return event
}

func wallTime(at time.Time) float64 {
	return float64(at.UnixNano()) / float64(time.Second)
}

func appendVarintField(dst []byte, fieldNumber int, value uint64) []byte {
	dst = appendVarint(dst, uint64(fieldNumber<<3))
	return appendVarint(dst, value)
}

func appendDoubleField(dst []byte, fieldNumber int, value float64) []byte {
	dst = appendVarint(dst, uint64(fieldNumber<<3|1))
	var raw [8]byte
	binary.LittleEndian.PutUint64(raw[:], math.Float64bits(value))
	return append(dst, raw[:]...)
}

func appendStringField(dst []byte, fieldNumber int, value string) []byte {
	return appendBytesField(dst, fieldNumber, []byte(value))
}

func appendMessageField(dst []byte, fieldNumber int, value []byte) []byte {
	return appendBytesField(dst, fieldNumber, value)
}

func appendBytesField(dst []byte, fieldNumber int, value []byte) []byte {
	dst = appendVarint(dst, uint64(fieldNumber<<3|2))
	dst = appendVarint(dst, uint64(len(value)))
	return append(dst, value...)
}

func appendFloatField(dst []byte, fieldNumber int, value float32) []byte {
	dst = appendVarint(dst, uint64(fieldNumber<<3|5))
	var raw [4]byte
	binary.LittleEndian.PutUint32(raw[:], math.Float32bits(value))
	return append(dst, raw[:]...)
}

func appendVarint(dst []byte, value uint64) []byte {
	for value >= 0x80 {
		dst = append(dst, byte(value)|0x80)
		value >>= 7
	}
	return append(dst, byte(value))
}
