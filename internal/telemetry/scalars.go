package telemetry

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type ScalarPoint struct {
	Tag       string
	Step      int64
	WallTime  float64
	Value     float64
	EventFile string
}

func ReadScalarPoints(dir string, tags []string) ([]ScalarPoint, error) {
	if dir == "" {
		return nil, fmt.Errorf("tensorboard directory is required")
	}

	wantedTags := make(map[string]bool, len(tags))
	for _, tag := range tags {
		wantedTags[tag] = true
	}

	paths, err := findEventFiles(dir)
	if err != nil {
		return nil, err
	}
	if len(paths) == 0 {
		return nil, fmt.Errorf("no TensorBoard event files found in %s", dir)
	}

	var points []ScalarPoint
	for _, path := range paths {
		if err := readEventFile(path, func(payload []byte) error {
			recordPoints, err := parseScalarPoints(payload, wantedTags)
			if err != nil {
				return err
			}
			for i := range recordPoints {
				recordPoints[i].EventFile = path
			}
			points = append(points, recordPoints...)
			return nil
		}); err != nil {
			return nil, fmt.Errorf("read %s: %w", path, err)
		}
	}
	return points, nil
}

func findEventFiles(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("read tensorboard directory %s: %w", dir, err)
	}

	var paths []string
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !strings.HasPrefix(name, eventFilePrefix+".") {
			continue
		}
		paths = append(paths, filepath.Join(dir, name))
	}
	sort.Strings(paths)
	return paths, nil
}

func readEventFile(path string, visit func(payload []byte) error) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	for {
		var lengthBytes [8]byte
		if _, err := io.ReadFull(file, lengthBytes[:]); err != nil {
			if err == io.EOF {
				return nil
			}
			return fmt.Errorf("read record length: %w", err)
		}

		var checksumBytes [4]byte
		if _, err := io.ReadFull(file, checksumBytes[:]); err != nil {
			return fmt.Errorf("read length checksum: %w", err)
		}
		if got, want := maskedCRC32C(lengthBytes[:]), binary.LittleEndian.Uint32(checksumBytes[:]); got != want {
			return fmt.Errorf("length checksum is %08x, expected %08x", got, want)
		}

		length := binary.LittleEndian.Uint64(lengthBytes[:])
		if length > uint64(int(^uint(0)>>1)) {
			return fmt.Errorf("record length %d is too large", length)
		}
		payload := make([]byte, int(length))
		if _, err := io.ReadFull(file, payload); err != nil {
			return fmt.Errorf("read record payload: %w", err)
		}
		if _, err := io.ReadFull(file, checksumBytes[:]); err != nil {
			return fmt.Errorf("read payload checksum: %w", err)
		}
		if got, want := maskedCRC32C(payload), binary.LittleEndian.Uint32(checksumBytes[:]); got != want {
			return fmt.Errorf("payload checksum is %08x, expected %08x", got, want)
		}

		if err := visit(payload); err != nil {
			return err
		}
	}
}

type parsedEvent struct {
	wallTime float64
	step     int64
	summary  []byte
}

type parsedSummaryValue struct {
	tag      string
	value    float64
	hasValue bool
}

func parseScalarPoints(payload []byte, wantedTags map[string]bool) ([]ScalarPoint, error) {
	event, err := parseEvent(payload)
	if err != nil {
		return nil, err
	}
	if len(event.summary) == 0 {
		return nil, nil
	}

	values, err := parseSummary(event.summary)
	if err != nil {
		return nil, err
	}

	points := make([]ScalarPoint, 0, len(values))
	for _, value := range values {
		if !value.hasValue || value.tag == "" {
			continue
		}
		if len(wantedTags) > 0 && !wantedTags[value.tag] {
			continue
		}
		points = append(points, ScalarPoint{
			Tag:      value.tag,
			Step:     event.step,
			WallTime: event.wallTime,
			Value:    value.value,
		})
	}
	return points, nil
}

func parseEvent(payload []byte) (parsedEvent, error) {
	var event parsedEvent
	for len(payload) > 0 {
		fieldNumber, wireType, rest, err := readProtoKey(payload)
		if err != nil {
			return event, err
		}
		payload = rest

		switch {
		case fieldNumber == 1 && wireType == 1:
			if len(payload) < 8 {
				return event, io.ErrUnexpectedEOF
			}
			event.wallTime = math.Float64frombits(binary.LittleEndian.Uint64(payload[:8]))
			payload = payload[8:]
		case fieldNumber == 2 && wireType == 0:
			step, rest, err := readProtoVarint(payload)
			if err != nil {
				return event, err
			}
			if step > math.MaxInt64 {
				return event, fmt.Errorf("event step %d overflows int64", step)
			}
			event.step = int64(step)
			payload = rest
		case fieldNumber == 5 && wireType == 2:
			value, rest, err := readProtoBytes(payload)
			if err != nil {
				return event, err
			}
			event.summary = value
			payload = rest
		default:
			var err error
			payload, err = skipProtoField(payload, wireType)
			if err != nil {
				return event, err
			}
		}
	}
	return event, nil
}

func parseSummary(payload []byte) ([]parsedSummaryValue, error) {
	var values []parsedSummaryValue
	for len(payload) > 0 {
		fieldNumber, wireType, rest, err := readProtoKey(payload)
		if err != nil {
			return nil, err
		}
		payload = rest

		if fieldNumber == 1 && wireType == 2 {
			rawValue, rest, err := readProtoBytes(payload)
			if err != nil {
				return nil, err
			}
			value, err := parseSummaryValue(rawValue)
			if err != nil {
				return nil, err
			}
			values = append(values, value)
			payload = rest
			continue
		}

		payload, err = skipProtoField(payload, wireType)
		if err != nil {
			return nil, err
		}
	}
	return values, nil
}

func parseSummaryValue(payload []byte) (parsedSummaryValue, error) {
	var value parsedSummaryValue
	for len(payload) > 0 {
		fieldNumber, wireType, rest, err := readProtoKey(payload)
		if err != nil {
			return value, err
		}
		payload = rest

		switch {
		case fieldNumber == 1 && wireType == 2:
			rawTag, rest, err := readProtoBytes(payload)
			if err != nil {
				return value, err
			}
			value.tag = string(rawTag)
			payload = rest
		case fieldNumber == 2 && wireType == 5:
			if len(payload) < 4 {
				return value, io.ErrUnexpectedEOF
			}
			value.value = float64(math.Float32frombits(binary.LittleEndian.Uint32(payload[:4])))
			value.hasValue = true
			payload = payload[4:]
		default:
			var err error
			payload, err = skipProtoField(payload, wireType)
			if err != nil {
				return value, err
			}
		}
	}
	return value, nil
}

func readProtoKey(payload []byte) (fieldNumber int, wireType uint64, rest []byte, err error) {
	key, rest, err := readProtoVarint(payload)
	if err != nil {
		return 0, 0, nil, err
	}
	fieldNumber = int(key >> 3)
	wireType = key & 0x7
	if fieldNumber <= 0 {
		return 0, 0, nil, fmt.Errorf("invalid protobuf field number %d", fieldNumber)
	}
	return fieldNumber, wireType, rest, nil
}

func readProtoBytes(payload []byte) ([]byte, []byte, error) {
	length, rest, err := readProtoVarint(payload)
	if err != nil {
		return nil, nil, err
	}
	if length > uint64(len(rest)) {
		return nil, nil, io.ErrUnexpectedEOF
	}
	return rest[:length], rest[length:], nil
}

func readProtoVarint(payload []byte) (uint64, []byte, error) {
	var value uint64
	for shift := 0; shift < 64; shift += 7 {
		if len(payload) == 0 {
			return 0, nil, io.ErrUnexpectedEOF
		}
		b := payload[0]
		payload = payload[1:]
		value |= uint64(b&0x7f) << shift
		if b < 0x80 {
			return value, payload, nil
		}
	}
	return 0, nil, fmt.Errorf("protobuf varint overflows uint64")
}

func skipProtoField(payload []byte, wireType uint64) ([]byte, error) {
	switch wireType {
	case 0:
		_, rest, err := readProtoVarint(payload)
		return rest, err
	case 1:
		if len(payload) < 8 {
			return nil, io.ErrUnexpectedEOF
		}
		return payload[8:], nil
	case 2:
		_, rest, err := readProtoBytes(payload)
		return rest, err
	case 5:
		if len(payload) < 4 {
			return nil, io.ErrUnexpectedEOF
		}
		return payload[4:], nil
	default:
		return nil, fmt.Errorf("unsupported protobuf wire type %d", wireType)
	}
}
