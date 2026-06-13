package telemetry

import (
	"math"
	"testing"
)

func TestReadScalarPointsReadsSelectedScalars(t *testing.T) {
	dir := t.TempDir()
	writer, err := NewTensorBoardWriter(dir)
	if err != nil {
		t.Fatalf("NewTensorBoardWriter() error = %v", err)
	}
	if err := writer.WriteScalar("train/mean_loss", 10, 2.5); err != nil {
		t.Fatalf("WriteScalar(train/mean_loss) error = %v", err)
	}
	if err := writer.WriteScalar("validation_delta_from_start", 10, -0.75); err != nil {
		t.Fatalf("WriteScalar(validation_delta_from_start) error = %v", err)
	}
	if err := writer.WriteScalar("learning_rate", 10, 0.01); err != nil {
		t.Fatalf("WriteScalar(learning_rate) error = %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	points, err := ReadScalarPoints(dir, []string{"train/mean_loss", "validation_delta_from_start"})
	if err != nil {
		t.Fatalf("ReadScalarPoints() error = %v", err)
	}
	if len(points) != 2 {
		t.Fatalf("ReadScalarPoints() returned %d points, want 2", len(points))
	}

	assertScalarPoint(t, points[0], "train/mean_loss", 10, 2.5)
	assertScalarPoint(t, points[1], "validation_delta_from_start", 10, -0.75)
}

func TestReadScalarPointsRejectsMissingEventFiles(t *testing.T) {
	_, err := ReadScalarPoints(t.TempDir(), []string{"train/mean_loss"})
	if err == nil {
		t.Fatal("ReadScalarPoints() error = nil, want missing event files error")
	}
}

func assertScalarPoint(t *testing.T, point ScalarPoint, tag string, step int64, value float64) {
	t.Helper()
	if point.Tag != tag {
		t.Fatalf("point.Tag = %q, want %q", point.Tag, tag)
	}
	if point.Step != step {
		t.Fatalf("point.Step = %d, want %d", point.Step, step)
	}
	if math.Abs(point.Value-value) > 1e-6 {
		t.Fatalf("point.Value = %g, want %g", point.Value, value)
	}
	if point.WallTime <= 0 {
		t.Fatalf("point.WallTime = %g, want positive", point.WallTime)
	}
	if point.EventFile == "" {
		t.Fatal("point.EventFile is empty")
	}
}
