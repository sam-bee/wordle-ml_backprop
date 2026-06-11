package inference

import (
	"reflect"
	"testing"
)

func TestTrunkHiddenDimsFromMetadata(t *testing.T) {
	content := []byte(`{
		"Variables": [
			{"ParameterName":"var:/policy_model/dense_trunk/input_to_hidden0/dense/weights","Dimensions":[321,256]},
			{"ParameterName":"var:/policy_model/dense_trunk/hidden0_to_hidden1/dense/weights","Dimensions":[256,256]},
			{"ParameterName":"var:/policy_model/dense_trunk/hidden1_to_hidden2/dense/weights","Dimensions":[256,256]},
			{"ParameterName":"var:/policy_model/dense_trunk/hidden2_to_hidden3/dense/weights","Dimensions":[256,128]},
			{"ParameterName":"var:/policy_model/dense_trunk/hidden3_to_hidden4/dense/weights","Dimensions":[128,128]},
			{"ParameterName":"var:/policy_model/dense_trunk/hidden4_to_hidden5/dense/weights","Dimensions":[128,128]},
			{"ParameterName":"var:/policy_model/dense_trunk/hidden5_to_output/dense/weights","Dimensions":[128,64]}
		]
	}`)

	got, err := TrunkHiddenDimsFromMetadata("checkpoint.json", content)
	if err != nil {
		t.Fatalf("TrunkHiddenDimsFromMetadata() error = %v", err)
	}
	want := []int{256, 256, 256, 128, 128, 128}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("hidden dims = %v, want %v", got, want)
	}
}

func TestTrunkHiddenDimsFromMetadataRejectsManifest(t *testing.T) {
	_, err := TrunkHiddenDimsFromMetadata("manifest.json", []byte(`{"latest_gomlx_checkpoint":"checkpoint"}`))
	if err == nil {
		t.Fatal("TrunkHiddenDimsFromMetadata() succeeded for manifest, want error")
	}
}
