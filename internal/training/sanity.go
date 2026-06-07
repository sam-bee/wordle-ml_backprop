package training

import (
	"fmt"

	"github.com/gomlx/gomlx/backends/simplego"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	gomlxtrain "github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/ml/train/losses"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"

	"github.com/sam-bee/wordle-ml_backprop/internal/data"
	"github.com/sam-bee/wordle-ml_backprop/internal/model"
)

const sanityLearningRate = 0.05

type SanityResult struct {
	InitialLoss     float64
	TrainingLoss    float64
	PostUpdateLoss  float64
	UpdateCompleted bool
}

func RunSanityStep(batch data.Batch) (SanityResult, error) {
	var result SanityResult
	if batch.Size() == 0 {
		return result, fmt.Errorf("batch is empty")
	}

	backend, err := simplego.New("")
	if err != nil {
		return result, fmt.Errorf("create SimpleGo backend: %w", err)
	}
	defer backend.Finalize()

	ctx := context.New()
	if err := ctx.SetRNGStateFromSeed(1); err != nil {
		return result, fmt.Errorf("seed GoMLX context: %w", err)
	}

	optimizer := optimizers.StochasticGradientDescent().
		WithLearningRate(sanityLearningRate).
		WithDecay(false).
		Done()
	trainer := gomlxtrain.NewTrainer(
		backend,
		ctx,
		model.SanityModel,
		losses.MeanSquaredError,
		optimizer,
		nil,
		nil,
	)

	initialLoss, err := evalLoss(trainer, batch)
	if err != nil {
		return result, fmt.Errorf("initial eval: %w", err)
	}
	trainingLoss, err := trainLoss(trainer, batch)
	if err != nil {
		return result, fmt.Errorf("train step: %w", err)
	}
	postUpdateLoss, err := evalLoss(trainer, batch)
	if err != nil {
		return result, fmt.Errorf("post-update eval: %w", err)
	}

	result.InitialLoss = initialLoss
	result.TrainingLoss = trainingLoss
	result.PostUpdateLoss = postUpdateLoss
	result.UpdateCompleted = true
	return result, nil
}

func BatchToTensors(batch data.Batch) (inputs, labels []*tensors.Tensor, err error) {
	if batch.Size() == 0 {
		return nil, nil, fmt.Errorf("batch is empty")
	}
	if len(batch.Inputs) != len(batch.Targets) {
		return nil, nil, fmt.Errorf("batch has %d inputs and %d targets", len(batch.Inputs), len(batch.Targets))
	}

	batchSize := batch.Size()
	features := make([]float32, 0, batchSize*model.InputFeatureCount)
	targets := make([]float32, 0, batchSize*model.OutputCount)
	for i := range batch.Inputs {
		appendInputFeatures(&features, batch.Inputs[i])
		appendTargets(&targets, batch.Targets[i])
	}

	if len(features) != batchSize*model.InputFeatureCount {
		return nil, nil, fmt.Errorf("encoded %d input features, expected %d", len(features), batchSize*model.InputFeatureCount)
	}
	if len(targets) != batchSize*model.OutputCount {
		return nil, nil, fmt.Errorf("encoded %d target values, expected %d", len(targets), batchSize*model.OutputCount)
	}

	input := tensors.FromFlatDataAndDimensions(features, batchSize, model.InputFeatureCount)
	label := tensors.FromFlatDataAndDimensions(targets, batchSize, model.OutputCount)
	return []*tensors.Tensor{input}, []*tensors.Tensor{label}, nil
}

func appendInputFeatures(features *[]float32, input data.BatchInput) {
	*features = append(*features, float32(input.TurnDepth)/float32(data.MaxTurns))
	*features = append(*features, float32(input.ShortlistSizeBefore)/float32(data.GlobalSolutionVocabSize))

	for _, word := range input.PreviousGuessWords {
		for _, b := range word {
			*features = append(*features, encodeWordByte(b))
		}
	}
	for _, turn := range input.PreviousFeedback {
		for _, feedback := range turn {
			*features = append(*features, encodeFeedback(feedback))
		}
	}
}

func appendTargets(targets *[]float32, target data.BatchTarget) {
	*targets = append(*targets, target.TopKReductionRatios[:]...)
}

func encodeWordByte(b byte) float32 {
	if b == 0 {
		return 0
	}
	return float32(b-'A'+1) / 26
}

func encodeFeedback(feedback data.Feedback) float32 {
	switch feedback {
	case data.FeedbackGreen:
		return 1
	case data.FeedbackYellow:
		return 0.5
	default:
		return 0
	}
}

func evalLoss(trainer *gomlxtrain.Trainer, batch data.Batch) (float64, error) {
	inputs, labels, err := BatchToTensors(batch)
	if err != nil {
		return 0, err
	}
	defer finalizeTensors(inputs)
	defer finalizeTensors(labels)

	metrics, err := trainer.EvalStep(nil, inputs, labels)
	if err != nil {
		return 0, err
	}
	defer finalizeTensors(metrics)

	return scalarTensor(metrics[0])
}

func trainLoss(trainer *gomlxtrain.Trainer, batch data.Batch) (float64, error) {
	inputs, labels, err := BatchToTensors(batch)
	if err != nil {
		return 0, err
	}
	defer finalizeTensors(inputs)
	defer finalizeTensors(labels)

	metrics, err := trainer.TrainStep(nil, inputs, labels)
	if err != nil {
		return 0, err
	}
	defer finalizeTensors(metrics)

	return scalarTensor(metrics[0])
}

func scalarTensor(tensor *tensors.Tensor) (float64, error) {
	switch tensor.DType() {
	case dtypes.Float32:
		return float64(tensors.ToScalar[float32](tensor)), nil
	case dtypes.Float64:
		return tensors.ToScalar[float64](tensor), nil
	default:
		return 0, fmt.Errorf("scalar tensor has unsupported dtype %s", tensor.DType())
	}
}

func finalizeTensors(tensorsToFinalize []*tensors.Tensor) {
	for _, tensor := range tensorsToFinalize {
		if tensor != nil {
			tensor.MustFinalizeAll()
		}
	}
}
