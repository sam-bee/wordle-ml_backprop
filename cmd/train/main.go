package main

import (
	"fmt"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	gomlxcontext "github.com/gomlx/gomlx/pkg/ml/context"
)

func main() {
	ctx := gomlxcontext.New()
	inputShape := shapes.Make(dtypes.Float32, 1)

	fmt.Println("wordle backprop trainer starting")
	fmt.Printf("gomlx ready: scope=%q input_shape=%s\n", ctx.Scope(), inputShape)
}
