package com.ml4ai.deeplearning.nn.core.optimizers;

import com.ml4ai.deeplearning.nn.core.Variable;

public interface NNOptimizer {

    void update();

    void initialize(Variable... parameters);

}
