package com.ml4ai.deeplearning.nn;

import com.ml4ai.deeplearning.nn.core.Variable;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

@Slf4j
public class Linear implements ForwardNetwork {

    private Variable weights;
    private Variable bias;

    public Linear(int input, int output) {
        INDArray weightsArray = Nd4jUtil.randn(new int[]{input, output});
        log.info("{}", weightsArray.data().asDouble());
        weights = new Variable(weightsArray);
        bias = new Variable(Nd4j.zeros(new int[]{output}));
    }

    @Override
    public Variable[] getParameters() {
        return new Variable[]{weights, bias};
    }

    @Override
    public Variable[] forward(Variable... inputs) {
        return new Variable[]{inputs[0].matMul(weights).addVec(bias)};
    }
}
