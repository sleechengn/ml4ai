package com.ml4ai.deeplearning.nn;

import com.ml4ai.deeplearning.nn.core.Variable;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class LeakyRelu implements ForwardNetwork {

    @Override
    public Variable[] getParameters() {
        return new Variable[]{};
    }

    @Override
    public Variable[] forward(Variable... inputs) {
        List<Variable> variables = Arrays.asList(inputs);
        List<Variable> result = variables.stream().map(Variable::leaky_relu).collect(Collectors.toList());
        return result.toArray(new Variable[0]);
    }
}
