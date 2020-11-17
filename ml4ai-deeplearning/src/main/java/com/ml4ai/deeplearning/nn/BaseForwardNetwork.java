package com.ml4ai.deeplearning.nn;

import com.ml4ai.deeplearning.nn.core.Variable;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public abstract class BaseForwardNetwork implements ForwardNetwork {

    private List<ForwardNetwork> sequential = new ArrayList<>();

    protected BaseForwardNetwork(ForwardNetwork... forwardNetworks) {
        for (ForwardNetwork forwardNetwork : forwardNetworks)
            add(forwardNetwork);
    }

    public void add(ForwardNetwork forwardNetwork) {
        sequential.add(forwardNetwork);
    }

    @Override
    public Variable[] getParameters() {
        List<Variable> variables = new ArrayList<>();
        for (int i = 0; i < sequential.size(); i++) {
            ForwardNetwork forwardNetwork = sequential.get(i);
            variables.addAll(Arrays.asList(forwardNetwork.getParameters()));
        }
        return variables.toArray(new Variable[0]);
    }

    @Override
    public Variable[] forward(Variable... inputs) {
        Variable[] actived = inputs;
        for (int i = 0; i < sequential.size(); i++) {
            ForwardNetwork forwardNetwork = sequential.get(i);
            actived = forwardNetwork.forward(actived);
        }
        return actived;
    }
}
