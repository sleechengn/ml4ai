package com.ml4ai.deeplearning.nn.core.optimizers;

import com.ml4ai.deeplearning.nn.core.Tensor;
import com.ml4ai.deeplearning.nn.core.Variable;

import java.util.Arrays;
import java.util.LinkedList;

public class GradientDescent implements NNOptimizer {

    private double delta;
    private LinkedList<Variable> params;

    /**
     * 优化的参数
     *
     * @param vars  //优化参数
     * @param delta //学习速率
     */
    public GradientDescent(Variable[] vars, double delta) {
        initialize(vars);
        GradientDescent optimizer = this;
        optimizer.delta = delta;
    }

    @Override
    public void update() {
        for (Variable variable : params) {
            if (variable.data.rank == Tensor.SCALAR_RANK) {
                variable.data.scalar -= delta * variable.grad.scalar;
            } else if (variable.data.rank > Tensor.SCALAR_RANK) {
                variable.data.tensor.subi(variable.grad.tensor.mul(delta));
            }
        }
    }

    @Override
    public void initialize(Variable... parameters) {
        params = new LinkedList<>();
        params.addAll(Arrays.asList(parameters));
    }

}