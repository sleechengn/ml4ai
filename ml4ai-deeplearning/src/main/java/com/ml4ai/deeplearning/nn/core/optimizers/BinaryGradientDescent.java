package com.ml4ai.deeplearning.nn.core.optimizers;

import com.ml4ai.deeplearning.nn.core.Tensor;
import com.ml4ai.deeplearning.nn.core.Variable;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.LinkedList;

public class BinaryGradientDescent implements NNOptimizer {

    private double delta;
    private LinkedList<Variable> params;

    /**
     * 优化的参数
     *
     * @param vars  //优化参数
     * @param delta //学习速率
     */
    public BinaryGradientDescent(Variable[] vars, double delta) {
        initialize(vars);
        BinaryGradientDescent optimizer = this;
        optimizer.delta = delta;
    }

    @Override
    public void update() {
        for (Variable variable : params) {
            if (variable.data.rank == Tensor.SCALAR_RANK) {
                variable.data.scalar -= delta * (variable.grad.scalar > 0 ? 1 : variable.grad.scalar < 0 ? -1 : 0);
            } else if (variable.data.rank > Tensor.SCALAR_RANK) {
                INDArray gt0gd = variable.grad.tensor.gt(0);
                INDArray lt0gd = variable.grad.tensor.lt(0);
                variable.data.tensor.subi(gt0gd.mul(delta));
                variable.data.tensor.subi(lt0gd.mul(delta).mul(-1));
            }
        }
    }

    @Override
    public void initialize(Variable... parameters) {
        params = new LinkedList<>();
        params.addAll(Arrays.asList(parameters));
    }

}