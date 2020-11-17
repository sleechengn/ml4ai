package com.ml4ai.deeplearning.nn.core.optimizers;

import com.ml4ai.deeplearning.nn.core.Tensor;
import com.ml4ai.deeplearning.nn.core.Variable;
import org.nd4j.linalg.factory.Nd4j;

import java.util.LinkedList;

public class Moment implements NNOptimizer {

    private double delta;
    private double gamma;
    private LinkedList<Parameter> params;

    private static class Parameter {

        Variable variable;      //the parameter to be optimized
        Tensor speed;           //the speed of the variable
        int type;

        public Parameter(Variable variable) {
            this.variable = variable;
            if (variable.data.type == Tensor.SCALAR_TYPE) {
                this.type = 0;
                this.speed = new Tensor(0D);
            } else if (variable.data.type > Tensor.SCALAR_TYPE) {
                this.type = 1;
                this.speed = new Tensor(Nd4j.zeros(this.variable.data.tensor.shape()));
            } else {
                throw new UnsupportedOperationException("不支持该参数");
            }
        }

    }

    /**
     * @param variables 要优化的参数
     * @param delta     学习速率
     * @param gamma     动量系数
     */
    public Moment(Variable[] variables, double delta, double gamma) {
        this.gamma = gamma;
        this.delta = delta;
        initialize(variables);
    }

    @Override
    public void update() {
        for (Parameter parameter : params) {
            switch (parameter.type) {
                case 0: {
                    parameter.speed.scalar = this.gamma * parameter.speed.scalar + parameter.variable.grad.scalar;
                    parameter.variable.data.scalar = parameter.variable.data.scalar - parameter.speed.scalar * delta;
                }
                break;
                case 1: {
                    parameter.speed.tensor = parameter.speed.tensor.mul(this.gamma).add(parameter.variable.grad.tensor);
                    parameter.variable.data.tensor.subi(parameter.speed.tensor.mul(delta));
                }
                break;
            }
        }
    }

    @Override
    public void initialize(Variable... parameters) {
        params = new LinkedList<>();
        for (Variable variable : parameters) {
            params.add(new Parameter(variable));
        }
    }

}


