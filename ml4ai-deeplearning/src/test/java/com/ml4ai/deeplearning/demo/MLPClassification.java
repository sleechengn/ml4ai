package com.ml4ai.deeplearning.demo;

import com.ml4ai.deeplearning.nn.BaseForwardNetwork;
import com.ml4ai.deeplearning.nn.Linear;
import com.ml4ai.deeplearning.nn.SequentialForward;
import com.ml4ai.deeplearning.nn.Sigmoid;
import com.ml4ai.deeplearning.nn.core.Toolkit;
import com.ml4ai.deeplearning.nn.core.Variable;
import com.ml4ai.deeplearning.nn.core.optimizers.Moment;
import com.ml4ai.deeplearning.nn.core.optimizers.NNOptimizer;
import org.nd4j.linalg.factory.Nd4j;

public class MLPClassification {

    public static void main(String[] args) {
        int epoch = 10000;
        Variable train = new Variable(Nd4j.create(new double[][]{
                {0.1, 0.2, 0.3, 0.4, 0.5},
                {0.2, 0.3, 0.4, 0.5, 0.6}
        }));
        Variable label = new Variable(Nd4j.create(new double[][]{
                {0, 0, 1},
                {0, 1, 0}
        }));
        MLPNetwork ml = new MLPNetwork();
        //NNOptimizer optim = new GradientDescent(ml.getParameters(),0.01); //梯度下降
        NNOptimizer optim = new Moment(ml.getParameters(), 0.01, 0.95);
        Toolkit tool = new Toolkit();
        for (int i = 0; i < epoch; i++) {
            Variable loss = ml.forward(train)[0].sub(label).square().mean();
            tool.grad2zero(loss);
            tool.backward(loss);
            optim.update();
            if (i % 100 == 0) {
                System.out.println(loss.data.scalar);
            }
        }
        System.out.println(ml.forward(new Variable(Nd4j.create(new double[]{
                0.11, 0.21, 0.31, 0.41, 0.51
        })))[0].data.tensor.data());
    }

    public static class MLPNetwork extends BaseForwardNetwork {

        MLPNetwork() {
            SequentialForward sequentialForward = new SequentialForward(
                    new Linear(5, 4),
                    new Sigmoid(),
                    new Linear(4, 3),
                    new Sigmoid()
            );
            add(sequentialForward);
        }

        @Override
        public Variable[] forward(Variable... inputs) {
            return super.forward(inputs);
        }
    }

}


