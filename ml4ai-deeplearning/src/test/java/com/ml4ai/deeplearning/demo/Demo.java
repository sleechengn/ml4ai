package com.ml4ai.deeplearning.demo;

import com.ml4ai.deeplearning.nn.BaseForwardNetwork;
import com.ml4ai.deeplearning.nn.Linear;
import com.ml4ai.deeplearning.nn.core.Toolkit;
import com.ml4ai.deeplearning.nn.core.Variable;
import com.ml4ai.deeplearning.nn.core.optimizers.Moment;
import com.ml4ai.deeplearning.nn.core.optimizers.NNOptimizer;
import org.nd4j.linalg.factory.Nd4j;

public class Demo {
    public static class KxPlusB extends BaseForwardNetwork {

        KxPlusB() {
            add(new Linear(1, 1));
        }

        @Override
        public Variable[] getParameters() {
            return super.getParameters();
        }

        @Override
        public Variable[] forward(Variable... inputs) {
            return super.forward(inputs);
        }
    }

    public static void main(String[] args) {

        Variable x = new Variable(Nd4j.randn(100, 1));  //产生随机数
        Variable y = x.mulScalar(new Variable(3)).add(new Variable(Nd4j.zeros(100, 1).add(2))); //乘3 + 2

        KxPlusB kxPlusB = new KxPlusB();
        Toolkit t = new Toolkit();
        NNOptimizer optimizer = new Moment(kxPlusB.getParameters(), 0.01, 0.99);
        for (int i = 0; i < 10000; i++) {
            Variable predict = kxPlusB.forward(x)[0];
            Variable loss = predict.sub(y).square().mean();
            t.grad2zero(loss);
            t.backward(loss);
            optimizer.update();
            if (i % 1000 == 0) {
                System.out.println(0);
                System.out.println(kxPlusB.forward(new Variable(Nd4j.create(new double[]{0.1})))[0].data);  //预测 0.1 输出
                System.out.println("k:" + kxPlusB.getParameters()[0].data.tensor.data());       //学习到的k
                System.out.println("p:" + kxPlusB.getParameters()[1].data.tensor.data());       //学习到的b
            }
        }

    }

}


