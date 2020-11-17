package com.ml4ai.deeplearning.demo;

import com.ml4ai.deeplearning.nn.*;
import com.ml4ai.deeplearning.nn.core.Toolkit;
import com.ml4ai.deeplearning.nn.core.Variable;
import com.ml4ai.deeplearning.nn.core.optimizers.Moment;
import com.ml4ai.deeplearning.nn.core.optimizers.NNOptimizer;
import org.nd4j.linalg.factory.Nd4j;

public class MomentTest {

    public static class XAutoEncoder extends BaseForwardNetwork {

        private ForwardNetwork encoder;
        private ForwardNetwork decoder;

        XAutoEncoder() {

            /**
             * 自动编码器的编码部分
             * 由线性单元和激活函数构成
             */
            encoder = new SequentialForward(
                    new Linear(128, 64),
                    new Tanh(),
                    new Linear(64, 32),
                    new Tanh(),
                    new Linear(32, 16),
                    new Tanh(),
                    new Linear(16, 8),
                    new Tanh(),
                    new Linear(8, 4),
                    new Tanh(),
                    new Linear(4, 2),
                    new Tanh()
            );

            /**
             * 自动编码器的解码部分
             */
            decoder = new SequentialForward(
                    new Linear(2, 4),
                    new Tanh(),
                    new Linear(4, 8),
                    new Tanh(),
                    new Linear(8, 16),
                    new Tanh(),
                    new Linear(16, 32),
                    new Tanh(),
                    new Linear(32, 64),
                    new Tanh(),
                    new Linear(64, 128)
            );

            //添加到基类中，以便于自动获取参数
            add(encoder);
            add(decoder);
        }

        @Override
        public Variable[] forward(Variable... inputs) {
            return super.forward(inputs);
        }

    }

    public static void main(String[] a) {
        /**
         * 初始化一个自动编码器
         */
        XAutoEncoder aex = new XAutoEncoder();
        /**
         * 随机生成数据集
         */
        Variable train = new Variable(Nd4j.rand(new int[]{256, 128}));
        int epoch = 100000;     //迭代次数
        double learn = 1e-3;    //学习速率
        Toolkit toolkit = new Toolkit();        //工具箱
        NNOptimizer optimizer = new Moment(aex.getParameters(), learn, 0.95);   //对aex的参数创建一个优化器
        for (int i = 0; i < epoch; i++) {
            Variable predict = aex.forward(train)[0];               //前向计算
            Variable loss = train.sub(predict).square().mean();     //与自身对比生成 mse loss
            toolkit.grad2zero(loss);                                //梯度清零
            toolkit.backward(loss);                                 //反向梯度计算，把梯度放入参数grad字段里
            optimizer.update();                                     //更新器更新，更新化器监视的参数
            if (i % 100 == 0) {
                System.out.println("loss:" + loss.data.scalar);     //查看损失
            }
        }

    }


}

