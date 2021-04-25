package com.ml4ai.deeplearning.demo;

import com.ml4ai.deeplearning.nn.Nd4jUtil;
import com.ml4ai.deeplearning.nn.core.Toolkit;
import com.ml4ai.deeplearning.nn.core.Variable;
import com.ml4ai.deeplearning.nn.core.optimizers.BinaryGradientDescent;
import com.ml4ai.deeplearning.nn.core.optimizers.Moment;
import com.ml4ai.deeplearning.nn.core.optimizers.NNOptimizer;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.factory.Nd4j;


/**
 * Created by lee on 2018/3/27.
 */
@Slf4j
public class AutoGrad2 {

    public static void main(String[] arguments) {

        //高斯分布初始化权重矩阵 shape[4*3]
        Variable weights1 = new Variable(Nd4jUtil.randn(new int[]{4, 3}));
        //初始化Bias[3]
        Variable bias1 = new Variable(Nd4jUtil.zero(new int[]{3}));
        //高斯分布初始化权重矩阵 shape[3*2]
        Variable weight2 = new Variable(Nd4jUtil.randn(new int[]{3, 2}));
        //初始化Bias[2]
        Variable bias2 = new Variable(Nd4jUtil.zero(new int[]{2}));

        //训练数据
        Variable train_data = new Variable(Nd4j.create(new double[][]{
                {0.1, 0.2, 0.3, 0.4},
                {0.2, 0.3, 0.4, 0.5}
        }));

        //标签
        Variable label = new Variable(Nd4j.create(new double[][]{
                {1, 0},
                {0, 1}
        }));

        //初始化图工具
        Toolkit t = new Toolkit();
        //初始化参数优化器，监视要优化的参数，用动量优化器，即权重矩阵和偏执各二项
        NNOptimizer optimizer = new BinaryGradientDescent(new Variable[]{weights1, weight2, bias1, bias2}, 0.001);

        for (int i = 0; i < 10000; i++) {
            //前向计算
            Variable var = train_data.matMul(weights1); //训练数据乘权重矩阵
            var = var.addVec(bias1);    //加偏执项
            var = var.sigmoid();        //应用sigmoid函数
            var = var.matMul(weight2);  //乘第二层权重矩阵
            var = var.addVec(bias2);    //加第二项偏执向量
            Variable predict = var.sigmoid();        //应用sigmoid函数

            Variable loss = predict.sub(label).square().mean();     //损失函数 loss = mean((predict - label)²) MSE均方误差
            t.grad2zero(loss);  //用工具将loss计算图中全部参数的导数置0
            t.backward(loss);   //从loss开始反向自动微分，并在参数中计算每个参数导数
            optimizer.update(); //优化器更新监视的参数，依据计算的导数

            if (i % 100 == 0) {
                System.out.println("损失 " + loss.data.scalar);
            }
        }

        //预测测试数据
        Variable test_data = new Variable(Nd4j.create(new double[]
                {0.09, 0.18, 0.31, 0.42}));

        Variable var = test_data.matMul(weights1); //训练数据乘权重矩阵
        var = var.addVec(bias1);    //加偏执项
        var = var.sigmoid();        //应用sigmoid函数
        var = var.matMul(weight2);  //乘第二层权重矩阵
        var = var.addVec(bias2);    //加第二项偏执向量
        Variable predict_test = var.sigmoid();        //应用sigmoid函数

        //最大参数第一类
        System.out.println(predict_test.data.tensor);

    }


}
