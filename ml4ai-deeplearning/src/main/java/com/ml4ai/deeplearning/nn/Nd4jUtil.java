package com.ml4ai.deeplearning.nn;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by leecheng on 2018/10/28.
 */
public class Nd4jUtil {

    public static INDArray randn(int[] shape) {
        INDArray indArray = Nd4j.randn(shape);
        indArray.muli(0.2);
        return indArray;
    }

    public static INDArray rand(int[] shape) {
        INDArray indArray = Nd4j.rand(shape);
        return indArray;
    }

    public static INDArray zero(int[] shape) {
        INDArray indArray = Nd4j.zeros(shape);
        return indArray;
    }

}
