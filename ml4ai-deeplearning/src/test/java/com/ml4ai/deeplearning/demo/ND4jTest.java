package com.ml4ai.deeplearning.demo;

import com.ml4ai.deeplearning.nn.Nd4jUtil;
import org.nd4j.linalg.api.ndarray.INDArray;

public class ND4jTest {

    public static void main(String[] args) {
        INDArray arr = Nd4jUtil.rand(new int[]{2, 2}).sub(0.5D);
        System.out.println(arr);

        System.out.println(arr.gt(0).sub(0.5).mul(2));

    }

}
