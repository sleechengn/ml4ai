package com.ml4ai.deeplearning.nn.core;

import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;

@Data
public class Tensor {

    public final static int SCALAR_RANK = 0;
    public final static int MATRIX_RANK = 2;

    public int rank;
    public double scalar;
    public INDArray tensor;

    public Tensor(INDArray tensor) {
        this.tensor = tensor;
        this.rank = tensor.rank();
    }

    public Tensor(double scalar) {
        this.scalar = scalar;
        this.rank = SCALAR_RANK;
    }
}
