package com.ml4ai.deeplearning.nn;

import com.ml4ai.deeplearning.nn.core.Variable;

public interface ForwardNetwork {

    // 获取学习参数
    public default Variable[] getParameters() {
        throw new UnsupportedOperationException("该操作未支持");
    }

    //前向传播
    public default Variable[] forward(Variable... inputs) {
        throw new UnsupportedOperationException("该方法没有实现");
    }

}
