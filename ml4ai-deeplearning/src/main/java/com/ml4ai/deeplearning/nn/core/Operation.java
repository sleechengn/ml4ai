package com.ml4ai.deeplearning.nn.core;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * Created by lee on 2018/3/27.
 */
@AllArgsConstructor
@Getter
public enum Operation {

    MatMul("矩阵-乘积"),
    Dot("矩阵-点积"),
    Hadamard("矩阵-哈达马积"),
    MulScalar("矩阵乘标量"),
    Kronecker("张量-直积"),
    Sum("张量-求和"),
    Log("对数"),
    Mean("张量-求平均"),
    Add("同类-加法"),
    AddVec("矩阵-加向量"),
    Sub("同类-减法"),
    Sigmoid("Sigmoid函數"),
    TanH("Tanh函數"),
    Square("平方"),
    RELU("Relu函数"),
    LRELU("LeakyRelu"),
    Separate("分割"),
    Combine("连接"),
    Refactor("重构"),
    Sin("Sin"),
    Cos("Cos函数"),
    ASSIGN("指定");

    private String name;

}
