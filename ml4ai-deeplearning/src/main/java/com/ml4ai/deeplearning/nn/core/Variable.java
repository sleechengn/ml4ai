package com.ml4ai.deeplearning.nn.core;

import lombok.*;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import static com.ml4ai.deeplearning.nn.core.Calc.*;

import java.util.function.Consumer;

/**
 * Created by lee on 2018/3/27.
 */
@AllArgsConstructor
@NoArgsConstructor
@Setter
@Getter
public class Variable {

    public boolean isRequireGrad = false;

    public static Variable NULL_VARIABLE = null;

    public Tensor data;

    public Tensor grad;

    public Operation operation;

    public Variable[] dependencies;

    public INDArrayIndex segmentation[];

    public double ground;

    public int dimension;

    public int backward = 0;

    public void backwardTreeOperation(Consumer<Variable> operation) {
        operation.accept(this);
        if (dependencies != null) {
            for (Variable dependency : dependencies) {
                dependency.backwardTreeOperation(operation);
            }
        }
    }

    public Variable(INDArray data) {
        this.isRequireGrad = false;
        this.data = new Tensor(data);
        this.operation = Operation.ASSIGN;
    }

    public Variable(INDArray data, boolean isRequireGrad) {
        this.isRequireGrad = isRequireGrad;
        this.operation = Operation.ASSIGN;
        this.data = new Tensor(data);
    }

    public Variable(double data) {
        this.isRequireGrad = false;
        this.data = new Tensor(data);
        this.operation = Operation.ASSIGN;
    }

    public Variable(double data, boolean isRequireGrad) {
        this.isRequireGrad = isRequireGrad;
        this.operation = Operation.ASSIGN;
        this.data = new Tensor(data);
    }

    //矩阵乘法
    public Variable matMul(Variable variable) {
        if (this.data.type == Tensor.MATRIX_TYPE && variable.data.type == Tensor.MATRIX_TYPE) {
            INDArray x_data = this.data.tensor;
            INDArray y_data = variable.data.tensor;
            INDArray z_data = x_data.mmul(y_data);

            Variable var = new Variable();
            var.setDependencies(new Variable[]{this, variable});
            var.setOperation(Operation.MatMul);
            var.setData(new Tensor(z_data));
            return var;
        } else {
            throw new IllegalStateException("此运算符不支持的数据类型！");
        }
    }

    //矩阵（二阶张量）乘标量
    public Variable mulScalar(Variable variable) {
        if (this.data.type == Tensor.MATRIX_TYPE && variable.data.type == Tensor.SCALAR_TYPE) {
            INDArray x = this.data.tensor;
            double y = variable.data.scalar;
            INDArray z = x.mul(y);

            Variable var = new Variable();
            var.setDependencies(new Variable[]{this, variable});
            var.setOperation(Operation.MulScalar);
            var.setData(new Tensor(z));
            return var;
        } else {
            throw new IllegalStateException("此运算符不支持的数据类型！");
        }
    }

    //同类相加法
    public Variable add(Variable variable) {
        //张量加法
        if (this.data.type > Tensor.SCALAR_TYPE && variable.data.type == this.data.type) {
            INDArray x = this.data.tensor;
            INDArray y = variable.data.tensor;
            INDArray z = x.add(y);
            Variable var = new Variable();
            var.setDependencies(new Variable[]{this, variable});
            var.setOperation(Operation.Add);
            var.setData(new Tensor(z));
            return var;
        } else if (this.data.type == Tensor.SCALAR_TYPE && this.data.type == variable.data.type) {
            double x = this.data.scalar;
            double y = variable.data.scalar;
            double z = x + y;

            Variable var = new Variable();
            var.setDependencies(new Variable[]{this, variable});
            var.setOperation(Operation.Add);
            var.setData(new Tensor(z));
            return var;
        } else {
            throw new IllegalStateException("错误的值类型,不支持[" + data.type + "]");
        }
    }

    //减法
    public Variable sub(Variable variable) {
        if (this.data.type > Tensor.SCALAR_TYPE && this.data.type == variable.data.type) {
            INDArray x = this.data.tensor;
            INDArray y = variable.data.tensor;
            INDArray z = x.sub(y);
            Variable var = new Variable();
            var.setDependencies(new Variable[]{this, variable});
            var.setOperation(Operation.Sub);
            var.setData(new Tensor(z));
            return var;
        }
        if (this.data.type == Tensor.SCALAR_TYPE && this.data.type == variable.data.type) {
            double x = this.data.scalar;
            double y = variable.data.scalar;
            double z = x - y;

            Variable var = new Variable();
            var.setDependencies(new Variable[]{this, variable});
            var.setOperation(Operation.Sub);
            var.setData(new Tensor(z));
            return var;
        }
        throw new IllegalStateException("错误的值类型,不支持[" + data.getType() + "]");
    }

    //哈達馬
    public Variable hadamard(Variable variable) {
        if (this.data.type > Tensor.SCALAR_TYPE && this.data.type == variable.data.type) {
            INDArray x = this.data.tensor;
            INDArray y = variable.data.tensor;
            INDArray z = x.mul(y);
            Variable var = new Variable();
            var.setDependencies(new Variable[]{this, variable});
            var.setOperation(Operation.Hadamard);
            var.setData(new Tensor(z));
            return var;
        }
        throw new IllegalStateException("错误的值类型,不支持[" + data.getType() + "]");
    }

    //激活函数
    public Variable sigmoid() {
        if (this.data.type > Tensor.SCALAR_TYPE) {
            INDArray x = this.data.tensor;
            INDArray y = Nd4j.zeros(x.shape());
            Nd4j.copy(x, y);
            INDArray z = Activation.SIGMOID.getActivationFunction().getActivation(y, true);

            Variable var = new Variable();
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.Sigmoid);
            var.setData(new Tensor(z));
            return var;
        }
        if (this.data.type == Tensor.SCALAR_TYPE) {
            double x = this.data.scalar;
            double z, y;
            y = mathSigmoid(x);
            z = y;
            Variable var = new Variable();
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.Sigmoid);
            var.setData(new Tensor(z));
            return var;
        }
        throw new IllegalStateException("错误的值类型,不支持[" + data.getType() + "]");
    }

    //双曲正切
    public Variable tanh() {
        if (this.data.type > Tensor.SCALAR_TYPE) {
            INDArray x = this.data.tensor;
            INDArray y = Nd4j.zeros(x.shape());
            Nd4j.copy(x, y);
            INDArray z = Activation.TANH.getActivationFunction().getActivation(y, true);

            Variable var = new Variable();
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.TanH);
            var.setData(new Tensor(z));
            return var;
        }
        if (this.data.type == Tensor.SCALAR_TYPE) {
            double x = this.data.scalar;
            double z, y;
            y = mathTanh(x);
            z = y;
            Variable var = new Variable();
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.TanH);
            var.setData(new Tensor(z));
            return var;
        }
        throw new IllegalStateException("错误的值类型,不支持[" + data.getClass().getName() + "]");
    }

    //平均值
    public Variable mean() {
        if (this.data.type > Tensor.SCALAR_TYPE) {
            INDArray x = this.data.tensor;
            double y = x.meanNumber().doubleValue();
            Variable var = new Variable();
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.Mean);
            var.setData(new Tensor(y));
            return var;
        }
        if (this.data.type == Tensor.SCALAR_TYPE) {
            return this;
        }
        throw new IllegalStateException("错误的值类型,不支持[" + data.getType() + "]");
    }

    //求和
    public Variable sum() {
        if (this.data.type > Tensor.SCALAR_TYPE) {
            INDArray x = this.data.tensor;
            double y = x.sumNumber().doubleValue();
            Variable var = new Variable();
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.Sum);
            var.setData(new Tensor(y));
            return var;
        }
        if (this.data.type == Tensor.SCALAR_TYPE) {
            return this;
        }
        throw new IllegalStateException("错误的值类型,不支持[" + data.type + "]");
    }

    //求和
    public Variable log(double y) {
        if (this.data.type > Tensor.SCALAR_TYPE) {
            INDArray x_data = this.data.tensor;
            INDArray x_cp = Nd4j.create(x_data.shape());
            Nd4j.copy(x_data, x_cp);
            INDArray z = Transforms.log(x_cp, y);
            Variable var = new Variable(z);
            var.ground = y;
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.Log);
            return var;
        } else if (this.data.type == Tensor.SCALAR_TYPE) {
            double x_data = this.data.scalar;
            double z = Math.log(x_data) / Math.log(y);
            Variable var = new Variable(z);
            var.ground = y;
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.Log);
            return var;
        }
        throw new IllegalStateException("错误的值类型,不支持[" + data.type + "]");
    }


    //平方
    public Variable square() {
        if (this.data.type > Tensor.SCALAR_TYPE) {
            INDArray x = this.data.tensor;
            INDArray y, z;
            y = x.mul(x);
            Variable var = new Variable(y);
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.Square);
            return var;
        }
        if (this.data.type == Tensor.SCALAR_TYPE) {
            double x = this.data.scalar;
            double y, z;
            y = Math.pow(x, 2);
            Variable var = new Variable(y);
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.Square);
            return var;
        }
        throw new IllegalStateException("错误的值类型,不支持[" + data.type + "]");
    }

    //线性整流
    public Variable relu() {
        if (this.data.type > Tensor.SCALAR_TYPE) {
            INDArray x = this.data.tensor;
            INDArray y = Nd4j.create(x.shape());
            Nd4j.copy(x, y);
            INDArray z = Activation.RELU.getActivationFunction().getActivation(y, true);
            Variable var = new Variable(z);
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.RELU);
            return var;
        }
        if (this.data.type == Tensor.SCALAR_TYPE) {
            double x = this.data.scalar;
            Variable var = new Variable(x > 0 ? x : 0D);
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.RELU);
            return var;
        }
        throw new IllegalStateException("错误的值类型,不支持[" + data.type + "]");
    }

    //线性整流
    public Variable leaky_relu() {
        if (this.data.type > Tensor.SCALAR_TYPE) {
            INDArray x = this.data.tensor;
            INDArray y = Nd4j.create(x.shape());
            Nd4j.copy(x, y);
            INDArray z = Activation.LEAKYRELU.getActivationFunction().getActivation(y, true);
            Variable var = new Variable(z);
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.LRELU);
            return var;
        }
        if (this.data.type == Tensor.SCALAR_TYPE) {
            double x = this.data.scalar;
            Variable var = new Variable(x > 0 ? x : x * 0.01);
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.LRELU);
            return var;
        }
        throw new IllegalStateException("错误的值类型,不支持[" + data.type + "]");
    }

    public Variable addVec(Variable variable) {
        if (this.data.type > Tensor.SCALAR_TYPE && this.data.type == Tensor.MATRIX_TYPE) {
            INDArray x = this.data.tensor;
            INDArray y = variable.data.tensor;
            int[] _x_shape = x.shape();
            int[] _y_shape = y.shape();

            int x_row = _x_shape[0];
            int x_column = _x_shape[1];
            int y_column = _y_shape[1];

            if (x.rank() != 2) {
                throw new IllegalStateException("现在只支持矩阵加向量，此变量不是矩阵，张量阶：" + x.rank());
            }
            if (y.rank() != 2 || _y_shape[0] != 1) {
                throw new IllegalStateException("些操作只能与向量相加，此变量不是向量，张量阶：" + y.rank());
            }
            if (x_column != y_column) {
                throw new IllegalThreadStateException("列不相同，无法合并");
            }
            //构造y向量到方阵对角线
            INDArray y_sq = Nd4j.diag(y);
            //构造全1的矩阵
            INDArray ones = Nd4j.ones(new int[]{x_row, x_column});
            INDArray stackVecMat = ones.mmul(y_sq);
            INDArray z = x.add(stackVecMat);

            Variable var = new Variable(z);
            var.setDependencies(new Variable[]{this, variable});
            var.setOperation(Operation.AddVec);
            return var;
        } else {
            throw new IllegalStateException("错误的值类型,不支持[" + data.type + "]");
        }
    }

    /**
     * 截取子张量
     *
     * @param indices
     * @return
     */
    public Variable separate(INDArrayIndex... indices) {
        Variable target = new Variable(this.data.tensor.get(indices));
        target.segmentation = indices;
        target.setDependencies(new Variable[]{this});
        target.operation = Operation.Separate;
        return target;
    }

    /**
     * 在shape of rank中重构以shape为维度元素每维边界的张量
     *
     * @param shape
     * @return
     */
    public Variable refactor(int... shape) {
        INDArray x = this.data.tensor;
        Variable var = new Variable(x.reshape(shape));
        var.operation = Operation.Refactor;
        var.dependencies = new Variable[]{this};
        return var;
    }

    /**
     * 张量联系，通過某個維度
     *
     * @param variable
     * @return
     */
    public Variable connect(Variable variable, int dim) {
        INDArray x = this.data.tensor;
        INDArray t = variable.data.tensor;
        INDArray z = Nd4j.concat(dim, x, t);
        Variable var = new Variable(z);
        var.dimension = dim;
        var.setOperation(Operation.Combine);
        var.dependencies = new Variable[]{this, variable};
        return var;
    }

    /**
     * 正弦函数
     */
    public Variable sin() {
        if (this.data.type > Tensor.SCALAR_TYPE) {
            INDArray x = this.data.tensor;
            INDArray y = Transforms.sin(x);
            Variable var = new Variable(y);
            var.setOperation(Operation.Sin);
            var.dependencies = new Variable[]{this};
            return var;
        } else if (this.data.type == Tensor.SCALAR_TYPE) {
            double x = this.data.scalar;
            double y = Math.sin(x);
            Variable var = new Variable(y);
            var.setOperation(Operation.Sin);
            var.setDependencies(new Variable[]{this});
            return var;
        } else {
            throw new UnsupportedOperationException("还不支持你选择的类型");
        }
    }

    /**
     * 余弦函数
     */
    public Variable cos() {
        if (this.data.type > Tensor.SCALAR_TYPE) {
            INDArray x = this.data.tensor;
            INDArray y = Transforms.cos(x);
            Variable var = new Variable(y);
            var.setOperation(Operation.Cos);
            var.dependencies = new Variable[]{this};
            return var;
        } else if (this.data.type == Tensor.SCALAR_TYPE) {
            double x = this.data.scalar;
            double y = Math.cos(x);
            Variable var = new Variable(y);
            var.setOperation(Operation.Cos);
            var.setDependencies(new Variable[]{this});
            return var;
        } else {
            throw new UnsupportedOperationException("还不支持你给定的类型");
        }
    }

    @Override
    public String toString() {
        if (data.type > 0) {
            return data.tensor.toString();
        } else {
            return Double.valueOf(data.scalar).toString();
        }
    }
}
