package com.ml4ai.deeplearning.nn.core;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import static com.ml4ai.deeplearning.nn.core.Calc.*;

import java.util.LinkedList;
import java.util.List;

/**
 * Created by lee on 2018/3/31.
 */
public class Toolkit {

    public void grad2zero(Variable variable) {
        variable.backwardTreeOperation(var -> {
            var.setBackward(0);
            if (var.data.rank > Tensor.SCALAR_RANK) {
                if (var.grad == null) {
                    var.setGrad(new Tensor(Nd4j.zeros(var.data.tensor.shape())));
                } else {
                    var.grad.tensor.muli(0);
                }
            }
            if (var.data.rank == Tensor.SCALAR_RANK) {
                if (var.grad == null) {
                    var.setGrad(new Tensor(Double.valueOf(0)));
                } else {
                    var.grad.scalar = 0;
                }
            }
        });
    }

    public void backward(Variable variable) {
        backward(variable, 1D);
    }

    public void backward(Variable variable, double scalar) {
        if (variable.data.rank == Tensor.SCALAR_RANK) {
            variable.setGrad(new Tensor(scalar));
            variable.backward = 0;
            backwardWith(variable);
        } else
            throw new IllegalStateException("不支持反向传播的类型[" + variable.data.rank + "]");
    }

    public void backward(Variable variable, INDArray tensor) {
        if (variable.data.rank > Tensor.SCALAR_RANK) {
            variable.setGrad(new Tensor(tensor));
            variable.backward = 0;
            backwardWith(variable);
        } else
            throw new IllegalStateException("不支持反向传播的类型[" + variable.data.rank + "]");
    }

    public void backwardWith(Variable variable) {
        variable.backwardTreeOperation(
                var ->
                        var.backward = 0
        );
        LinkedList<Variable> leaveVars = new LinkedList<>();
        leaveVars.add(variable);
        while (leaveVars.size() > 0) {
            Variable current = leaveVars.removeFirst();
            if (current.backward > 0) {
                current.backward++;
            } else {
                current.backward++;
                if (current.getDependencies() != null && current.getDependencies().length > 0) {
                    Variable[] dependencies = current.getDependencies();
                    for (Variable dep : dependencies) {
                        leaveVars.add(dep);
                    }
                }
            }
        }
        variable.backward = 0;
        LinkedList<Variable> variables = new LinkedList<>();
        variables.add(variable);
        while (variables.size() > 0) {
            Variable current = variables.removeFirst();
            switch (current.getOperation()) {
                case Add: {
                    Variable x = current.dependencies[0];
                    Variable y = current.dependencies[1];
                    /**
                     * 张量相加
                     */
                    if (current.data.rank > Tensor.SCALAR_RANK && x.data.rank == y.data.rank && x.data.rank == current.data.rank) {
                        x.grad.tensor.addi(current.grad.tensor);
                        y.grad.tensor.addi(current.grad.tensor);
                        if (--x.backward == 0) {
                            variables.add(x);
                        }
                        if (--y.backward == 0) {
                            variables.add(y);
                        }
                        /**
                         * 标量相加
                         */
                    } else if (current.data.rank == Tensor.SCALAR_RANK && x.data.rank == y.data.rank && x.data.rank == current.data.rank) {
                        x.grad.scalar += current.grad.scalar;
                        y.grad.scalar += current.grad.scalar;
                        if (--x.backward == 0) {
                            variables.add(x);
                        }
                        if (--y.backward == 0) {
                            variables.add(y);
                        }
                    } else {
                        throw new IllegalStateException("不支持此类型：Rank:" + current.data.rank + " + " + "Rank:" + x.data.rank + " Rank:" + y.data.rank);
                    }
                }
                break;
                case Sub: {
                    Variable x = current.dependencies[0];
                    Variable y = current.dependencies[1];
                    if (current.data.rank == Tensor.SCALAR_RANK) {
                        x.grad.scalar += current.grad.scalar;
                        y.grad.scalar -= 1 * current.grad.scalar;
                        if (--x.backward == 0) {
                            variables.add(x);
                        }
                        if (--y.backward == 0) {
                            variables.add(y);
                        }
                    } else if (current.data.rank > Tensor.SCALAR_RANK) {
                        x.grad.tensor.addi(current.grad.tensor);
                        y.grad.tensor.subi(current.grad.tensor);
                        if (--x.backward == 0) {
                            variables.add(x);
                        }
                        if (--y.backward == 0) {
                            variables.add(y);
                        }
                    } else {
                        throw new IllegalStateException("不支持");
                    }
                }
                break;
                case Mean: {
                    Variable x = current.dependencies[0];
                    if (current.data.rank == Tensor.SCALAR_RANK) {
                        if (x.data.rank > Tensor.SCALAR_RANK) {
                            INDArray x_data = x.data.tensor;
                            List<Integer> shape = new LinkedList<>();
                            int[] x_data_shape = x_data.shape();
                            for (int i = 0; i < x_data_shape.length; i++) {
                                shape.add(x_data_shape[i]);
                            }
                            int product = shape.stream().reduce((a, b) -> a * b).get();
                            x.grad.tensor.addi(Nd4j.ones(x_data_shape).div(product).mul((current.grad.scalar)));
                            if (--x.backward == 0) {
                                variables.add(x);
                            }
                        } else if (x.data.rank == Tensor.SCALAR_RANK) {
                            x.grad.scalar += current.grad.scalar;
                            if (--x.backward == 0) {
                                variables.add(x);
                            }
                        } else {
                            throw new IllegalStateException("不支持的类型");
                        }
                    } else {
                        throw new IllegalStateException("不支持的类型");
                    }
                }
                break;
                case Sum: {
                    Variable x = current.dependencies[0];
                    if (current.data.rank == Tensor.SCALAR_RANK) {
                        if (x.data.rank > Tensor.SCALAR_RANK) {
                            INDArray x_data = x.data.tensor;
                            int[] x_data_shape = x_data.shape();
                            x.grad.tensor.addi(Nd4j.ones(x_data_shape).mul(current.grad.scalar));
                            if (--x.backward == 0) {
                                variables.add(x);
                            }
                        } else if (x.data.rank == Tensor.SCALAR_RANK) {
                            x.grad.scalar += current.grad.scalar;
                            if (--x.backward == 0) {
                                variables.add(x);
                            }
                        } else
                            throw new IllegalStateException("不支持");
                    } else {
                        throw new IllegalStateException("不支持");
                    }
                }
                break;
                case Square: {
                    Variable x = current.dependencies[0];
                    if (current.data.rank > Tensor.SCALAR_RANK) {
                        INDArray x_data = x.data.tensor;
                        x.grad.tensor.addi(x_data.mul(2).mul(current.grad.tensor));
                        if (--x.backward == 0) {
                            variables.add(x);
                        }
                    } else if (current.data.rank == Tensor.SCALAR_RANK) {
                        x.grad.scalar += x.data.scalar * 2 * current.grad.scalar;
                        if (--x.backward == 0) {
                            variables.add(x);
                        }
                    } else {
                        throw new IllegalStateException("不支持");
                    }
                }
                break;
                case MatMul: {
                    if (current.data.rank > Tensor.SCALAR_RANK) {
                        Variable x = current.dependencies[0];
                        Variable y = current.dependencies[1];
                        y.grad.tensor.addi((x.data.tensor.transpose().mmul(current.grad.tensor)));
                        x.grad.tensor.addi(current.grad.tensor.mmul((y.data.tensor.transpose())));
                        if (--x.backward == 0) {
                            variables.add(x);
                        }
                        if (--y.backward == 0) {
                            variables.add(y);
                        }
                    } else {
                        throw new IllegalStateException("非法");
                    }
                }
                break;
                case Sigmoid: {
                    if (current.data.rank > Tensor.SCALAR_RANK) {
                        Variable x = current.dependencies[0];
                        INDArray x_data = current.dependencies[0].data.tensor;
                        INDArray x_data_c = Nd4j.create(x_data.shape());
                        Nd4j.copy(x_data, x_data_c);
                        x.grad.tensor.addi(Activation.SIGMOID.getActivationFunction().backprop(x_data_c, current.grad.tensor).getFirst());
                        if (--x.backward == 0) {
                            variables.add(x);
                        }
                    } else if (current.data.rank == Tensor.SCALAR_RANK) {
                        Variable y = current.dependencies[0];
                        double x = y.data.scalar;
                        y.grad.scalar += mathSigmoid(x) * (1 - mathSigmoid(x));
                        if (--y.backward == 0) {
                            variables.add(y);
                        }
                    } else {
                        throw new IllegalStateException("不支持");
                    }
                }
                break;
                case TanH: {
                    if (current.data.rank > Tensor.SCALAR_RANK) {
                        Variable x = current.dependencies[0];
                        INDArray x_data = x.data.tensor;
                        INDArray x_data_c = Nd4j.zeros(x_data.shape());
                        Nd4j.copy(x_data, x_data_c);
                        x.grad.tensor.addi(Activation.TANH.getActivationFunction().backprop(x_data_c, current.grad.tensor).getFirst());
                        if (--x.backward == 0) {
                            variables.add(x);
                        }
                    } else if (current.data.rank == Tensor.SCALAR_RANK) {
                        Variable y = current.dependencies[0];
                        double x = y.data.scalar;
                        y.grad.scalar += 1 - Math.pow(mathTanh(x), 2);
                        if (--y.backward == 0) {
                            variables.add(y);
                        }
                    } else {
                        throw new IllegalStateException("不支持");
                    }
                }
                break;
                case RELU: {
                    if (current.data.rank > Tensor.SCALAR_RANK) {
                        Variable x = current.dependencies[0];
                        INDArray x_data = x.data.tensor;
                        INDArray x_data_cpp = Nd4j.create(x_data.shape());
                        Nd4j.copy(x_data, x_data_cpp);
                        x.grad.tensor.addi(Activation.RELU.getActivationFunction().backprop(x_data_cpp, current.grad.tensor).getFirst());
                        if (--x.backward == 0) {
                            variables.add(x);
                        }
                    } else if (current.data.rank == Tensor.SCALAR_RANK) {
                        Variable y = current.dependencies[0];
                        double x = y.data.scalar;
                        y.grad.scalar += x >= 0 ? 1 : 0;
                        if (--y.backward == 0) {
                            variables.add(y);
                        }
                    } else {
                        throw new IllegalStateException("不支持");
                    }
                }
                break;
                case LRELU: {
                    if (current.data.rank > Tensor.SCALAR_RANK) {
                        Variable x = current.dependencies[0];
                        INDArray x_data = x.data.tensor;
                        INDArray x_data_cpp = Nd4j.create(x_data.shape());
                        Nd4j.copy(x_data, x_data_cpp);
                        x.grad.tensor.addi(Activation.LEAKYRELU.getActivationFunction().backprop(x_data_cpp, current.grad.tensor).getFirst());
                        if (--x.backward == 0) {
                            variables.add(x);
                        }
                    } else if (current.data.rank == Tensor.SCALAR_RANK) {
                        Variable y = current.dependencies[0];
                        double x = y.data.scalar;
                        y.grad.scalar += x >= 0 ? 1 : 0.01;
                        if (--y.backward == 0) {
                            variables.add(y);
                        }
                    } else {
                        throw new IllegalStateException("不支持");
                    }
                }
                break;
                case AddVec: {
                    Variable x = current.dependencies[0];
                    Variable y = current.dependencies[1];
                    INDArray c_grad = current.grad.tensor;
                    INDArray x_data = x.data.tensor;
                    INDArray oneColumn = Nd4j.ones(new int[]{1, x_data.shape()[0]});
                    INDArray vecGradMat = oneColumn.mmul(c_grad);
                    x.grad.tensor.addi(c_grad);
                    y.grad.tensor.addi(vecGradMat.getRow(0));
                    if (--y.backward == 0) {
                        variables.add(y);
                    }
                    if (--x.backward == 0) {
                        variables.add(x);
                    }
                }
                break;
                case MulScalar: {
                    INDArray current_grad = current.grad.tensor;
                    Variable x = current.dependencies[0];
                    Variable y = current.dependencies[1];
                    INDArray x_data = x.data.tensor;
                    double y_data = y.data.scalar;
                    y.grad.scalar += x_data.mul(current_grad).sumNumber().doubleValue();
                    x.grad.tensor.addi(current_grad.mul(y_data));
                    if (--y.backward == 0) {
                        variables.add(y);
                    }
                    if (--x.backward == 0) {
                        variables.add(x);
                    }
                }
                break;
                case Hadamard: {
                    INDArray current_gradient = current.grad.tensor;
                    Variable x = current.dependencies[0];
                    Variable y = current.dependencies[1];
                    INDArray x_data = x.data.tensor;
                    INDArray y_data = y.data.tensor;
                    x.grad.tensor.addi(current_gradient.mul(y_data));
                    y.grad.tensor.addi(current_gradient.mul(x_data));
                    if (--y.backward == 0) {
                        variables.add(y);
                    }
                    if (--x.backward == 0) {
                        variables.add(x);
                    }
                }
                case ASSIGN: {

                }
                break;
                case Log: {
                    if (current.data.rank > Tensor.SCALAR_RANK) {
                        INDArray current_gradient = current.grad.tensor;
                        Variable x = current.dependencies[0];
                        INDArray x_dat = x.data.tensor;
                        x.grad.tensor.addi(current_gradient.mul(Nd4j.ones(x_dat.shape()).div(x_dat).mul(Math.log(current.ground))));
                        if (--x.backward == 0) {
                            variables.add(x);
                        }
                    } else if (current.data.rank == Tensor.SCALAR_RANK) {
                        double current_gradient = current.grad.scalar;
                        Variable x = current.dependencies[0];
                        double x_dat = x.data.scalar;
                        x.grad.scalar += current_gradient * Math.log(current.ground) / x_dat;
                        if (--x.backward == 0) {
                            variables.add(x);
                        }
                    } else {
                        throw new IllegalStateException("不支持");
                    }
                }
                break;
                case Separate: {
                    if (current.data.rank > Tensor.SCALAR_RANK) {
                        INDArray x = current.dependencies[0].data.tensor;
                        INDArray grad_x = current.dependencies[0].grad.tensor;
                        INDArrayIndex[] segmentation = current.segmentation;
                        INDArray subGrad = grad_x.get(segmentation);
                        subGrad.addi(current.grad.tensor);
                        grad_x.put(segmentation, subGrad);
                        if (--current.dependencies[0].backward == 0) {
                            variables.add(current.dependencies[0]);
                        }
                    } else {
                        throw new RuntimeException("不支持的类型:" + current.data.rank);
                    }
                }
                break;
                case Refactor: {
                    current.dependencies[0].grad.tensor.addi(current.grad.tensor.reshape(current.dependencies[0].data.tensor.shape()));
                    if (--current.dependencies[0].backward == 0) {
                        variables.add(current.dependencies[0]);
                    }
                }
                break;
                case Dot: {
                    throw new UnsupportedOperationException("不支持点积的反射传播！");
                }
                case Combine: {
                    INDArray x = current.data.tensor;
                    INDArray y = current.dependencies[0].data.tensor;
                    INDArray z = current.dependencies[1].data.tensor;
                    int dimension = current.dimension;
                    int x_rank = x.rank();
                    INDArrayIndex[] indices = new INDArrayIndex[x_rank];
                    INDArrayIndex[] elIndices = new INDArrayIndex[x_rank];
                    for (int i = 0; i < x_rank; i++) {
                        if (i == dimension) {
                            indices[i] = NDArrayIndex.interval(0, y.shape()[i]);
                            elIndices[i] = NDArrayIndex.interval(y.shape()[i], x.shape()[i]);
                        } else {
                            indices[i] = NDArrayIndex.all();
                            elIndices[i] = NDArrayIndex.all();
                        }
                    }
                    current.dependencies[0].grad.tensor.addi(current.grad.tensor.get(indices));
                    current.dependencies[1].grad.tensor.addi(current.grad.tensor.get(elIndices));
                    if (--current.dependencies[0].backward == 0) {
                        variables.add(current.dependencies[0]);
                    }
                    if (--current.dependencies[1].backward == 0) {
                        variables.add(current.dependencies[1]);
                    }
                }
                break;
                case Sin: {
                    if (current.data.rank > Tensor.SCALAR_RANK) {
                        Variable input_x = current.dependencies[0];
                        INDArray x_data = input_x.data.tensor;
                        input_x.grad.tensor.addi(Transforms.cos(x_data).mul(current.grad.tensor));
                        if (--input_x.backward == 0) {
                            variables.add(input_x);
                        }
                    } else if (current.data.rank == Tensor.SCALAR_RANK) {
                        Variable input = current.dependencies[0];
                        double input_x = input.data.scalar;
                        input.grad.scalar += current.grad.scalar * Math.cos(input_x);
                        if (--input.backward == 0) {
                            variables.add(input);
                        }
                    } else {
                        throw new UnsupportedOperationException("不支持的操作");
                    }
                }
                break;
                case Cos: {
                    if (current.data.rank > Tensor.SCALAR_RANK) {
                        Variable input_x = current.dependencies[0];
                        INDArray x_data = input_x.data.tensor;
                        input_x.grad.tensor.subi(Transforms.sin(x_data).mul(current.grad.tensor));
                        if (--input_x.backward == 0) {
                            variables.add(input_x);
                        }
                    } else if (current.data.rank == Tensor.SCALAR_RANK) {
                        Variable input = current.dependencies[0];
                        double input_x = input.data.scalar;
                        input.grad.scalar -= current.grad.scalar * Math.sin(input_x);
                        if (--input.backward == 0) {
                            variables.add(input);
                        }
                    } else {
                        throw new UnsupportedOperationException("不支持的操作");
                    }
                }
                break;
                case Kronecker: {
                    throw new UnsupportedOperationException("还未实现！");
                }
            }
        }
    }

}
