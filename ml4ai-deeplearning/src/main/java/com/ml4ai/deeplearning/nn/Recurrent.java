package com.ml4ai.deeplearning.nn;

import com.ml4ai.deeplearning.nn.core.Operation;
import com.ml4ai.deeplearning.nn.core.Variable;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class Recurrent extends BaseForwardNetwork {

    public static class RNNCell extends BaseForwardNetwork {
        private int input_size;
        private int hidden_size;
        private Variable U;
        private Variable W;
        private Variable bias;
        private Operation activation;

        @Override
        public Variable[] getParameters() {
            return new Variable[]{U, W, bias};
        }

        public RNNCell(int input_size, int hidden_size, Operation activation) {
            this.activation = activation;
            this.input_size = input_size;
            this.hidden_size = hidden_size;
            U = new Variable(Nd4j.randn(input_size, hidden_size));
            bias = new Variable(Nd4j.zeros(hidden_size));
            W = new Variable(Nd4j.randn(hidden_size, hidden_size));
        }

        @Override
        public Variable[] forward(Variable... inputs) {
            Variable x = inputs[0];
            int[] tensor_of_shape = x.data.tensor.shape();
            int batch_size = tensor_of_shape[0];
            int time_step = tensor_of_shape[1];
            int input_size = tensor_of_shape[2];

            if (input_size != this.input_size) {
                throw new IllegalStateException("样本特征长度为：" + input_size + "，矩阵特征长度：" + this.input_size);
            }

            Variable state;         //shape of [batch,hidden]
            if (inputs.length > 1) {
                state = inputs[1];  //shape [batch,hidden]
            } else {
                state = new Variable(Nd4j.zeros(batch_size, hidden_size));
            }
            // x of shape is [batch_size,time_step,input_size]
            Variable[] output_dat = new Variable[time_step];
            for (int i = 0; i < time_step; i++) {
                Variable this_time_data = x.separate(NDArrayIndex.all(), NDArrayIndex.point(i), NDArrayIndex.all()); //shape [bat_size,input_size]
                Variable this_time_data_matiply_u = this_time_data.matMul(this.U); //shape [batch_size,hidden]
                Variable this_time_state_matiply_w = state.matMul(W);
                Variable this_time_sum = this_time_data_matiply_u.add(this_time_state_matiply_w).addVec(bias);
                Variable this_time_out; //batch_size,hidden
                switch (activation) {
                    case TanH: {
                        this_time_out = this_time_sum.tanh();
                    }
                    break;
                    case Sigmoid: {
                        this_time_out = this_time_sum.sigmoid();
                    }
                    break;
                    case RELU: {
                        this_time_out = this_time_sum.relu();
                    }
                    break;
                    case LRELU: {
                        this_time_out = this_time_sum.leaky_relu();
                    }
                    break;
                    default: {
                        throw new IllegalStateException("不支持函数：[" + activation.getName() + "]");
                    }
                }
                state = this_time_out;
                output_dat[i] = state.refactor(batch_size, 1, hidden_size);
            }
            // 将堆叠的time_step激活后的数据串起来
            Variable var = Variable.NULL_VARIABLE;
            for (int i = 0; i < time_step; i++) {
                if (i == 0) {
                    var = output_dat[i];
                } else {
                    var = var.connect(output_dat[i], 1);
                }
            }
            return new Variable[]{var, state};
        }
    }

    public Recurrent() {
        throw new UnsupportedOperationException("超类")
                ;
    }

    @Override
    public Variable[] forward(Variable... inputs) {
        return super.forward(inputs);
    }

    @Override
    public Variable[] getParameters() {
        return super.getParameters();
    }
}
