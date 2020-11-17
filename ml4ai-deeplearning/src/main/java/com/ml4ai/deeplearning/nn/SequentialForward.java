package com.ml4ai.deeplearning.nn;

public class SequentialForward extends BaseForwardNetwork {

    public SequentialForward(ForwardNetwork... forwards) {
        super(forwards);
    }

}
