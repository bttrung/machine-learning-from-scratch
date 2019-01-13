package com.diy.ml.regression;

import org.nd4j.linalg.api.ndarray.INDArray;

public class LinearRegression {
    public double computeCost(INDArray trainingInputs, INDArray actualOutput, INDArray theta) {

        int m = actualOutput.rows();
        INDArray hypothesis = trainingInputs.mmul(theta);
//        INDArray errors = hypothesis.sub(actualOutput);
//        double squaredDistance = errors.norm2Number().doubleValue();
        double squaredDistance = hypothesis.squaredDistance(actualOutput);
        return squaredDistance / (2 * m);
    }
}
