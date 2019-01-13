package com.diy.ml.regression;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class LinearRegression {

    public double computeCost(INDArray trainingInputs, INDArray actualOutput, INDArray theta) {
        int m = actualOutput.rows();
        INDArray hypothesis = trainingInputs.mmul(theta);
        double squaredDistance = hypothesis.squaredDistance(actualOutput);
        return squaredDistance / (2 * m);
    }

    public INDArray gradientDescent(INDArray trainingInputs, INDArray actualOutput, INDArray theta,
                                    double alpha, int numberOfIterations) {

        int m = actualOutput.rows();
        for (int i = 0; i < numberOfIterations; i++) {
            INDArray hypothesis = trainingInputs.mmul(theta);

            INDArray errors = hypothesis.sub(actualOutput);
            INDArray gradient = trainingInputs.transpose().mmul(errors);

            INDArray thetaChange = Nd4j.zeros(gradient.length(), 1);
            for (int element = 0; element < gradient.length(); element++) {
                double grad = gradient.getDouble(element, 1);
                thetaChange.putScalar(element, 1, grad * alpha / m);

            }

            theta = theta.sub(thetaChange);

        }

        return theta;

    }

}
