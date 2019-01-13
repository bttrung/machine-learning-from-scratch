package com.diy.ml.regression;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.inverse.InvertMatrix;


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

            int columnIndex = 1;
            INDArray thetaChange = Nd4j.zeros(gradient.length(), columnIndex);
            for (int rowElement = 0; rowElement < gradient.length(); rowElement++) {
                double grad = gradient.getDouble(rowElement, columnIndex);
                thetaChange.putScalar(rowElement, columnIndex, grad * alpha / m);

            }
            theta = theta.sub(thetaChange);
        }
        return theta;
    }

    public INDArray featureNormalize(INDArray inputArray) {
        INDArray normalized = inputArray.dup();
        INDArray meanI = normalized.mean(0);
        INDArray sigmaI = normalized.std(0);

        for (int col = 0; col < normalized.columns(); col++) {
            for (int row = 0; row < normalized.rows(); row++) {
                double aDouble = normalized.getDouble(row, col);
                double normalizeVal = (aDouble - meanI.getDouble(1, col)) / sigmaI.getDouble(1, col);
                normalized.putScalar(row, col, normalizeVal);
            }
        }
        return normalized;
    }

    public INDArray normalEquation(INDArray trainingSetArr, INDArray outputArr) {
        INDArray transposeX = trainingSetArr.transpose();
        INDArray mmul = transposeX.mmul(trainingSetArr);
        INDArray pinvert = InvertMatrix.pinvert(mmul, true);
        return pinvert.mmul(transposeX).mmul(outputArr);
    }
}
