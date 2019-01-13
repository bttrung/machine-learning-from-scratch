package com.diy.ml.regression;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import study.FileUtils;

import static org.junit.Assert.assertEquals;


public class LinearRegressionTest {

    LinearRegression linearRegression = new LinearRegression();


    @Test
    public void computeCostTestCase1() {
        double[] training = {1, 2, 1, 3, 1, 4, 1, 5};
        INDArray trainingSetArr = Nd4j.create(training, new int[]{4, 2}, 'c');

        double[] output = {7, 6, 5, 4};
        INDArray outputArr = Nd4j.create(output, new int[]{output.length, 1});

        double[] theta = {0.1, 0.2};
        INDArray thetaArr = Nd4j.create(theta, new int[]{2, 1});

        double result = linearRegression.computeCost(trainingSetArr, outputArr, thetaArr);

        assertEquals(11.9450, result, 0.1);

    }

    @Test
    public void computeCostTestCase2() {
        double[] training = {1, 2, 3, 1, 3, 4, 1, 4, 5, 1, 5, 6};
        INDArray trainingSetArr = Nd4j.create(training, new int[]{4, 3}, 'c');

        double[] output = {7, 6, 5, 4};
        INDArray outputArr = Nd4j.create(output, new int[]{output.length, 1});

        double[] theta = {0.1, 0.2, 0.3};
        INDArray thetaArr = Nd4j.create(theta, new int[]{3, 1});

        double result = linearRegression.computeCost(trainingSetArr, outputArr, thetaArr);

        assertEquals(7.0175, result, 0.1);

    }

    @Test
    public void computeCostTestCase3() {
        INDArray fromFile = FileUtils.getFromFile("/ex1data1.txt");

        INDArray trainingSetArr = fromFile.getColumn(0);
        INDArray ones = Nd4j.ones(trainingSetArr.rows(), 1);
        INDArray concat = Nd4j.concat(1, ones, trainingSetArr);

        INDArray outputArr = fromFile.getColumn(1);
        INDArray thetaArr = Nd4j.zeros(2, 1);

        double result = linearRegression.computeCost(concat, outputArr, thetaArr);

        assertEquals(32.07, result, 0.1);

    }
}

