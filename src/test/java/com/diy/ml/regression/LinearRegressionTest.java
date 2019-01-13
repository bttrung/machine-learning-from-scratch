package com.diy.ml.regression;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import study.FileUtils;

import static org.junit.Assert.assertEquals;


public class LinearRegressionTest {

    LinearRegression linearRegression = new LinearRegression();


    @Test
    public void computeCostSingleTestCase1() {
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
    public void computeCostSingleTestCase2() {
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
    public void computeCostSingleTestCase3() {
        INDArray fromFile = FileUtils.getFromFile("/ex1data1.txt");

        INDArray trainingSetArr = fromFile.getColumn(0);
        INDArray ones = Nd4j.ones(trainingSetArr.rows(), 1);
        INDArray concat = Nd4j.concat(1, ones, trainingSetArr);

        INDArray outputArr = fromFile.getColumn(1);
        INDArray thetaArr = Nd4j.zeros(2, 1);

        double result = linearRegression.computeCost(concat, outputArr, thetaArr);

        assertEquals(32.07, result, 0.1);
    }

    @Test
    public void computeCostMultiTestCase1() {
        double[] training = {2, 1, 3, 7, 1, 9, 1, 8, 1, 3, 7, 4};
        INDArray trainingSetArr = Nd4j.create(training, new int[]{4, 3}, 'c');

        double[] output = {2, 5, 5, 6};
        INDArray outputArr = Nd4j.create(output, new int[]{output.length, 1});

        double[] theta = {0.4, 0.6, 0.8};
        INDArray thetaArr = Nd4j.create(theta, new int[]{3, 1});

        double result = linearRegression.computeCost(trainingSetArr, outputArr, thetaArr);

        assertEquals(5.2950, result, 0.1);

    }

    @Test
    public void gradientDescentTestCase1() {

        double[] training = {1, 5, 1, 2, 1, 4, 1, 5};
        INDArray trainingSetArr = Nd4j.create(training, new int[]{4, 2}, 'c');

        double[] output = {1, 6, 4, 2};
        INDArray outputArr = Nd4j.create(output, new int[]{output.length, 1});

        double[] theta = {0, 0};
        INDArray thetaArr = Nd4j.create(theta, new int[]{2, 1});

        INDArray results = linearRegression.gradientDescent(trainingSetArr, outputArr, thetaArr, 0.01, 1000);

        assertEquals(1, results.columns());
        assertEquals(5.2148, results.getColumn(0).getDouble(0), 0.1);
        assertEquals(-0.5733, results.getColumn(0).getDouble(1), 0.1);
    }

    @Test
    public void gradientDescentTestCase2() {

        double[] training = {1, 5, 1, 2};
        INDArray trainingSetArr = Nd4j.create(training, new int[]{2, 2}, 'c');

        double[] output = {1, 6};
        INDArray outputArr = Nd4j.create(output, new int[]{output.length, 1});

        double[] theta = {0.5, 0.5};
        INDArray thetaArr = Nd4j.create(theta, new int[]{2, 1});

        INDArray results = linearRegression.gradientDescent(trainingSetArr, outputArr, thetaArr, 0.1, 10);

        assertEquals(1, results.columns());
        assertEquals(1.70986, results.getColumn(0).getDouble(0), 0.1);
        assertEquals(0.19229, results.getColumn(0).getDouble(1), 0.1);
    }

    @Test
    public void gradientDescentTestCase3() {
        INDArray fromFile = FileUtils.getFromFile("/ex1data1.txt");
        INDArray trainingSetArr = fromFile.getColumn(0);
        INDArray ones = Nd4j.ones(trainingSetArr.rows(), 1);
        INDArray concat = Nd4j.concat(1, ones, trainingSetArr);

        INDArray outputArr = fromFile.getColumn(1);
        INDArray thetaArr = Nd4j.zeros(2, 1);

        INDArray results = linearRegression.gradientDescent(concat, outputArr, thetaArr, 0.01, 1500);

        assertEquals(1, results.columns());
        assertEquals(-3.63029, results.getColumn(0).getDouble(0), 0.1);
        assertEquals(1.16636, results.getColumn(0).getDouble(1), 0.1);
    }

    @Test
    public void gradientDescentMultiTestCase1() {

        double[] training = {2, 1, 3, 7, 1, 9, 1, 8, 1, 3, 7, 4};
        INDArray trainingSetArr = Nd4j.create(training, new int[]{4, 3}, 'c');

        double[] output = {2, 5, 5, 6};
        INDArray outputArr = Nd4j.create(output, new int[]{output.length, 1});

        INDArray thetaArr = Nd4j.zeros(3, 1);
        INDArray results = linearRegression.gradientDescent(trainingSetArr, outputArr, thetaArr, 0.01, 100);

        assertEquals(1, results.columns());
        assertEquals(0.23680, results.getColumn(0).getDouble(0), 0.1);
        assertEquals(0.56524, results.getColumn(0).getDouble(1), 0.1);
        assertEquals(0.31248, results.getColumn(0).getDouble(2), 0.1);
    }

    @Test
    public void featureNormalizeTestCase1() {

        double[] input = {1, 2, 3};
        INDArray inputArray = Nd4j.create(input, new int[]{input.length, 1});
        INDArray normalizedArray = linearRegression.featureNormalize(inputArray);
        assertEquals(-1, normalizedArray.getDouble(0, 0), 0.1);
        assertEquals(0, normalizedArray.getDouble(1, 0), 0.1);
        assertEquals(1, normalizedArray.getDouble(2, 0), 0.1);

    }

    @Test
    public void featureNormalizeTestCase2() {
        double[] input = {8, 1, 6, 3, 5, 7, 4, 9, 2};
        INDArray inputArray = Nd4j.create(input, new int[]{3, 3});
        INDArray normalizedArray = linearRegression.featureNormalize(inputArray);
        assertEquals(1.13389, normalizedArray.getDouble(0, 0), 0.1);
        assertEquals(-0.75593, normalizedArray.getDouble(1, 0), 0.1);
        assertEquals(-0.37796, normalizedArray.getDouble(2, 0), 0.1);

        assertEquals(-1.00000, normalizedArray.getDouble(0, 1), 0.1);
        assertEquals(0.00000, normalizedArray.getDouble(1, 1), 0.1);
        assertEquals(1.00000, normalizedArray.getDouble(2, 1), 0.1);

        assertEquals(0.37796, normalizedArray.getDouble(0, 2), 0.1);
        assertEquals(0.75593, normalizedArray.getDouble(1, 2), 0.1);
        assertEquals(-1.13389, normalizedArray.getDouble(2, 2), 0.1);

    }

    @Test
    public void normalEquationTestCase1() {
        double[] training = {2, 1, 3, 7, 1, 9, 1, 8, 1, 3, 7, 4};
        INDArray trainingSetArr = Nd4j.create(training, new int[]{4, 3}, 'c');

        double[] output = {2, 5, 5, 6};
        INDArray outputArr = Nd4j.create(output, new int[]{output.length, 1});

        INDArray theta = linearRegression.normalEquation(trainingSetArr, outputArr);

        assertEquals(0.0083857, theta.getDouble(0, 0), 0.1);
        assertEquals(0.5681342, theta.getDouble(1, 0), 0.1);
        assertEquals(0.4863732, theta.getDouble(2, 0), 0.1);
    }
}

