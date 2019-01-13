package com.diy.ml.regression;

import static org.junit.Assert.assertEquals;
import org.junit.Test;

public class LinearRegressionTest {

    LinearRegression linearRegression = new LinearRegression();

    @Test
    public void testCostFunction() {

        long result = linearRegression.computeCostFunction();
        assertEquals(0, result);
    }
}
