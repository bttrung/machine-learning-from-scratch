package com.diy.ml.regression;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;


public class LinearRegressionTest {

    LinearRegression linearRegression = new LinearRegression();

    @Test
    public void studyNd4j() {

        //https://deeplearning4j.org/docs/latest/nd4j-overview#inmemory

        INDArray ones = Nd4j.ones(2, 3);
        System.out.printf(ones.toString());

        System.out.printf("\n----------\n");
        INDArray zeros = Nd4j.zeros(2, 3);
        System.out.printf(zeros.toString());

        System.out.printf("\n---adding 10 to each value----------\n");
        INDArray addi = zeros.addi(10);
        System.out.printf(addi.toString());

        // Random Arrays
        System.out.printf("\n---random arrays----------\n");
//        Nd4j.getRandom().setSeed(1000);
        INDArray rand = Nd4j.rand(3, 2);
        System.out.printf(rand.toString());


        // Creating NDArrays from Java arrays
        // row vector
        double[] doubles = {1, 2, 3};
        INDArray doublesND = Nd4j.create(doubles);
        System.out.printf("\n---row vector----------\n");
        System.out.printf(doublesND.toString());


        // column vector
        float[] floats = {1, 2, 3};
        int[] ints = new int[]{floats.length, 1};
        INDArray colVec = Nd4j.create(floats, ints);
        System.out.printf("\n\n---column vector----------\n");
        System.out.printf(colVec.toString());

        // 2nd arrays
        float[][] floats2d = new float[][]{
                {2, 3, 1},
                {4, 5, 6}
        };
        INDArray array2D = Nd4j.create(floats2d);
        System.out.printf("\n\n---2d arrays----------\n");
        System.out.printf(array2D.toString());

        // from shape 2x4
        double[] doubles1 = {1, 2, 3, 4, 5, 6, 7, 8};// 2x4 = 8 flat elements
        int[] shape = {2, 4};// create array 2x4 dimension
        INDArray fromShapeCOrder = Nd4j.create(doubles1, shape, 'c');
        System.out.printf("\n\n---2d fromShapeC order----------\n");
        System.out.printf(fromShapeCOrder.toString() + "\n");
        System.out.printf(fromShapeCOrder.shapeInfoToString());

        INDArray fromShapeFOrder = Nd4j.create(doubles1, shape, 'f');
        System.out.printf("\n\n---2d fromShape F order----------\n");
        System.out.printf(fromShapeFOrder.toString() + "\n");
        System.out.printf(fromShapeFOrder.shapeInfoToString());

        // create array from others
        //Creating an exact copy of an existing NDArray
        System.out.printf("\n\n---duplicated array----------\n");
        INDArray dup = array2D.dup();
        System.out.printf(dup.toString());

        //Creating a subset array
        INDArray hstack = Nd4j.hstack(ones, zeros);
        System.out.printf("\n\n---hstack----------\n");
        System.out.printf(hstack.toString());

        INDArray vstack = Nd4j.vstack(ones, zeros);
        System.out.printf("\n\n---vstack----------\n");
        System.out.printf(vstack.toString());

        // transport
        INDArray transpose = zeros.transpose();
        System.out.printf("\n\n---before transport----------\n");
        System.out.printf(zeros.toString());
        System.out.printf("\n\n---after transport----------\n");
        System.out.printf(transpose.toString());
        System.out.printf(transpose.shapeInfoToString());

        // ND4J.concat: combines arrays along a dimension

    }

    @Test
    public void computeCostTest() {
        long result = linearRegression.computeCost(Nd4j.ones(2, 2));
        assertEquals(0, result);
    }
}
