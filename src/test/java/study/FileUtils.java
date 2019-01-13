package study;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;

public class FileUtils {

    private static String makeResourcePath(String template) {
        return FileUtils.class.getResource(template).getPath();
    }

    public static INDArray getFromFile(String source) {
        INDArray readFromText = null;
        try {
            readFromText = Nd4j.readNumpy(makeResourcePath(source), ",");
        } catch (IOException e) {
            e.printStackTrace();
        }
        return readFromText;
    }
}
