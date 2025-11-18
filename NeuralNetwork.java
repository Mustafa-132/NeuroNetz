import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {

    private List<Layer> layers;
    private int inputSize;

    public NeuralNetwork(int inputSize) {
        layers = new ArrayList<>();
        this.inputSize = inputSize;
    }

    public void addLayer(int numNeurons, ActivationFunction activation){

        int numInputs = layers.isEmpty() ? inputSize : layers.get(layers.size() - 1).getNeurons().length;

        layers.add(new Layer(numNeurons, numInputs, activation));
    }

    public double[] forward(double[] inputs){

        double[] outputs = inputs;

        for(int i = 0; i < layers.size(); i++){
            outputs = layers.get(i).output(outputs);
        }

        return outputs;

    }

    public List<Layer> getLayers() {
        return layers;
    }

    public double[] computeDifference(double[] yOut, double[] yTarget){

        double[] diff = new double[yOut.length];

        for(int i = 0; i < yOut.length; i++){
            diff[i] = yTarget[i] - yOut[i];
        }

        return diff;

    }

}