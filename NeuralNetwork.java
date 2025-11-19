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

        for (int i = 0; i < layers.size(); i++){
            outputs = layers.get(i).output(outputs);
        }

        return outputs;

    }

    //Backpropagation
    public void backpropagate(double[] target, double learningRate){

        Layer outputLayer = layers.get(layers.size() - 1);
        outputLayer.computeOutputDelta(target);

        for (int i = layers.size() - 2; i >= 0; i--) {
            Layer current = layers.get(i);
            Layer next = layers.get(i + 1);

            current.computeHiddenDelta(next);
        }

        for (Layer layer : layers) {
            for (Neuron n : layer.getNeurons()) {
                n.updateWeights(learningRate);
            }
        }

    }

    //Backpropagation
    public void train(double[][] inputs, double[][] targets, double eta, int epochs, double minError){

        double error = 1000000000;

        for (int e = 0; e < epochs; e++) {
            double totalError = 0.0;
            for (int i = 0; i < inputs.length; i++) {
                double[] output = forward(inputs[i]);
                double sampleError = 0.0;
                for (int j = 0; j < output.length; j++) {
                    sampleError += Math.pow(targets[i][j] - output[j], 2);
                }
                sampleError /= output.length;
                System.out.println("Sample error: " + sampleError);
                backpropagate(targets[i], eta);
                totalError += sampleError;
                System.out.println("Total error: " + totalError);
            }
            double epochError = totalError / inputs.length;
            //if (epochError < error) eta *= 1.1;
            //else eta *= 0.5;

            System.out.println("Epoch error: " + epochError);
            if (epochError <= minError) break;
            error = epochError;
        }
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