public class Layer {

    private Neuron[] neurons;

    public Layer(int numNeurons, int numInputsPerNeuron, ActivationFunction activation){

        neurons = new Neuron[numNeurons];

        for(int i = 0; i < numNeurons; i++){
            neurons[i] = new Neuron(numInputsPerNeuron, activation);
        }

    }

    public Neuron[] getNeurons(){
        return neurons;
    }

    public Neuron getNeuron(int index){
        return neurons[index];
    }

    public void setNeuron(int index, Neuron neuron){
        neurons[index] = neuron;
    }

    public double[] output(double[] inputs){

        double[] outputs = new double[neurons.length];

        if (inputs.length != neurons[0].getWeights().length){
            System.out.println("Error: inputs.length != neurons[0].getWeights().length");
            return outputs;
        }

        for(int i = 0; i < neurons.length; i++){
            outputs[i] = neurons[i].output(inputs);
        }

        return outputs;

    }

    public void computeOutputDelta(double[] target){

        for (int i = 0; i < neurons.length; i++){

            double out = neurons[i].lastOutput;
            double deriv = neurons[i].getActivation().derivative(neurons[i].lastSum);
            neurons[i].delta = deriv * (target[i] - out);

        }

    }

    //Backpropagation
    public void computeHiddenDelta(Layer next){

        for (int i = 0; i < neurons.length; i++){

            double sum = 0.0;
            for (Neuron neuron : next.getNeurons()){
                sum += neuron.delta * neuron.getWeights()[i];
            }
            double deriv = neurons[i].getActivation().derivative(neurons[i].lastSum);
            neurons[i].delta = deriv * sum;

        }

    }

}