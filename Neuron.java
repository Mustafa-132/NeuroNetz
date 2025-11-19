public class Neuron {

    private double[] weights;
    private double bias;
    private ActivationFunction activation;

    public double[] lastInputs;
    public double lastSum;
    public double lastOutput;
    public double delta;

    public Neuron(int numInputs, ActivationFunction activation){

        this.weights = new double[numInputs];
        this.bias = 0.0;
        this.activation = activation;
        initializeWeights();

    }

    private void initializeWeights(){

        for(int i = 0; i < weights.length; i++){
            weights[i] = Math.random() - 0.5;
        }

    }

    public double output(double[] inputs){

        lastInputs = inputs;
        double sum = bias;

        if (inputs.length != weights.length){
            System.out.println("Error: inputs.length != weights.length");
            return 0.0;
        }

        for(int i = 0; i < inputs.length; i++){
            sum += inputs[i] * weights[i];
        }

        lastSum = sum;
        lastOutput = activation.activate(sum);

        return lastOutput;

    }

    public double[] getWeights(){
        return weights;
    }

    public void setWeights(double[] newWeights){

        if (weights.length == newWeights.length)
            weights = newWeights;
        else
            System.out.println("Error: weights array length does not match");
    }

    //Backpropagation
    public void updateWeights(double learningRate) {

        for (int i = 0; i < weights.length; i++){
            weights[i] += learningRate * delta * lastInputs[i];
        }
        bias += learningRate * delta;

    }

    public double getBias(){
        return bias;
    }

    public void setBias(double bias){
        this.bias = bias;
    }

    public ActivationFunction getActivation(){
        return activation;
    }

    public void setActivation(ActivationFunction activation){
        this.activation = activation;
    }
    
}