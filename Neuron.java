public class Neuron {

    private double[] weights;
    private double bias;
    private ActivationFunction activation;

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

        double sum = bias;

        for(int i = 0; i < inputs.length; i++){
            sum += inputs[i] * weights[i];
        }

        return activation.activate(sum);

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