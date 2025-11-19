public class Step implements ActivationFunction {

    private double step;

    public Step(){
        this(0.0);
    }

    public Step(double step){
        this.step = step;
    }

    @Override
    public double activate (double x){
        return x >= step ? 1.0 : 0.0;
    }

    @Override
    public double derivative (double x){
        return 0.0;
    }

    public double getStep(){
        return step;
    }

    public void setStep(double step){
        this.step = step;
    }

}