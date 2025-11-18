import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class NeuralNetworkTest {

    @Test
    public void test1() {

        NeuralNetwork net = new NeuralNetwork(3);
        net.addLayer(1, new Step(1.0));

        double[] inputs = {0.2, 0.8, 0.6};
        double[] weights = {0.5, 0.2, 0.6};

        net.getLayers().get(0).getNeuron(0).setWeights(weights);
        net.getLayers().get(0).getNeuron(0).setBias(0.1);

        double[] output = net.forward(inputs);

        assertEquals(0.0, output[0], 1e-9);

    }

    @Test
    public void test2() {

        NeuralNetwork net = new NeuralNetwork(2);
        net.addLayer(2, new Identity());
        net.addLayer(2, new Identity());

        double[] inputs = {-10, 10};
        double[] weights11 = {-0.5, -0.4};
        double[] weights12 = {-0.2, -0.1};
        double[] weights21 = {0.1, 0.2};
        double[] weights22 = {0.4, 0.5};

        net.getLayers().get(0).getNeuron(0).setWeights(weights11);
        net.getLayers().get(0).getNeuron(1).setWeights(weights12);
        net.getLayers().get(0).getNeuron(0).setBias(-0.3);
        net.getLayers().get(0).getNeuron(1).setBias(0.1);
        net.getLayers().get(1).getNeuron(0).setWeights(weights21);
        net.getLayers().get(1).getNeuron(1).setWeights(weights22);
        net.getLayers().get(1).getNeuron(0).setBias(0.3);
        net.getLayers().get(1).getNeuron(1).setBias(0.6);

        double[] output = net.forward(inputs);

        double e11 = (-10) * (-0.5) + 10 * (-0.4) - 0.3;
        double e12 = (-10) * (-0.2) + 10 * (-0.1) + 0.1;
        double e21 = e11 * 0.1 + e12 * 0.2 + 0.3;
        double e22 = e11 * 0.4 + e12 * 0.5 + 0.6;

        assertEquals(e21, output[0], 1e-9);
        assertEquals(e22, output[1], 1e-9);

    }
    
}
