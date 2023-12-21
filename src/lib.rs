mod central;


#[cfg(test)]
pub mod tests {

    use nalgebra::DMatrix;
    use rand::Rng;

    pub use crate::central::Value;

    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn basic() {
        let a = Value::new(DMatrix::zeros(1, 1));
        let b = Value::new(DMatrix::from_element(1, 1, 1.0));
        assert!((a + b).data()[0] == 1.0);
    }

    #[test]
    fn basic_backward() {
        let a = Value::new(DMatrix::from_element(1, 1, 1.0));
        let b = Value::new(DMatrix::from_element(1, 1, 2.0));
        let result = a + b;

        result.backward();
        assert!(result.grad()[0] == 1.0);
        assert!(a.grad()[0] == 1.0);
        assert!(b.grad()[0] == 1.0);
    }

    #[test]
    fn basic_multiplication() {
        let a = Value::new(DMatrix::from_element(1, 1, 2.0));
        let b = Value::new(DMatrix::from_element(1, 1, 5.0));
        assert!((a * b).data()[0] == 10.0);
    }

    #[test]
    fn basic_multiplication_backward() {
        let a = Value::new(DMatrix::from_element(1, 1, 2.0));
        let b = Value::new(DMatrix::from_element(1, 1, 5.0));
        let result = a * b;

        result.backward();
        assert!(result.data()[0] == 10.0);
        assert!(result.grad()[0] == 1.0);
        assert!(a.grad()[0] == 5.0);
        assert!(b.grad()[0] == 2.0);
    }

    #[test]
    fn basic_pow() {
        let a = Value::new(DMatrix::from_element(1, 1, 2.0));
        let b = a.pow(DMatrix::from_element(1, 1, 3.0));
        b.backward();
        assert!(b.data()[0] == 8.0);
        assert!(a.grad()[0] == 12.0);
    }

    #[test]
    fn basic_exp() {
        let a = Value::new(DMatrix::from_element(1, 1, 1.0));
        let c = a.exp();
        c.backward();
    }

    #[test]
    fn chained_backward() {
        let a = Value::new(DMatrix::from_element(1, 1, 2.0));
        let b = Value::new(DMatrix::from_element(1, 1, 5.0));
        let c = a * b;
        let result = c + 2.0;
        result.backward();
        assert!(result.data()[0] == 12.0);
        assert!(result.grad()[0] == 1.0);
        assert!(c.data()[0] == 10.0);
        assert!(c.grad()[0] == 1.0);
        assert!(a.grad()[0] == 5.0);
        assert!(b.grad()[0] == 2.0);
    }

    #[test]
    fn basic_div_test() {
        let a = Value::new(DMatrix::from_element(1, 1, 10.0));
        let b = Value::new(DMatrix::from_element(1, 1, 5.0));
        let c = a / b;
        c.backward();
        assert!(c.data()[0] == 2.0);
        assert!(a.grad()[0] == 0.2);

        // b.grad comes out to roughly -0.39999 
        // we we look at the abs minus the expected value
        // and we just want to know that difference is small enough
        assert!(approx_equal(b.grad()[0], -0.4, 0.0001));
    }

    #[test]
    fn micrograd_copy_test() {
        let x1 = Value::new(DMatrix::from_element(1, 1, 2.0));
        let x2 = Value::new(DMatrix::from_element(1, 1, 0.0));

        let w1 = Value::new(DMatrix::from_element(1, 1, -3.0));
        let w2 = Value::new(DMatrix::from_element(1, 1, 1.0));

        let b = Value::new(DMatrix::from_element(1, 1, 6.8813735870195432));

        let x1w1 = x1 * w1;
        let x2w2 = x2 * w2;

        let x1w1x2w2 = x1w1 + x2w2;
        let n = x1w1x2w2 + b;
        let l = 2.0f32 * n;
        let e = l.exp();
        let o = (e - 1.0) / (e + 1.0);
        o.backward();

        assert!(n.data()[0] == 0.8813734);
        assert!(approx_equal(n.grad()[0], 0.5, 1e-6));

        assert!(x1w1x2w2.data()[0] == -6.0);
        assert!(approx_equal(x1w1x2w2.grad()[0], 0.5, 1e-6));


        assert!(b.data()[0] == 6.8813735870195432);
        assert!(approx_equal(b.grad()[0], 0.5, 1e-6));


        assert!(x2w2.data()[0] == 0.0);
        assert!(approx_equal(x2w2.grad()[0], 0.5, 1e-6));

        assert!(x1w1.data()[0] == -6.0);
        assert!(approx_equal(x1w1.grad()[0], 0.5, 1e-6));

        assert!(w2.data()[0] == 1.0);
        assert!(w2.grad()[0] == 0.0);

        assert!(x2.data()[0]== 0.0);
        assert!(approx_equal(x2.grad()[0], 0.5, 1e-6));

        assert!(w1.data()[0] == -3.0);
        assert!(approx_equal(w1.grad()[0], 1.0, 1e-6));

        assert!(x1.data()[0] == 2.0);
        assert!(approx_equal(x1.grad()[0], -1.5, 1e-6));
    }


    pub struct Neuron {
        pub weights: Vec<Value>,
        pub bias: Value
    }

    impl Neuron {
        pub fn new(number_of_inputs: usize) -> Neuron {
            let mut rng = rand::thread_rng();
            let mut weights = vec![];
            for _ in 0..number_of_inputs {
                weights.push(Value::new(DMatrix::from_element(1, 1, rng.gen_range(-1.0..1.0))));
            }

            Neuron {
                weights,
                bias: Value::new(DMatrix::from_element(1, 1, rng.gen_range(-1.0..1.0)))    
            }
        }

        
        pub fn call(&self, inputs: &Vec<Value>) -> Value {
            let result = self.weights.iter().zip(inputs.iter()).map(|(a, b)| *a * *b);

            let mut new_summed_bias = self.bias;
            for r in result {
                new_summed_bias = new_summed_bias + r;
            }
            // Calculate w * x + b and apply the hyperbolic tangent
            return new_summed_bias.tanh();
        }

    }
    pub struct Layer {
        neurons: Vec<Neuron>
    }

    impl Layer {
        pub fn new(number_of_inputs: usize, number_of_outputs: usize) -> Layer {
            let mut neurons = vec![];
            for _ in 0..number_of_outputs {
                neurons.push(Neuron::new(number_of_inputs));
            }
            Layer {
                neurons
            }
        }

        pub fn call(&self, inputs: &Vec<Value>) -> Vec<Value> {
            let mut outputs = vec![];
            for n in self.neurons.iter() {
                outputs.push(n.call(inputs));
            }

            return outputs;
        }
    }

    pub struct MLP {
        layers: Vec<Layer>
    }

    impl MLP {
        pub fn new(number_of_inputs: usize,mut neurons_per_layers: Vec<usize> ) -> MLP {
            let mut sizes = vec![number_of_inputs];
            sizes.append(&mut neurons_per_layers);

            let mut layers = vec![];
            for a in sizes.chunks(2) {
                layers.push(Layer::new(a[0], a[1]));
            }
            MLP {
                layers
            }
        }

        pub fn call(&self, input: &Vec<f32>) -> Vec<Value>  {
            let mut values : Vec<Value> = input.iter().map(|x|Value::new(DMatrix::from_element(1, 1, *x))).collect();
            
            for layer in self.layers.iter() {
                values = layer.call(&values);
            }
            return values;
        }
    }


    #[test]
    fn mlp_test() {
        let mlp = MLP::new(3, vec![4, 4, 1]);
        let inputs = vec![
            vec![2.0f32, 3.0, -1.0],
            vec![3.0, -1.0, 0.5],
            vec![0.5, 1.0, 1.0],
            vec![1.0, 1.0, -1.0],
        ];

        let outputs = vec![1.0f32, -1.0, -1.0, 1.0];
        for _ in 0..50 {
            {
                let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
                singleton.zero_grad();
            }
            let mut predicted_ouputs = vec![];
            for input in inputs.iter() {
                predicted_ouputs.push(mlp.call(input));
            }
    
            let collected = outputs.iter().zip(predicted_ouputs.iter());
            let mut loss = Value::new(DMatrix::from_element(1, 1, 0.0));
    
            for (a, b) in collected {
                let predicted_as_value = Value::new(DMatrix::from_element(1, 1, *a));
                // TOOD: Write a proper Value - Value function
                let difference = b[0] + -predicted_as_value;
                let power = difference.pow(DMatrix::from_element(1, 1, 2.0));
                loss = loss + power;
            }
            println!("{:?}", loss.data());
            loss.backward();

            {
                let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
                singleton.update_grad();
            }
    
        }
    }


}