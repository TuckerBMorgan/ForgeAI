mod central;

pub use central::Value;

#[cfg(test)]
pub mod tests {

    use rand::Rng;

    use ndarray::prelude::*;

    pub use crate::central::Value;

    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn mean_test() {
        let a = Value::new(ArrayD::ones(vec![2,2]));
        let b = Value::new(ArrayD::ones(vec![2,2]));
        let c = a + b;
        // Rust and Python elvaulate - and functions if a different order
        // the () are needed
        let loss = -(c.log().mean());
        loss.backward();
        // Values taken from a pytorch file
        assert!(approx_equal(a.grad()[[0, 0]], -0.125, 0.005));
    }

    #[test] 
    fn log_test() {
        let a = Value::new(ArrayD::ones(vec![1]));
        let b = Value::new(ArrayD::ones(vec![1]));
        let c = a + b;
        let d = c.log();
        d.backward();
        assert!(approx_equal(a.grad()[[0]], 0.5, 0.005));
    }

    #[test]
    fn basic() {
        let a = Value::new(ArrayD::zeros(vec![1, 1]));
        let b = Value::new(ArrayD::from_elem(vec![1, 1], 1.0));
        assert!((a + b).data()[[0, 0]] == 1.0);
    }

    #[test]
    fn basic_backward() {
        let a = Value::new(ArrayD::from_elem(vec![1, 1], 1.0));
        let b = Value::new(ArrayD::from_elem(vec![1, 1], 2.0));
        let result = a + b;

        result.backward();
        assert!(result.grad()[[0, 0]] == 1.0);
        assert!(a.grad()[[0, 0]] == 1.0);
        assert!(b.grad()[[0, 0]] == 1.0);
    }

    #[test]
    fn basic_multiplication() {
        let a = Value::new(ArrayD::from_elem(vec![1, 1], 2.0));
        let b = Value::new(ArrayD::from_elem(vec![1, 1], 5.0));
        assert!((a * b).data()[[0, 0]] == 10.0);
    }

    #[test]
    fn basic_multiplication_backward() {
        let a = Value::new(ArrayD::from_elem(vec![1, 1], 2.0));
        let b = Value::new(ArrayD::from_elem(vec![1, 1], 5.0));
        let result = a * b;

        result.backward();
        assert!(result.data()[[0,0]] == 10.0);
        assert!(result.grad()[[0, 0]] == 1.0);
        assert!(a.grad()[[0, 0]] == 5.0);
        assert!(b.grad()[[0, 0]] == 2.0);
    }

    #[test]
    fn basic_pow() {
        let a = Value::new(ArrayD::from_elem(vec![1], 2.0));
        
        let b = a.pow(ArrayD::from_elem(vec![1], 3.0));
        b.backward();
        assert!(b.data()[[0]] == 8.0);
        assert!(a.grad()[[0]] == 12.0);
         
    }

    #[test]
    fn basic_exp() {
        let a = Value::new(ArrayD::from_elem(vec![1, 1], 1.0));
        let c = a.exp();
        c.backward();
    }

    #[test]
    fn chained_backward() {
        let a = Value::new(ArrayD::from_elem(vec![1], 2.0));
        let b = Value::new(ArrayD::from_elem(vec![1], 5.0));
        let c = a * b;
        let result = c + 2.0;
        result.backward();
        assert!(result.data()[[0]] == 12.0);
        assert!(result.grad()[[0]] == 1.0);
        assert!(c.data()[[0]] == 10.0);
        assert!(c.grad()[[0]] == 1.0);
        assert!(a.grad()[[0]] == 5.0);
        assert!(b.grad()[[0]] == 2.0);
    }

    #[test]
    fn basic_div_test() {
        let a = Value::new(ArrayD::from_elem(vec![1], 10.0));
        let b = Value::new(ArrayD::from_elem(vec![1], 5.0));
        let c = a / b;
        c.backward();
        assert!(c.data()[[0]] == 2.0);
        assert!(a.grad()[[0]] == 0.2);

        // b.grad comes out to roughly -0.39999 
        // we we look at the abs minus the expected value
        // and we just want to know that difference is small enough
        assert!(approx_equal(b.grad()[[0]], -0.4, 0.0001));
    }

    #[test] 
    fn basic_view_test() {
        let a = Value::new(ArrayD::ones(vec![20, 30]));
        let b = Value::arange(10);
        let c = Value::arange(15);
        let view = a.view(b, c);
        let d = Value::new(ArrayD::ones(vec![10]));
        let e = view + d;
        e.backward();
        println!("{:?}", a.grad());
        assert!(approx_equal(a.grad()[[0, 0]], 1.0, 0.001));
    }

    #[test]
    fn micrograd_copy_test() {
        let x1 = Value::new(ArrayD::from_elem(vec![1], 2.0));
        let x2 = Value::new(ArrayD::from_elem(vec![1], 0.0));

        let w1 = Value::new(ArrayD::from_elem(vec![1], -3.0));
        let w2 = Value::new(ArrayD::from_elem(vec![1], 1.0));

        let b = Value::new(ArrayD::from_elem(vec![1], 6.8813735870195432));

        let x1w1 = x1 * w1;
        let x2w2 = x2 * w2;

        let x1w1x2w2 = x1w1 + x2w2;
        let n = x1w1x2w2 + b;
        let l = 2.0f32 * n;
        let e = l.exp();
        let o = (e - 1.0) / (e + 1.0);
        o.backward();

        assert!(n.data()[[0]] == 0.8813734);
        assert!(approx_equal(n.grad()[[0]], 0.5, 1e-6));

        assert!(x1w1x2w2.data()[[0]] == -6.0);
        assert!(approx_equal(x1w1x2w2.grad()[[0]], 0.5, 1e-6));


        assert!(b.data()[[0]] == 6.8813735870195432);
        assert!(approx_equal(b.grad()[[0]], 0.5, 1e-6));


        assert!(x2w2.data()[[0]] == 0.0);
        assert!(approx_equal(x2w2.grad()[[0]], 0.5, 1e-6));

        assert!(x1w1.data()[[0]] == -6.0);
        assert!(approx_equal(x1w1.grad()[[0]], 0.5, 1e-6));

        assert!(w2.data()[[0]] == 1.0);
        assert!(w2.grad()[[0]] == 0.0);

        assert!(x2.data()[[0]]== 0.0);
        assert!(approx_equal(x2.grad()[[0]], 0.5, 1e-6));

        assert!(w1.data()[[0]] == -3.0);
        assert!(approx_equal(w1.grad()[[0]], 1.0, 1e-6));

        assert!(x1.data()[[0]] == 2.0);
        assert!(approx_equal(x1.grad()[[0]], -1.5, 1e-6));
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
                weights.push(Value::new(ArrayD::from_elem(vec![1], rng.gen_range(-1.0..1.0))));
            }

            Neuron {
                weights,
                bias: Value::new(ArrayD::from_elem(vec![1], rng.gen_range(-1.0..1.0)))    
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
            let mut values : Vec<Value> = input.iter().map(|x|Value::new(ArrayD::from_elem(vec![1], *x))).collect();
            
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
            let mut loss = Value::new(ArrayD::from_elem(vec![1], 0.0));
    
            for (a, b) in collected {
                let predicted_as_value = Value::new(ArrayD::from_elem(vec![1], *a));
                let difference = b[0] + -predicted_as_value;
                let power = difference.pow(ArrayD::from_elem(vec![1], 2.0));
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

    use core::f32;
    use std::fs::read_to_string;
    use std::collections::{HashMap, HashSet};

    fn read_lines(filename: &str) -> Vec<String> {
        let mut result = Vec::new();

        for line in read_to_string(filename).unwrap().lines() {
            result.push(line.to_string())
        }

        result
    }

    fn genereate_dataset() -> (HashMap<char, usize>, HashMap<usize, char>, Vec<usize>, Vec<usize>) {
        let names = read_lines("./data/bigram/names.txt");

        let chars: HashSet<char> = names.iter()
        .flat_map(|word| word.chars())
        .collect();    
        let mut chars_vec: Vec<char> = chars.into_iter().collect();


        chars_vec.sort_unstable();

        let mut stoi: HashMap<char, usize> = HashMap::new();
        for (i, &c) in chars_vec.iter().enumerate() {
            stoi.insert(c, i + 1);
        }
        stoi.insert('.', 0);


        let itos: HashMap<usize, char> = stoi.iter()
            .map(|(&c, &i)| (i, c))
            .collect();

            let mut xs: Vec<usize> = vec![];
            let mut ys = vec![];

            // I BUILT MY DATASET WRONG
            // Less tired, and with more time on your hand, this will be easy to solve
            for name in names.iter() {
                let fixed = String::from(".") + &name + ".";
                let chars: Vec<char> = fixed.chars().collect();
                for i in 0..chars.len() - 1 {
                    let pair = (chars[i], chars[i + 1]);
                    xs.push(stoi[&pair.0]);
                    ys.push(stoi[&pair.1]);
                }
            }

            let var_name: (HashMap<char, usize>, HashMap<usize, char>, Vec<usize>, Vec<usize>) = (stoi, itos, xs, ys);
            return var_name;
    }

    #[test]
    fn bigram_test() {

        // Set up constants
        const NUMBER_OF_CHARACTERS : usize = 27;
        const TEST_BATCH : usize = 1;


        // Setup the weights of the network
        let mut rng = rand::thread_rng();
        let weights = Value::new(ArrayD::from_shape_fn(vec![27, 27], |_x|{rng.gen_range(-1.0..1.0)}));

        weights.set_requires_grad(true);
        // Generate out our dataset
        let (stoi, itos, xs, ys) = genereate_dataset();
        // Prepare a batch of data
        let combined = xs.iter().take(TEST_BATCH).zip(ys.iter().take(TEST_BATCH));
        let mut inputs = ArrayD::zeros(vec![TEST_BATCH, NUMBER_OF_CHARACTERS]);
        let mut outputs = ArrayD::zeros(vec![TEST_BATCH]);
        for (index, (input, output)) in combined.enumerate() {
            inputs[[index, *input]] = 1.0f32;
            outputs[[index]] = *output as f32;
        }

        let inputs = Value::new(inputs);
        const EPOCHS : usize = 1;

        for _ in 0..EPOCHS {
            {
                let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
                singleton.zero_grad();
            }

            // logits = xenc @ W
            let logits = inputs.matrix_mul(weights);

            // counts = logits.exot()
            let counts = logits.exp();

            // probs = counts / counts.sum(1, keepdims=True)
            let counts_sum = counts.sum(1, true);
            let counts_cum_inv = counts_sum.pow(ArrayD::from_elem(vec![1], -1.0));
            let counts_data = counts.data();
            let shape = counts_data.shape();
            let counts_cum_inv_broadcasted = counts_cum_inv.broadcast([shape[0], shape[1]]);
            let probs = counts * counts_cum_inv_broadcasted;
            
            // loss = -probs[torach.arange(num), ys].log().mean()
            let xs = Value::arange(TEST_BATCH);
            let value = Value::new(outputs.clone());
            let views = probs.view(xs, value);
            let logged = -(views.log().mean());

            println!("{:?}", logged.data());
            logged.backward();
            {
                let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
                singleton.update_grad();
            }
        }

            let logits = inputs.matrix_mul(weights);
            let counts = logits.exp();
            let counts_sum = counts.sum(1, true);
            /*
            Check broadcasting

            doube check work from last night
 */
            let counts_cum_inv = counts_sum.pow(ArrayD::from_elem(vec![1], -1.0));
            let counts_data = counts.data();
            let shape = counts_data.shape();
            let counts_cum_inv_broadcasted = counts_cum_inv.broadcast([shape[0], shape[1]]);
            let probs = counts * counts_cum_inv_broadcasted;
            println!("{:?}", probs.data());

    }

}