use std:: ops::{Add, Mul, Div, Neg, Sub};

use ndarray::prelude::*;
 
use crate::central::*;


#[derive(Hash, PartialEq, Eq, Copy, Clone)]
/// A user facing handle for a tensor value
pub struct Value {
    /// What opertion created this value, it will be Operation::Nop for values created without an operation
    operation: Operation,
    /// A key for the InternalValue stored by the equation itself
    pub value: ValueKey
}

impl Value {
    /// Returns a new value, creating an InternalValue in the Equation
    /// # Arguments
    /// * 'data' - A Dynamic Array that holds the data stored in the Equation
    /// 
    /// # Examples
    /// ```
    /// // This will create a value in the equation of 5 wide and 5 high, set all to one
    /// 
    /// use forge_ai::Value;
    /// use ndarray::ArrayD;
    /// let value = Value::new(ArrayD::ones(vec![5, 5]));
    /// ```
    pub fn new(data: ArrayD<f32>) -> Value {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let value_key = singleton.create_value(data, Operation::Nop);
        Value {
            value: value_key,
            operation: Operation::Nop
        }
    }

    fn new_from_op(data: ArrayD<f32>, operation: Operation) -> Value {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let value_key = singleton.create_value(data, operation);
        Value {
            value: value_key,
            operation
        }
    }


    /**
    A function that returns a Value that holds an array of size [1, end]
    filled with values between [0, end)
    # Arguments

    * 'end' - The end of the range, not included

    # Examples
    ```
    // This will return an array with values between 0 and 10
    use forge_ai::Value;
    let value = Value::arange(10);

    ```
    */
    pub fn arange(end: usize) -> Value {
        let data = ArrayD::from_shape_vec(vec![end], (0..end).collect()).unwrap().mapv(|x|f32::from(x as u8));
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let value_key = singleton.create_value(data, Operation::Nop);
        Value {
            value: value_key,
            operation: Operation::Nop
        }
    }


    /// Function that returns a copy of the undelaying data currently stored in the array
    /// Works differently for View Values 
    /// # Examples
    /// ```
    /// // This will return an array with values between 0 and 10
    /// use forge_ai::Value;
    /// let value = Value::arange(10);
    /// 
    /// ```
    pub fn data(&self) -> ArrayD<f32> {
        match self.operation {
            // A view operation does not actually store the values that the view in based around
            Operation::View(a, b, c) => {
                let singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();

                // All the Value node holds is the source array, and the indices
                // Then we copy them over to a fresh array and return it
                let source_data = singleton.get_value(a);
                let x_inputs = singleton.get_value(b);
                let y_inputs = singleton.get_value(c);

                let zipped_iter = x_inputs.iter().zip(y_inputs.iter());

                let mut data_vec = vec![];
                for (x, y) in zipped_iter {
                    data_vec.push(source_data[[*x as usize, *y as usize]]);
                }
                let zeores = ArrayD::from_shape_vec(vec![x_inputs.shape()[0]], data_vec).unwrap();              
                return zeores;
            },
            _ => {
                // Return a copy of the data in the array, exspensive
                let singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
                return singleton.get_value(self.value);
            }
        }
    }

    /// Returns the Grad for the Node
    pub fn grad(&self) -> ArrayD<f32> {
        let singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        return singleton.get_grad(self.value);
    }

    /// Propogates the value of this node into all of its childern nodes, it will clear the grad of all
    /// values when called
    /// 
    /// # Examples
    /// 
    /// ```
    /// use forge_ai::Value;
    /// use ndarray::ArrayD;
    /// let a = Value::new(ArrayD::from_elem(vec![1], 1.0));
    /// let b = Value::new(ArrayD::from_elem(vec![1], 2.0));
    /// let result = a + b;
    /// result.backward();
    /// assert!(result.grad()[[0]] == 1.0);
    /// assert!(a.grad()[[0]] == 1.0);
    /// assert!(b.grad()[[0]] == 1.0);
    /// ```
    pub fn backward(&self) {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        singleton.backward(self.value);
    }

    /// Returns a new node with the values of this node with the exp function called on them
    pub fn exp(&self) -> Value {
        return Value::new_from_op(self.data().map(|x|x.exp()), Operation::Exp(self.value));
    }

    /// Returns a new node with values of the node with the tanh function called on them
    pub fn tanh(&self) -> Value {
        let hold = Value::new(ArrayD::from_elem(vec![1], 2.0));
        let k = *self * hold;
        let e = k.exp();
        let o = (e - 1.0) / (e + 1.0);
        return o;
    }

    /// Returns a new node with the values of this node raised to the provided power
    /// # Arugments
    /// * 'other' - An ArrayD that contains the value
    pub fn pow(&self, other: ArrayD<f32>) -> Value {
        let holding_value = Value::new(other.clone());

        let power = other[0];
        return Value::new_from_op(self.data().map(|x|x.powf(power)), Operation::Pow(self.value, holding_value.value));
    }

    /// Returns a new Value that is the result of a matrix multiplication of this node, and the provided node
    /// # Arguments
    /// * 'value' - The other value too be multiplied by
    pub fn matrix_mul(&self, value: Value) -> Value {
        // TODO: Make sure that the two matrices are able to multiplied with each other

        let data_as_2d = self.data().into_dimensionality::<Ix2>().unwrap();
        let other_as_2d = value.data().into_dimensionality::<Ix2>().unwrap();
        let result_array = data_as_2d.dot(&other_as_2d);
        return Value::new_from_op(result_array.into_dyn(), Operation::MatrixMultiplication(self.value, value.value));
    }

    /// Returns a new value that contains all of the values of this node passed to the natural log function
    pub fn log(&self) -> Value  {
        let mut value = self.data();
        value.par_map_inplace(|x|*x = x.ln());
        return Value::new_from_op(value, Operation::Log10(self.value));
    }

    /// Returns a value that acts as a placeholder for the idea of a view 
    /// This is a really special function that is trying to replicated some 
    /// behaviors of the way torch works
    /// As I want to be able to copy a line from the zero to hero lecture
    /// probs[torch.arange(num), ys] = -1/n
    /// Now both torch.arange(num) and ys are 2d vectors, where one dim is 1
    /// And it acts first by picking out the rows by torch.arange, and then selecting
    /// into them with ys
    /// this is not really doable with Ndarray
    /// so this works by taking it two 1d arrays, and then when .data() is called on it
    /// running through the pairs of values, and copying them into a new array, not perfect
    pub fn view(&self, xs: Value, ys: Value) -> Value {
        return Value::new_from_op(ArrayD::from_elem(vec![1], 0.0), Operation::View(self.value, xs.value, ys.value));
    }


    /// Sums up a value along the provided axis
    /// this will always remove the dimension, unlike the same function in
    /// torch which provides the options not to
    pub fn sum(&self, index: usize) -> Value {
        // TODO: provide the ability to say that you want to keep the dimension that is removed when this sum happens
        // The inverse of this is a broadcast operatsion
        let result = self.data().sum_axis(Axis(index));
        return Value::new_from_op(result, Operation::Sum(self.value, index));
    }

    /// Returns the mean of the array behind the value
    pub fn mean(&self) -> Value {
        let result = self.data().mean().unwrap();
        return Value::new_from_op(ArrayD::from_elem(vec![1], result), Operation::Mean(self.value));
    }

    pub fn set_requires_grad(&self, required_grad: bool) {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        singleton.set_requires_grad(self.value, required_grad);
    }
}

impl Add for Value {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        return Value::new_from_op(self.data() + rhs.data(), Operation::Add(self.value, rhs.value));
    }
}

impl Mul for Value {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        return Value::new_from_op(self.data() * rhs.data(), Operation::Multiplication(self.value, rhs.value));
    }
}

impl Add<f32> for Value {
    type Output = Value;

    fn add(self, other: f32) -> Value {
        let as_matrix = Value::new(ArrayD::from_elem(vec![1], other));
        return Value::new_from_op(self.data() + as_matrix.data(), Operation::Add(self.value, as_matrix.value));
    }
}

impl Mul<f32> for Value {
    type Output = Self;

    fn mul(self, other: f32) -> Self::Output {
        let holding_value = Value::new(ArrayD::from_elem(vec![1], other));
        return Value::new_from_op(self.data() * other, Operation::Multiplication(self.value, holding_value.value));
    }
}

impl Add<Value> for f32 {
    type Output = Value;

    fn add(self, other: Value) -> Value {
        let holding_value = Value::new(ArrayD::from_elem(vec![1], self));
        return Value::new_from_op(holding_value.data() + other.data(), Operation::Add(holding_value.value, other.value));
    }
}

impl Mul<Value> for f32 {
    type Output = Value;

    fn mul(self, other: Value) -> Value {
        let holding_value = Value::new(ArrayD::from_elem(vec![1], self));
        return Value::new_from_op(holding_value.data() * other.data(), Operation::Multiplication(holding_value.value, other.value));
    }
}

impl Div for Value {
    type Output = Self;
    fn div(self, rhs: Value) -> Self::Output {
        let intermediate = rhs.pow(ArrayD::from_elem(vec![1], -1.0));

        return Value::new_from_op(self.data().mul(&intermediate.data()), Operation::Multiplication(self.value, intermediate.value));
    }
}
 

impl Neg for Value {
    type Output = Self;
    fn neg(self) -> Self::Output {
        return self * -1.0
    }
}

impl Sub<f32> for Value {
    type Output = Self;
    fn sub(self, rhs: f32) -> Self::Output {
        let holding_value = Value::new(ArrayD::from_elem(vec![1], rhs));
        let intermeditate = -holding_value;
        return Value::new_from_op(self.data() + intermeditate.data(), Operation::Add(self.value, intermeditate.value));
    }
}