use std:: ops::{Add, Mul, Div, Neg, Sub};

use ndarray::prelude::*;
use ndarray::parallel::prelude::*;
 
use crate::central::*;
#[derive(Hash, PartialEq, Eq, Copy, Clone)]
pub struct Value {
    operation: Operation,
    value: ValueKey
}

impl Value {

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

    pub fn arange(end: usize) -> Value {
        let data = ArrayD::from_shape_vec(vec![end], (0..end).collect()).unwrap().mapv(|x|f32::from(x as u8));
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let value_key = singleton.create_value(data, Operation::Nop);
        Value {
            value: value_key,
            operation: Operation::Nop
        }
    }

    pub fn data(&self) -> ArrayD<f32> {
        match self.operation {
            Operation::View(a, b, c) => {
                let singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();

                let source_data = singleton.get_value(a);
                let x_inputs = singleton.get_value(b);
                let y_inputs = singleton.get_value(c);

                let zipped_iter = x_inputs.iter().zip(y_inputs.iter());
                let mut zeores = ArrayD::zeros(vec![x_inputs.len(), y_inputs.shape()[1]]);
                for (x, y) in zipped_iter {
                    zeores[[*x as usize, *y as usize]] = source_data[[*x as usize, *y as usize]];
                }
                return zeores;
            },
            _ => {
                let singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
                return singleton.get_value(self.value);        
            }
        }
    }

    pub fn grad(&self) -> ArrayD<f32> {
        let singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        return singleton.get_grad(self.value);
    }

    pub fn backward(&self) {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        singleton.backward(self.value);
    }

    pub fn exp(&self) -> Value {
        return Value::new_from_op(self.data().map(|x|x.exp()), Operation::Exp(self.value));
    }

    pub fn tanh(&self) -> Value {
        let hold = Value::new(ArrayD::from_elem(vec![1, 1], 2.0));
        let k = *self * hold;
        let e = k.exp();
        let o = (e - 1.0) / (e + 1.0);
        return o;
    }

    pub fn pow(&self, other: ArrayD<f32>) -> Value {
        let holding_value = Value::new(other.clone());

        let power = other[[0, 0]];
        return Value::new_from_op(self.data().map(|x|x.powf(power)), Operation::Pow(self.value, holding_value.value));
    }

    pub fn matrix_mul(&self, value: Value) -> Value {
        let result = self.data().mul(value.data());
        return Value::new_from_op(result, Operation::MatrixMultiplication(self.value, value.value));
    }

    pub fn log(&self) -> Value  {
        let mut value = self.data();
        value.par_map_inplace(|x|*x = x.ln());
        return Value::new_from_op(value, Operation::Log10(self.value));
    }

    pub fn view(&self, xs: Value, ys: Value) -> Value {

        return Value::new_from_op(ArrayD::from_elem(vec![1, 1], 0.0), Operation::View(self.value, xs.value, ys.value));
    }

    pub fn sum(&self, index: usize) -> Value {
        // The inverse of this is a broadcast operatsion
        let result = self.data().sum_axis(Axis(index));
        return Value::new_from_op(result, Operation::Sum(self.value, index));
    }

    pub fn mean(&self) -> Value {
        let result = self.data().mean().unwrap();
        return Value::new_from_op(ArrayD::from_elem(vec![1, 1], result), Operation::Mean(self.value));
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
        let as_matrix = Value::new(ArrayD::from_elem(vec![1, 2], other));
        return Value::new_from_op(self.data() + as_matrix.data(), Operation::Add(self.value, as_matrix.value));
    }
}

impl Mul<f32> for Value {
    type Output = Self;

    fn mul(self, other: f32) -> Self::Output {
        let holding_value = Value::new(ArrayD::from_elem(vec![1, 2], other));
        return Value::new_from_op(self.data() * other, Operation::Multiplication(self.value, holding_value.value));
    }
}

impl Add<Value> for f32 {
    type Output = Value;

    fn add(self, other: Value) -> Value {
        let holding_value = Value::new(ArrayD::from_elem(vec![1, 2], self));
        return Value::new_from_op(holding_value.data() + other.data(), Operation::Add(holding_value.value, other.value));
    }
}

impl Mul<Value> for f32 {
    type Output = Value;

    fn mul(self, other: Value) -> Value {
        let holding_value = Value::new(ArrayD::from_elem(vec![1, 1], self));
        return Value::new_from_op(holding_value.data() * other.data(), Operation::Multiplication(holding_value.value, other.value));
    }
}

impl Div for Value {
    type Output = Self;
    fn div(self, rhs: Value) -> Self::Output {
        let intermediate = rhs.pow(ArrayD::from_elem(vec![1, 1], -1.0));

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
        let holding_value = Value::new(ArrayD::from_elem(vec![1, 1], rhs));
        let intermeditate = -holding_value;
        return Value::new_from_op(self.data() + intermeditate.data(), Operation::Add(self.value, intermeditate.value));
    }
}