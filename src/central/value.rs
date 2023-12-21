use std:: ops::{Add, Mul, Div, Neg, Sub};

use nalgebra::DMatrix;

use crate::central::*;
#[derive(Hash, PartialEq, Eq, Copy, Clone)]
pub struct Value {
    value: ValueKey
}

impl Value {

    pub fn new(data: DMatrix<f32>) -> Value {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let value_key = singleton.create_value(data, Operation::Nop);
        Value {
            value: value_key
        }
    }

    fn new_from_op(data: DMatrix<f32>, operation: Operation) -> Value {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        let value_key = singleton.create_value(data, operation);
        Value {
            value: value_key
        }
    }

    pub fn data(&self) -> DMatrix<f32> {
        let singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        return singleton.get_value(self.value);
    }

    pub fn grad(&self) -> DMatrix<f32> {
        let singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        return singleton.get_grad(self.value);
    }

    pub fn backward(&self) {
        let mut singleton = crate::central::SINGLETON_INSTANCE.lock().unwrap();
        singleton.backward(self.value);
    }

    pub fn exp(&self) -> Value {
        return Value::new_from_op(self.data().exp(), Operation::Exp(self.value));
    }

    pub fn tanh(&self) -> Value {
        let hold = Value::new(DMatrix::from_element(1, 1, 2.0));
        let k = *self * hold;
        let e = k.exp();
        let o = (e - 1.0) / (e + 1.0);
        return o;
    }

    pub fn pow(&self, other: DMatrix<f32>) -> Value {
        let power = other[0];
        let holding_value = Value::new(other);
        return Value::new_from_op(self.data().map(|x|x.powf(power)), Operation::Pow(self.value, holding_value.value));
    }

    pub fn matrix_mul(&self, value: Value) -> Value {
        let result = self.data().mul(value.data());
        return Value::new_from_op(result, Operation::MatrixMultiplication(self.value, value.value));
    }

    pub fn log(&self) -> Value  {
        let result = self.data().map(|x|x.log(10.0));
        return Value::new_from_op(result, Operation::Log10(self.value));
    }

    // Only getting this to work for a the case 1, N matrices, the bigger case is ... just harder
    pub fn view(&self, index: usize) -> Value {
        let shape = self.data().shape();
        if shape.0 != 1 {
            panic!("First dimenion does not equal 1, does not work for that at the moment");
        }
        let mut mask = DMatrix::from_element(shape.0, shape.1, 0.0f32);
        *mask.get_mut((0, index)).unwrap() = 1.0;    
        return Value::new_from_op(mask, Operation::View(self.value));
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
        let as_matrix = DMatrix::from_element(1, 1, other);
        let holding_value = Value::new(as_matrix);
        return Value::new_from_op(self.data() + holding_value.data(), Operation::Add(self.value, holding_value.value));
    }
}

impl Mul<f32> for Value {
    type Output = Self;

    fn mul(self, other: f32) -> Self::Output {
        let holding_value = Value::new(DMatrix::from_element(1, 1, other));
        return Value::new_from_op(self.data() * other, Operation::Multiplication(self.value, holding_value.value));
    }
}

impl Add<Value> for f32 {
    type Output = Value;

    fn add(self, other: Value) -> Value {
        let holding_value = Value::new(DMatrix::from_element(1, 1, self));
        return Value::new_from_op(holding_value.data() + other.data(), Operation::Add(holding_value.value, other.value));
    }
}

impl Mul<Value> for f32 {
    type Output = Value;

    fn mul(self, other: Value) -> Value {
        let holding_value = Value::new(DMatrix::from_element(1, 1, self));
        return Value::new_from_op(holding_value.data() * other.data(), Operation::Multiplication(holding_value.value, other.value));
    }
}

impl Div for Value {
    type Output = Self;
    fn div(self, rhs: Value) -> Self::Output {
        let intermediate = rhs.pow(DMatrix::from_element(1, 1, -1.0));
        return Value::new_from_op(self.data().component_mul(&intermediate.data()), Operation::Multiplication(self.value, intermediate.value));
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
        let holding_value = Value::new(DMatrix::from_element(1, 1, rhs));
        let intermeditate = -holding_value;
        return Value::new_from_op(self.data() + intermeditate.data(), Operation::Add(self.value, intermeditate.value));
    }
}