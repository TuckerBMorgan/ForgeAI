use crate::central::*;
use ndarray::{ArrayD, Shape};

pub struct InternalValue {
    pub data: ArrayD<f32>,
    pub grad: ArrayD<f32>,
    pub operation: Operation
}

impl InternalValue {
    pub fn new(data: ArrayD<f32>, operation: Operation) -> InternalValue {
        InternalValue {
            data:data.clone(),
            grad: ArrayD::zeros(data.shape()),
            operation
        }
    }

    #[allow(dead_code)]
    pub fn update_data(&mut self, learning_rate: f32) {
        self.data = &self.data + (&self.grad * learning_rate);
    }

    pub fn zero_grad(&mut self) {
        self.grad = ArrayD::zeros(self.data.shape());
    }

    pub fn dependencies(&self) -> Vec<ValueKey> {
        match self.operation {
            Operation::Add(a, b) => {
                return vec![a, b];
            },
            Operation::Multiplication(a, b) => {
                return vec![a, b];
            },
            Operation::Exp(a) => {
                return vec![a];
            },
            Operation::Nop => {
                return vec![];
            },
            Operation::Pow(a, b) => {
                return vec![a, b];
            },
            Operation::MatrixMultiplication(a, b) => {
                return vec![a, b];
            },
            Operation::Log10(a) => {
                return vec![a];
            },
            Operation::View(a, b, c) => {
                return vec![a];
            },
            Operation::Sum(a, _index) => {
                return  vec![a];
            },
            Operation::Mean(a) => {
                return vec![a];
            }
        }
    }
}