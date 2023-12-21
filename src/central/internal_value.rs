use crate::central::*;
use nalgebra::*;


pub struct InternalValue {
    pub data: DMatrix<f32>,
    pub grad: DMatrix<f32>,
    pub operation: Operation
}

impl InternalValue {
    pub fn new(data: DMatrix<f32>, operation: Operation) -> InternalValue {
        InternalValue {
            data:data.clone(),
            grad: DMatrix::zeros(data.nrows(), data.ncols()),
            operation
        }
    }

    #[allow(dead_code)]
    pub fn update_data(&mut self, learning_rate: f32) {
        self.data += self.grad.clone() * learning_rate;
    }

    pub fn zero_grad(&mut self) {
        self.grad = DMatrix::zeros(self.data.nrows(), self.data.ncols());
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
            Operation::View(a) => {
                return vec![a];
            }
        }
    }
}