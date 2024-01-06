
use crate::central::*;
use ndarray::ArrayD;

/// The struct that represents a Node in our graph
pub struct InternalValue {
    /// What data is in this node
    data: ArrayD<f32>,
    /// What the gradiante for this node is, starts at zero
    /// is always the same shape as data
    grad: ArrayD<f32>,
    /// What operation created this value, will be NOP for basic creation
    pub operation: Operation,

    pub requires_grad: bool
}

impl InternalValue {
    pub fn new(data: ArrayD<f32>, operation: Operation) -> InternalValue {
        InternalValue {
            data:data.clone(),
            grad: ArrayD::zeros(data.shape()),
            operation,
            requires_grad: false
        }
    }

    pub fn set_data(&mut self, data: ArrayD<f32>) {
        if self.data.shape() != data.shape() {
            panic!("Tried setting wrong sized data aarrays, must be equal right {:?} wrong {:?}", self.data.shape(), data.shape());
        }
        self.data = data;
    }

    pub fn get_data(&self) -> ArrayD<f32> {
        self.data.clone()
    }

    pub fn set_grad(&mut self, grad: ArrayD<f32>) {
        if self.grad.shape() != grad.shape() {
            panic!("Tried setting wrong grad sized arrays, must be equal right {:?} wrong {:?}", self.grad.shape(), grad.shape());
        }
        self.grad = grad;
    }

    pub fn get_grad(&self) -> ArrayD<f32> {
        self.grad.clone()
    }

    #[allow(dead_code)]
    pub fn update_data(&mut self, learning_rate: f32) {
        if self.requires_grad == false {
            self.data = &self.data + (&self.grad * learning_rate);
        }
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
            Operation::View(a, _b, _c) => {
                // We ignore B and C, as they are just the arrays that hold the indices
                // that make up the view, and as such we don't need to preform
                // backprop on them
                return vec![a];
            },
            Operation::Sum(a, _index) => {
                return  vec![a];
            },
            Operation::Mean(a) => {
                return vec![a];
            },
            Operation::Broadcasting(a, _b) => {
                return vec![a];
            }
        }
    }
}