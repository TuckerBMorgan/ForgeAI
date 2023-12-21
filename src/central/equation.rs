use std::collections::{HashMap, HashSet};
use nalgebra::DMatrix;
    #[allow(dead_code)]
use crate::central::*;

#[derive(Clone, Copy)]
pub enum Operation {
    Nop,
    Add(ValueKey, ValueKey),
    Multiplication(ValueKey, ValueKey),
    Exp(ValueKey),
    Pow(ValueKey, ValueKey),
    MatrixMultiplication(ValueKey, ValueKey),
    Log10(ValueKey),
    View(ValueKey)
}

unsafe impl Send for Operation {}
unsafe impl Sync for Operation {}


pub struct Equation {
    values: HashMap<ValueKey, InternalValue>,
    value_count: usize
}

impl Equation {
    pub fn new() -> Equation {
        Equation {
            values: HashMap::new(),
            value_count: 0
        }
    }

    #[allow(dead_code)]
    pub fn zero_grad(&mut self) {
        for (_, value) in self.values.iter_mut() {
            value.zero_grad();
        }
    }

    pub fn update_grad(&mut self) {
        for (_, value) in self.values.iter_mut() {
            value.update_data(-0.01);
        }
    }

    pub fn create_value(&mut self, data: DMatrix<f32>, operation: Operation) -> ValueKey {
        self.value_count += 1;
        let new_value_id = ValueKey::new(self.value_count);
        let new_value = InternalValue::new(data, operation);
        self.values.insert(new_value_id.clone(), new_value);
        return new_value_id; 
    }
    
    pub fn get_value(&self, value_key: ValueKey) -> DMatrix<f32> {
        return self.values[&value_key].data.clone();
    }

    pub fn get_grad(&self, value_key: ValueKey) -> DMatrix<f32> {
        return self.values[&value_key].grad.clone();
    }

    fn backward_for_value(&mut self, value_key: ValueKey) {
        let op = self.values.get(&value_key).unwrap().operation;
        let out_data = self.values.get(&value_key).unwrap().data.clone();
        let out_grad = self.values.get(&value_key).unwrap().grad.clone();
        match op {
            Operation::Nop => {
            },
            Operation::Add(left_hand_side, right_hand_size) => {
                self.values.get_mut(&left_hand_side).unwrap().grad += out_grad.clone();
                self.values.get_mut(&right_hand_size).unwrap().grad += out_grad;
            },
            Operation::Multiplication(left_hand_side, right_hand_side) => {
                let left_hand_data = self.values.get(&left_hand_side).unwrap().data.clone();
                let right_hand_data = self.values.get(&right_hand_side).unwrap().data.clone();
                self.values.get_mut(&left_hand_side).unwrap().grad += right_hand_data * out_grad.clone();
                self.values.get_mut(&right_hand_side).unwrap().grad += left_hand_data * out_grad;
            },
            Operation::Exp(base_value) => {
                self.values.get_mut(&base_value).unwrap().grad += out_grad * out_data;
            },
            Operation::Pow(base, power) => {
                let other_data = &self.values.get(&power).unwrap().data;
                let base_data = &self.values.get(&base).unwrap().data;
                let power = other_data[0] - 1.0;
                let grad_update = other_data * base_data.map(|x|x.powf(power)) * out_grad;
                self.values.get_mut(&base).unwrap().grad += grad_update;
            },
            Operation::MatrixMultiplication(left_hand,right_hand ) => {
                self.values.get_mut(&left_hand).unwrap().grad = out_data.clone() * self.values.get(&right_hand).unwrap().data.transpose();
                self.values.get_mut(&right_hand).unwrap().grad = out_data * self.values.get(&left_hand).unwrap().data.transpose();   
            },
            Operation::Log10(base) => {
                let grad = self.values.get(&base).unwrap().data.map(|x| 1.0 / x);
                self.values.get_mut(&base).unwrap().grad += grad;
            },
            Operation::View(value) => {
                //let element_wise = 
            },

        }
    }

    fn topological_sort_util(&self, node: ValueKey, visited: &mut HashSet<ValueKey>, stack: &mut Vec<ValueKey>) {
        visited.insert(node);

        // Assuming 'dependencies' method returns all the nodes that the current node depends on
        if let Some(dependencies) = self.values.get(&node).map(|n| n.dependencies()) {
            for dep in dependencies {
                if !visited.contains(&dep) {
                    self.topological_sort_util(dep, visited, stack);
                }
            }
        }

        stack.push(node);
    }

    pub fn backward(&mut self, starting_value: ValueKey) {
        // Initialize visited set and stack for topological sort
        let mut visited = HashSet::new();
        let mut stack = Vec::new();

        // Perform the topological sort
        self.topological_sort_util(starting_value, &mut visited, &mut stack);

        // Initialize gradients
        self.values.get_mut(&starting_value).unwrap().grad.fill(1.0);

        // Process nodes in topologically sorted order
        while let Some(node) = stack.pop() {
            let _ = self.backward_for_value(node); // Assuming this calculates and returns the children of the node
        }
    }
}
