#[allow(dead_code)]
use crate::central::*;
use ndarray::prelude::*;
use std::collections::{HashMap, HashSet};

/// An enum used by the equation to know what backward propgation method to preform
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operation {
    /// No operation, this will not pass any gradient 
    Nop, 
    /// (Left Side Operator, Right Side Operator)
    Add(ValueKey, ValueKey),
    /// (Left Side Operator, Right Side Operator)
    Multiplication(ValueKey, ValueKey),
    /// (Array that is having the EXP operation being applied to each element)
    Exp(ValueKey),
    /// (Array that is having its elements raised to a power, array that holds that power) //TODO: Maybe turn this into ValueKey, float
    Pow(ValueKey, ValueKey), 
    /// (Left Side Operator, Right Side Operator)
    MatrixMultiplication(ValueKey, ValueKey), 
    /// (Array where each element will passed to the natural log function)
    Log10(ValueKey), 
    /// (Array that will be summed, the index of axis that will be summed across, if an array has dimension (3, 4, 1), 0 would sum the 3, if an array had dimensions (4, 5), 1 would sum the 5)
    Sum(ValueKey, usize), 
    /// (Base Array that the view maps to, The rows to select, the indices inside those arrays to select)
    View(ValueKey, ValueKey, ValueKey), 
    /// (The array where the mean value is found)
    Mean(ValueKey)
}

unsafe impl Send for Operation {}
unsafe impl Sync for Operation {}

/// A struct that holds all of the graph information
/// It is not instatied directly, and it instead is hidden from the user 
/// to allow a closer expereince to pytorch
pub struct Equation {
    values: HashMap<ValueKey, InternalValue>,
    value_count: usize,
}

impl Equation {
    /// Returns a new Equation, should never be called directly
    /// TODO: Hide this
    pub fn new() -> Equation {
        Equation {
            values: HashMap::new(),
            value_count: 0,
        }
    }


    #[allow(dead_code)]
    /// Goes through a zeros out the grad field of all InternalValues
    /// Should be called before a forward pass as grads are accumlated not set
    pub fn zero_grad(&mut self) {
        for (_, value) in self.values.iter_mut() {
            value.zero_grad();
        }
    }

    /// Will update the data field of each InternalValue its grad field * learning_rate
    /// Needs to be called after backwards is called, elsewise the grad will be 0
    pub fn update_grad(&mut self) {
        for (_, value) in self.values.iter_mut() {
            value.update_data(-0.01);
        }
    }

    /// Creates an InternalValue, that holds the data and the operation
    /// # Arugments
    /// 'data' - The Data stored in the InternalValue
    /// 'operation' - What type operation created this value. Will be Nop for basic creations, or Add 
    pub fn create_value(&mut self, data: ArrayD<f32>, operation: Operation) -> ValueKey {
        self.value_count += 1;
        let new_value_id = ValueKey::new(self.value_count);
        let new_value = InternalValue::new(data, operation);
        self.values.insert(new_value_id.clone(), new_value);
        return new_value_id;
    }

    /// Gets the Data that underlays a Value
    /// # Arguments
    /// 'value_key' - THe Key used to look up the array
    pub fn get_value(&self, value_key: ValueKey) -> ArrayD<f32> {
        return self.values[&value_key].data.clone();
    }
    /// Gets the Grad that is associated with the Value
    /// # Arguments
    /// 'value_key' - THe Key used to look up the Value
    pub fn get_grad(&self, value_key: ValueKey) -> ArrayD<f32> {
        return self.values[&value_key].grad.clone();
    }

    /// Preforms a backpropgation value for a single Value in the graph
    /// # Arguments
    /// 'value_key' - The Key for the value that is being back propogated
    fn backward_for_value(&mut self, value_key: ValueKey) {
        let op = self.values.get(&value_key).unwrap().operation;
        let out_data = self.values.get(&value_key).unwrap().data.clone();
        let out_grad = self.values.get(&value_key).unwrap().grad.clone();
        match op {
            Operation::Nop => {}
            Operation::Add(left_hand_side, right_hand_size) => {
                self.values.get_mut(&left_hand_side).unwrap().grad = &self.values.get_mut(&left_hand_side).unwrap().grad + out_grad.clone();
                self.values.get_mut(&right_hand_size).unwrap().grad = &self.values.get_mut(&right_hand_size).unwrap().grad + out_grad;
            }
            Operation::Multiplication(left_hand_side, right_hand_side) => {
                let left_hand_data = self.values.get(&left_hand_side).unwrap().data.clone();
                let right_hand_data = self.values.get(&right_hand_side).unwrap().data.clone();
                self.values.get_mut(&left_hand_side).unwrap().grad = &self.values.get_mut(&left_hand_side).unwrap().grad + right_hand_data * out_grad.clone();
                self.values.get_mut(&right_hand_side).unwrap().grad = &self.values.get_mut(&right_hand_side).unwrap().grad  + left_hand_data * out_grad;
            }
            Operation::Exp(base_value) => {
                self.values.get_mut(&base_value).unwrap().grad = &self.values.get_mut(&base_value).unwrap().grad +  out_grad * out_data;
            }
            Operation::Pow(base, power) => {
                let other_data = &self.values.get(&power).unwrap().data;
                let base_data = &self.values.get(&base).unwrap().data;
                let power = other_data[[0]] - 1.0;
                let grad_update = other_data * base_data.map(|x| x.powf(power)) * out_grad;
                self.values.get_mut(&base).unwrap().grad = &self.values.get_mut(&base).unwrap().grad + grad_update;
            }
            Operation::MatrixMultiplication(left_hand, right_hand) => {
                let out_data = out_data.into_dimensionality::<Ix2>().unwrap();
                let other = self.values.get(&right_hand).unwrap().data.t().into_dimensionality::<Ix2>().unwrap();
                self.values.get_mut(&left_hand).unwrap().grad =
                    out_data.clone().dot(&other).into_dyn();
                let other = self.values.get(&left_hand).unwrap().data.t().into_dimensionality::<Ix2>().unwrap();
                self.values.get_mut(&right_hand).unwrap().grad =
                    out_data.dot(&other).into_dyn();


            }
            Operation::Log10(base) => {
                let grad = self.values.get(&base).unwrap().data.map(|x| 1.0 / x);
                self.values.get_mut(&base).unwrap().grad = &self.values.get_mut(&base).unwrap().grad + (grad * out_grad);
            }
            Operation::Sum(origin, _index) => {
                let shape: &[usize] = self.values.get(&origin).unwrap().grad.shape();
                self.values.get_mut(&origin).unwrap().grad = &self.values.get(&origin).unwrap().grad + &out_grad.broadcast(shape).unwrap();
            }
            Operation::Mean(origin) => {
                let value = &self.values.get(&origin).unwrap().data;
                // this should be getting back the batch size. As we want to evenlly distribute the loss over all the values that we have 
                // takent he mean off
                let amount = -1.0 / (value.len() as f32);
                self.values.get_mut(&origin).unwrap().grad = &self.values.get(&origin).unwrap().grad + (ArrayD::<f32>::ones(value.shape()) * amount);
            }
            Operation::View(value, x_indices, y_indices) => {
                let total_iterator : Vec<((f32, f32), f32)>  = {
                    let xs = self.values.get(&x_indices).unwrap();
                    let ys = self.values.get(&y_indices).unwrap();
                    let zipped_iter = xs.data.iter().zip(ys.data.iter()).map(|(x, y)|(*x, *y)).zip(out_grad.iter().map(|d|*d));

                    zipped_iter.collect()
                };

                let origin = self.values.get_mut(&value).unwrap();

                for ((index_x, index_y), data) in total_iterator {
                    origin.grad[[index_x as usize, index_y as usize]] = data;
                }
            }
        }
    }

    /// A function used by the backwards function to determine the flow of grad through the graph
    /// # Arugments
    /// 'node' - The Node of the graph we are starting from
    /// 'visited' - The set of Nodes we have seen already, so we don't loop
    /// 'stack' - The stack of nodes we might consider 
    fn topological_sort_util(
        &self,
        node: ValueKey,
        visited: &mut HashSet<ValueKey>,
        stack: &mut Vec<ValueKey>,
    ) {
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

    /// When called by a user with a provided value, preforms one step of backprogation
    /// BUT does not zero grad, or update the data 
    /// 'starting_value' - which value in the graph to back propogate from
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
