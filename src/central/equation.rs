#[allow(dead_code)]
use crate::central::*;
use ndarray::prelude::*;
use std::collections::{HashMap, HashSet};

/// An enum used by the equation to know what backward propgation method to preform
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
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
    Mean(ValueKey),
    Broadcasting(ValueKey, [usize;2])
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
        self.get_actual_data(value_key)
    }
    /// Gets the Grad that is associated with the Value
    /// # Arguments
    /// 'value_key' - THe Key used to look up the Value
    pub fn get_grad(&self, value_key: ValueKey) -> ArrayD<f32> {
        return self.values[&value_key].get_grad().clone();
    }

    /// Preforms a backpropgation value for a single Value in the graph
    /// # Arguments
    /// 'value_key' - The Key for the value that is being back propogated
    fn backward_for_value(&mut self, value_key: ValueKey) {
        let op = self.values.get(&value_key).unwrap().operation;
        let out_data = self.get_actual_data(value_key);
        let out_grad = self.get_actual_grad(value_key);
        match op {
            Operation::Nop => {}
            Operation::Add(left_hand_side, right_hand_side) => {
                let left_hand_grad = self.get_actual_grad(left_hand_side);
                let right_hand_grad = self.get_actual_grad(right_hand_side);

                self.values.get_mut(&left_hand_side).unwrap().set_grad(left_hand_grad + out_grad.clone());
                self.values.get_mut(&right_hand_side).unwrap().set_grad(right_hand_grad + out_grad);
            }
            Operation::Multiplication(left_hand_side, right_hand_side) => {
                let left_hand_grad = self.get_actual_grad(left_hand_side);
                let right_hand_grad = self.get_actual_grad(right_hand_side);

                let left_hand_data = self.get_actual_data(left_hand_side);
                let right_hand_data = self.get_actual_data(right_hand_side);

                let new_left_side_grad = left_hand_grad + right_hand_data * out_grad.clone();
                let new_right_side_grad = right_hand_grad  + left_hand_data * out_grad;

                self.values.get_mut(&left_hand_side).unwrap().set_grad(new_left_side_grad);
                self.values.get_mut(&right_hand_side).unwrap().set_grad(new_right_side_grad);
            }
            Operation::Exp(base_value) => {
                let grad = self.get_actual_grad(base_value);
                self.values.get_mut(&base_value).unwrap().set_grad(grad +  out_grad * out_data);
            }
            Operation::Pow(base, power) => {
                
                let other_data = self.get_actual_data(power);
                let base_data = self.get_actual_data(base);
 
                let power = other_data[[0]] - 1.0;
                let grad_update = other_data * base_data.map(|x| x.powf(power)) * out_grad;

                let grad = self.get_actual_grad(base);

                self.values.get_mut(&base).unwrap().set_grad(grad + grad_update);
            }
            Operation::MatrixMultiplication(left_hand, right_hand) => {
                let out_data = out_grad.into_dimensionality::<Ix2>().unwrap();
                let right_hand_data =  self.get_actual_data(right_hand);
                let right_hand_data_tranpose  = right_hand_data.t();

                let other = right_hand_data_tranpose.into_dimensionality::<Ix2>().unwrap();
                let left_hand_data = self.get_actual_grad(left_hand);
                self.values.get_mut(&left_hand).unwrap().set_grad(left_hand_data + out_data.clone().dot(&other).into_dyn());

                let left_hand_data = self.get_actual_data(left_hand);
                let right_hand_grad = self.get_actual_grad(right_hand);
                
                let other = left_hand_data.t().into_dimensionality::<Ix2>().unwrap();
                let temp = right_hand_grad + other.dot(&out_data).into_dyn();
                self.values.get_mut(&right_hand).unwrap().set_grad(temp);
            }
            Operation::Log10(base) => {
                let data = self.get_actual_data(base);
                // We need to add EPISILON, so that way we will never divide by 0
                let grad = data.map(|x| 1.0 / (x +std::f32::EPSILON));
                let existing_grad = self.get_actual_grad(base);
                self.values.get_mut(&base).unwrap().set_grad(existing_grad + (grad * out_grad));
            }
            Operation::Sum(origin, _index) => {
                let origin_grad = self.get_actual_grad(origin);
                let shape: &[usize] = origin_grad.shape();
                self.values.get_mut(&origin).unwrap().set_grad(&origin_grad + &out_grad.broadcast(shape).unwrap());
            }
            Operation::Mean(origin) => {
                let value =self.get_actual_data(origin);
                let grad = self.get_actual_grad(origin);
                // this should be getting back the batch size. As we want to evenlly distribute the loss over all the values that we have 
                // takent he mean off
                let amount = -1.0 / (value.len() as f32);
                println!("{:?}", amount);
                self.values.get_mut(&origin).unwrap().set_grad(&grad + (ArrayD::<f32>::ones(grad.shape()) * amount));
            }
            Operation::View(value, x_indices, y_indices) => {
                let total_iterator : Vec<((f32, f32), f32)>  = {
                    let xs = self.get_actual_data(x_indices);
                    let ys = self.get_actual_data(y_indices);
                    let zipped_iter = xs.iter().zip(ys.iter()).map(|(x, y)|(*x, *y)).zip(out_grad.iter().map(|d|*d));

                    zipped_iter.collect()
                };

                let mut origin = self.values.get_mut(&value).unwrap().get_grad();

                for ((index_x, index_y), data) in total_iterator {
                    //println!("{:?} {:?} {:?}", index_x, index_y, data);
                    origin[[index_x as usize, index_y as usize]] = data;
                }
                self.values.get_mut(&value).unwrap().set_grad(origin);
            },
            Operation::Broadcasting(value, shape) => {
                // Broadcasting needs to sum up along the axis that it was broascasted over
                let old_shape = self.get_actual_data(value);
                if old_shape.shape()[0] != shape[0] {
                    let new = out_grad.sum_axis(Axis(0));
                    let existing_grad = self.get_actual_grad(value);
                    self.values.get_mut(&value).unwrap().set_grad(existing_grad + new);
                }
                else if old_shape.shape()[1] != shape[1] {
                    let new = out_grad.sum_axis(Axis(1)).into_shape(vec![old_shape.shape()[0], 1]).unwrap();
                    let existing_grad = self.get_actual_grad(value);
                    self.values.get_mut(&value).unwrap().set_grad(existing_grad + new);
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
        let ones = ArrayD::ones(self.values.get(&starting_value).unwrap().get_grad().shape());
        self.values.get_mut(&starting_value).unwrap().set_grad(ones);

        // Process nodes in topologically sorted order
        while let Some(node) = stack.pop() {
            let _ = self.backward_for_value(node); // Assuming this calculates and returns the children of the node
        }
    }

    pub fn set_requires_grad(&mut self, value: ValueKey, requires_grad: bool) {
        self.values.get_mut(&value).unwrap().requires_grad = requires_grad;
    }

    fn get_actual_data(&self, value_key: ValueKey) -> ArrayD<f32> {
        let value = self.values.get(&value_key).unwrap();
        match value.operation {
            Operation::View(a, b, c) => {
                let source_data = self.values.get(&a).unwrap().get_data();
                let x_inputs = self.values.get(&b).unwrap().get_data();
                let y_inputs = self.values.get(&c).unwrap().get_data();

                let zipped_iter = x_inputs.iter().zip(y_inputs.iter());

                let mut data_vec = vec![];
                for (x, y) in zipped_iter {
                    data_vec.push(source_data[[*x as usize, *y as usize]]);
                }
                let zeores = ArrayD::from_shape_vec(vec![x_inputs.shape()[0]], data_vec).unwrap();              
                return zeores;
            },
            _ => {
                return value.get_data();
            }
        }
    }


    fn get_actual_grad(&self, value_key: ValueKey) -> ArrayD<f32> {
        let value = self.values.get(&value_key).unwrap();
        match value.operation {
            Operation::View(a, b, c) => {
                return value.get_grad();
                let source_data = self.values.get(&a).unwrap().get_grad();
                let x_inputs = self.values.get(&b).unwrap().get_grad();
                let y_inputs = self.values.get(&c).unwrap().get_grad();

                let zipped_iter = x_inputs.iter().zip(y_inputs.iter());

                let mut data_vec = vec![];
                for (x, y) in zipped_iter {
                    data_vec.push(source_data[[*x as usize, *y as usize]]);
                }
                let zeores = ArrayD::from_shape_vec(vec![x_inputs.shape()[0]], data_vec).unwrap();              
                return zeores;
            },
            _ => {
                return value.get_grad();
            }
        }
    }
}
