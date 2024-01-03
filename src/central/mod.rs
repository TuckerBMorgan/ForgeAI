use lazy_static::lazy_static;

use std::sync::Mutex;

mod equation;
mod value;
mod value_key;
mod internal_value;

pub use value::*;
pub use value_key::*;
pub use internal_value::*;

pub use equation::*;


// This SINGELTON_INSTANCE lets gives us the ability 
// to have a global reference to a single "equation" without the need
// to pass around one to each Value we create
// This is very not in the style of rust, and was made this way mostly because
// I wanted to move fast and make something similar to the python code 
// that I was copying
// Big drawback, this is a forced atomic opeartion, so if two parts of code
// try to take it at the same time they can deadlock
lazy_static! {
    pub static ref SINGLETON_INSTANCE: Mutex<Equation> = Mutex::new(Equation::new());
}
