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

lazy_static! {
    pub static ref SINGLETON_INSTANCE: Mutex<Equation> = Mutex::new(Equation::new());
}
