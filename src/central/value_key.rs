


#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug)]
pub struct ValueKey {
    key: usize
}

impl ValueKey {
    pub fn new(key: usize) -> ValueKey {
        ValueKey {
            key
        }
    }
}
