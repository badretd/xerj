use std::fmt;

#[derive(Debug, Clone)]
pub enum Error {
    Io(String),
    Format(String),
    Unsupported(String),
    Tokenizer(String),
    Input(String),
    Executor(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(s) => write!(f, "io: {s}"),
            Error::Format(s) => write!(f, "format: {s}"),
            Error::Unsupported(s) => write!(f, "unsupported: {s}"),
            Error::Tokenizer(s) => write!(f, "tokenizer: {s}"),
            Error::Input(s) => write!(f, "input: {s}"),
            Error::Executor(s) => write!(f, "executor: {s}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
