pub mod epoll;
pub mod pty;

#[cfg(test)]
mod pty_tests;

#[derive(Debug, Clone)]
pub enum PtyActionCommand {
    Write(Vec<u8>),
    Resize { cols: u16, rows: u16 },
}
