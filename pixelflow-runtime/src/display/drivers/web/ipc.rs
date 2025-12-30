use crate::display::DisplayEvent;
use crate::error::RuntimeError;
use js_sys::{Atomics, Int32Array, SharedArrayBuffer, Uint8Array};

// Layout in SharedArrayBuffer:
// Offset 0 (Indices[0]): Write Index (in bytes, relative to Data Start)
// Offset 4 (Indices[1]): Read Index (in bytes, relative to Data Start)
// Offset 8: Data Start
const IDX_WRITE: u32 = 0;
const IDX_READ: u32 = 1;
const HEADER_BYTES: u32 = 8;

pub struct SharedRingBuffer {
    indices: Int32Array,
    data: Uint8Array,
    capacity: u32,
}

impl SharedRingBuffer {
    pub fn new(sab: &SharedArrayBuffer) -> Self {
        let indices = Int32Array::new(sab);
        let data = Uint8Array::new(sab);
        let capacity = data.length() - HEADER_BYTES;
        Self {
            indices,
            data,
            capacity,
        }
    }

    fn read_idx(&self) -> i32 {
        Atomics::load(&self.indices, IDX_READ).unwrap_or(0)
    }

    fn write_idx(&self) -> i32 {
        Atomics::load(&self.indices, IDX_WRITE).unwrap_or(0)
    }

    /// Block until an event is available or timeout occurs.
    /// Timeout is in milliseconds.
    pub fn blocking_read_timeout(
        &self,
        timeout_ms: i32,
    ) -> Result<Option<DisplayEvent>, RuntimeError> {
        loop {
            let read_pos = self.read_idx();
            let write_pos = self.write_idx();

            if read_pos == write_pos {
                // Buffer is empty. Wait for notification.
                match Atomics::wait_with_timeout(
                    &self.indices,
                    IDX_WRITE,
                    write_pos,
                    timeout_ms as f64,
                ) {
                    Ok(val) => {
                        // val is "ok", "not-equal", or "timed-out"
                        if val.as_string().as_deref() == Some("timed-out") {
                            return Ok(None);
                        }
                        // If ok or not-equal, continue loop to check indices
                        continue;
                    }
                    Err(e) => return Err(RuntimeError::AtomicsError(format!("{:?}", e))),
                }
            }

            // Data available. Read frame length (4 bytes).
            // We need to handle wrapping.
            let mut len_bytes = [0u8; 4];
            self.read_bytes(read_pos as u32, &mut len_bytes);
            let len = u32::from_le_bytes(len_bytes);

            // Read payload
            let mut payload = vec![0u8; len as usize];
            self.read_bytes((read_pos as u32 + 4) % self.capacity, &mut payload);

            // Update Read Index
            // New pos = (read_pos + 4 + len) % capacity
            // Note: We need to align to 4 bytes? No, byte level is fine for data.
            // But strict alignment helps. Let's assume byte alignment.
            let next_read_pos = (read_pos + 4 + len as i32) % (self.capacity as i32);
            Atomics::store(&self.indices, IDX_READ, next_read_pos)
                .map_err(|e| RuntimeError::AtomicsStoreError(format!("{:?}", e)))?;

            // Notify writer (if it was waiting for space) - optional for unbounded writer?
            // Better to notify.
            Atomics::notify(&self.indices, IDX_READ)
                .map_err(|e| RuntimeError::AtomicsNotifyError(format!("{:?}", e)))?;

            let event: DisplayEvent = bincode::deserialize(&payload)
                .map_err(|e| RuntimeError::InitError(format!("Bincode error: {}", e)))?;
            return Ok(Some(event));
        }
    }

    fn read_bytes(&self, start_offset: u32, buf: &mut [u8]) {
        let cap = self.capacity;
        let start = start_offset % cap;
        let mut current = start;
        for b in buf.iter_mut() {
            // data view includes header, so add HEADER_BYTES
            let idx = current + HEADER_BYTES;
            *b = self.data.get_index(idx);
            current = (current + 1) % cap;
        }
    }

    // Called by Main Thread (Writer)
    pub fn write(&self, event: &DisplayEvent) -> Result<(), RuntimeError> {
        let payload = bincode::serialize(event)
            .map_err(|e| RuntimeError::InitError(format!("Bincode error: {}", e)))?;
        let len = payload.len() as u32;
        let total_len = 4 + len;

        // Space check loop
        loop {
            let read_pos = self.read_idx();
            let write_pos = self.write_idx();

            // Calculate free space
            // If read == write, empty (full capacity)
            // If write > read, free = capacity - (write - read)
            // If read > write, free = read - write
            // We reserve 1 byte to distinguish full vs empty?
            // Or just use counts. Byte indices: we need 1 byte gap.

            let free_space = if write_pos >= read_pos {
                self.capacity as i32 - (write_pos - read_pos)
            } else {
                read_pos - write_pos
            };

            // We need total_len + 1 (gap)
            if free_space < (total_len as i32 + 1) {
                // Full. Wait for space?
                // Main thread (UI) cannot block! Atomics.wait throws on main thread.
                // So we must return error or drop event.
                // Or spin? Spinning on main thread is bad.
                // We'll return Error::WouldBlock or simply error.
                return Err(RuntimeError::RingBufferFull);
            }

            // Write Length
            let len_bytes = len.to_le_bytes();
            self.write_bytes(write_pos as u32, &len_bytes);

            // Write Payload
            self.write_bytes((write_pos as u32 + 4) % self.capacity, &payload);

            // Update Write Index
            let next_write_pos = (write_pos + 4 + len as i32) % (self.capacity as i32);
            Atomics::store(&self.indices, IDX_WRITE, next_write_pos)
                .map_err(|e| RuntimeError::AtomicsStoreError(format!("{:?}", e)))?;

            // Notify Reader (Worker)
            Atomics::notify(&self.indices, IDX_WRITE)
                .map_err(|e| RuntimeError::AtomicsNotifyError(format!("{:?}", e)))?;

            return Ok(());
        }
    }

    fn write_bytes(&self, start_offset: u32, buf: &[u8]) {
        let cap = self.capacity;
        let start = start_offset % cap;
        let mut current = start;
        for &b in buf.iter() {
            let idx = current + HEADER_BYTES;
            self.data.set_index(idx, b);
            current = (current + 1) % cap;
        }
    }
}
