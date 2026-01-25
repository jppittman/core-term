use pixelflow_core::{Field, Manifold, X, Y, W, vector::Vec4};
use wasm_bindgen::prelude::*;

struct PsychedelicScene {
    t: f32,
    width: f32,
    height: f32,
}

impl Manifold for PsychedelicScene {
    type Output = Vec4;

    fn eval_raw(&self, _x: Field, _y: Field, _z: Field, _w: Field) -> Self::Output {
        let scale = 2.0 / self.height;
        let half_width = self.width * 0.5;
        let half_height = self.height * 0.5;
        let x = (X - half_width) * scale;
        let y = (half_height - Y) * scale;

        let time = W + self.t;

        let r_sq = x * x + y * y;
        let radial = (r_sq - 0.7).abs();

        let swirl_scale = (1.0 - radial) * 5.0;
        let vx = x * swirl_scale;
        let vy = y * swirl_scale;

        let phase = time * 0.5;
        let sin_w03 = (time * 0.3).sin();
        let sin_w20 = (time * 2.0).sin();

        let swirl = ((vx + phase).sin() + 1.0) * ((vx + phase) - (vy + phase * 0.7)).abs() * 0.2 + 0.001;

        let pulse = 1.0 + sin_w20 * 0.1;
        let radial_factor = (radial * -4.0 * pulse).exp();

        let y_factor_r = (y * 1.0 + sin_w03 * 0.2).exp();
        let raw_r = y_factor_r * radial_factor / swirl;
        let soft_r = raw_r / (raw_r.abs() + 1.0);
        let red = (soft_r + 1.0) * 0.5;

        let y_factor_g = (y * -1.0 + sin_w03 * 0.2).exp();
        let raw_g = y_factor_g * radial_factor / swirl;
        let soft_g = raw_g / (raw_g.abs() + 1.0);
        let green = (soft_g + 1.0) * 0.5;

        let y_factor_b = (y * -2.0 + sin_w03 * 0.2).exp();
        let raw_b = y_factor_b * radial_factor / swirl;
        let soft_b = raw_b / (raw_b.abs() + 1.0);
        let blue = (soft_b + 1.0) * 0.5;

        Vec4::new(red, green, blue, Field::from(1.0))
    }
}

#[wasm_bindgen]
pub struct Renderer {
    width: u32,
    height: u32,
    buffer: Vec<u8>,
}

#[wasm_bindgen]
impl Renderer {
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            buffer: vec![0u8; (width * height * 4) as usize],
        }
    }

    pub fn render(&mut self, time: f32) {
        let scene = PsychedelicScene {
            t: time,
            width: self.width as f32,
            height: self.height as f32,
        };

        // Evaluate the scene row-by-row for better performance.
        for y in 0..self.height {
            let y_f32 = y as f32;
            let domain = (0.0..self.width as f32, y_f32, 0.0, time);
            let color_vec = scene.eval(domain);

            let (r_slice, g_slice, b_slice, a_slice) = color_vec.channels();

            let start_idx = (y * self.width * 4) as usize;
            for x in 0..self.width as usize {
                let r = (r_slice[x] * 255.0) as u8;
                let g = (g_slice[x] * 255.0) as u8;
                let b = (b_slice[x] * 255.0) as u8;
                let a = (a_slice[x] * 255.0) as u8;

                let idx = start_idx + x * 4;
                self.buffer[idx] = r;
                self.buffer[idx+1] = g;
                self.buffer[idx+2] = b;
                self.buffer[idx+3] = a;
            }
        }
    }

    pub fn buffer_ptr(&self) -> *const u8 {
        self.buffer.as_ptr()
    }

    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }
}
