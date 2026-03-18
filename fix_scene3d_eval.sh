sed -i 's/self.inner.eval(r_x, r_y, r_z, w)/self.inner.eval((r_x, r_y, r_z, w))/g' pixelflow-graphics/src/scene3d.rs
