// build.rs - pixelflow-render build script
// Font baking is no longer needed - we use pixelflow-fonts at runtime

fn main() {
    println!("cargo:rerun-if-changed=assets/NotoSansMono-Regular.ttf");
}
