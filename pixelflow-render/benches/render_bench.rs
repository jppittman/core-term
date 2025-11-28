use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pixelflow_render::{process_frame, Color, NamedColor, Op};

const SCREEN_WIDTH: usize = 1920;
const SCREEN_HEIGHT: usize = 1080;
const CELL_WIDTH: usize = 10;
const CELL_HEIGHT: usize = 20;

fn bench_process_frame_text(c: &mut Criterion) {
    // 80x24 terminal
    const GRID_COLS: usize = 80;
    const GRID_ROWS: usize = 24;

    let mut fb = vec![0u32; SCREEN_WIDTH * SCREEN_HEIGHT];

    // Fill with text ops
    let mut commands: Vec<Op<&[u8]>> = Vec::with_capacity(GRID_COLS * GRID_ROWS + 1);
    commands.push(Op::Clear {
        color: Color::Named(NamedColor::Black),
    });

    for y in 0..GRID_ROWS {
        for x in 0..GRID_COLS {
            commands.push(Op::Text {
                ch: 'A',
                x,
                y,
                fg: Color::Named(NamedColor::White),
                bg: Color::Named(NamedColor::Black),
            });
        }
    }

    c.bench_function("process_frame_80x24_text", |b| {
        b.iter(|| {
            process_frame(
                black_box(&mut fb),
                black_box(SCREEN_WIDTH),
                black_box(SCREEN_HEIGHT),
                black_box(CELL_WIDTH),
                black_box(CELL_HEIGHT),
                black_box(&commands),
            );
        })
    });
}

criterion_group!(benches, bench_process_frame_text);
criterion_main!(benches);
