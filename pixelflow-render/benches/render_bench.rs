use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pixelflow_render::{process_frame, Color, NamedColor, Op};

const SCREEN_WIDTH: usize = 1920;
const SCREEN_HEIGHT: usize = 1080;
const CELL_WIDTH: usize = 10;
const CELL_HEIGHT: usize = 20;

fn bench_text_styles(c: &mut Criterion) {
    let mut group = c.benchmark_group("render_text_styles");
    // 80x24 terminal
    const GRID_COLS: usize = 80;
    const GRID_ROWS: usize = 24;

    let mut fb = vec![0u32; SCREEN_WIDTH * SCREEN_HEIGHT];

    // Helper to generate commands
    let make_commands = |bold: bool, italic: bool| {
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
                    bold,
                    italic,
                });
            }
        }
        commands
    };

    let cmds_normal = make_commands(false, false);
    let cmds_bold = make_commands(true, false);
    let cmds_italic = make_commands(false, true);
    let cmds_bold_italic = make_commands(true, true);

    group.bench_function("Normal", |b| {
        b.iter(|| {
            process_frame(
                black_box(&mut fb),
                black_box(SCREEN_WIDTH),
                black_box(SCREEN_HEIGHT),
                black_box(CELL_WIDTH),
                black_box(CELL_HEIGHT),
                black_box(&cmds_normal),
            );
        })
    });

    group.bench_function("Bold", |b| {
        b.iter(|| {
            process_frame(
                black_box(&mut fb),
                black_box(SCREEN_WIDTH),
                black_box(SCREEN_HEIGHT),
                black_box(CELL_WIDTH),
                black_box(CELL_HEIGHT),
                black_box(&cmds_bold),
            );
        })
    });

    group.bench_function("Italic", |b| {
        b.iter(|| {
            process_frame(
                black_box(&mut fb),
                black_box(SCREEN_WIDTH),
                black_box(SCREEN_HEIGHT),
                black_box(CELL_WIDTH),
                black_box(CELL_HEIGHT),
                black_box(&cmds_italic),
            );
        })
    });

    group.bench_function("BoldItalic", |b| {
        b.iter(|| {
            process_frame(
                black_box(&mut fb),
                black_box(SCREEN_WIDTH),
                black_box(SCREEN_HEIGHT),
                black_box(CELL_WIDTH),
                black_box(CELL_HEIGHT),
                black_box(&cmds_bold_italic),
            );
        })
    });

    group.finish();
}

fn bench_primitives(c: &mut Criterion) {
    let mut group = c.benchmark_group("render_primitives");
    let mut fb = vec![0u32; SCREEN_WIDTH * SCREEN_HEIGHT];

    // Rect
    // Fill the screen with small rects
    let mut cmds_rect: Vec<Op<&[u8]>> = Vec::new();
    cmds_rect.push(Op::Clear { color: Color::Named(NamedColor::Black) });
    for y in (0..SCREEN_HEIGHT).step_by(100) {
        for x in (0..SCREEN_WIDTH).step_by(100) {
             cmds_rect.push(Op::Rect {
                x, y, w: 50, h: 50,
                color: Color::Named(NamedColor::Red),
            });
        }
    }

    group.bench_function("Rects", |b| {
        b.iter(|| {
            process_frame(
                black_box(&mut fb),
                black_box(SCREEN_WIDTH),
                black_box(SCREEN_HEIGHT),
                black_box(CELL_WIDTH),
                black_box(CELL_HEIGHT),
                black_box(&cmds_rect),
            );
        })
    });

    // Blit
    // Blit a large image (e.g. 500x500)
    let blit_w = 500;
    let blit_h = 500;
    let blit_data = vec![0xFF; blit_w * blit_h * 4];
    let cmds_blit = vec![
        Op::Clear { color: Color::Named(NamedColor::Black) },
        Op::Blit {
            data: &blit_data,
            w: blit_w,
            x: 100,
            y: 100,
        }
    ];

    group.bench_function("Blit_500x500", |b| {
        b.iter(|| {
             process_frame(
                black_box(&mut fb),
                black_box(SCREEN_WIDTH),
                black_box(SCREEN_HEIGHT),
                black_box(CELL_WIDTH),
                black_box(CELL_HEIGHT),
                black_box(&cmds_blit),
            );
        })
    });

    group.finish();
}

criterion_group!(benches, bench_text_styles, bench_primitives);
criterion_main!(benches);
