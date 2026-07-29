window.BENCHMARK_DATA = {
  "lastUpdate": 1785313722158,
  "repoUrl": "https://github.com/jppittman/pixelflow",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "noreply@anthropic.com",
            "name": "Claude",
            "username": "claude"
          },
          "committer": {
            "email": "166569057+jppittman@users.noreply.github.com",
            "name": "JP Pittman",
            "username": "jppittman"
          },
          "distinct": true,
          "id": "73536e9a254dacf246c3171b136066954b78f744",
          "message": "fix(pixelflow-ir): aarch64 pool must survive the collapse compile's two bodies\n\nCI caught both on the LICM push:\n\n- macOS (NEON): Aarch64Backend::begin() REPLACED the constant pool per\n  emission, but a collapse compile now emits two bodies through one\n  backend — the LICM prologue, then the loop body. The prologue's bytes\n  carry the first pool's X17-relative offsets baked in; the reset left\n  them pointing into the body's rebuilt pool, loading wrong constants\n  (glyph 'A' baked zero ink). begin() now appends into the existing\n  pool — entries are append-only so baked offsets stay valid, push_f32\n  dedups, and every compile constructs a fresh backend, so appending is\n  reset-equivalent for single-body compiles. ConstPool::from_schedule\n  deleted (begin was its only caller).\n\n- Clippy: emit_collapse_loop's six parameters are the scaffold's full\n  framing contract; allow the lint at the trait declaration.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01PHPBWpn4HqSDpo1DoqcTWW",
          "timestamp": "2026-07-28T16:42:14-07:00",
          "tree_id": "b23838d2e92c45d10d64a30855bb993845709057",
          "url": "https://github.com/jppittman/pixelflow/commit/73536e9a254dacf246c3171b136066954b78f744"
        },
        "date": 1785295107032,
        "tool": "cargo",
        "benches": [
          {
            "name": "data_throughput_under_control_flood",
            "value": 60440826,
            "range": "± 28490",
            "unit": "ns/iter"
          },
          {
            "name": "burst_limit_vs_unlimited",
            "value": 50364503,
            "range": "± 35589",
            "unit": "ns/iter"
          },
          {
            "name": "four_control_flooders_vs_data",
            "value": 60577041,
            "range": "± 294459",
            "unit": "ns/iter"
          },
          {
            "name": "slow_receiver_backpressure",
            "value": 63270046,
            "range": "± 302614",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/10",
            "value": 21136,
            "range": "± 3066",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/32",
            "value": 21735,
            "range": "± 3288",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/100",
            "value": 22648,
            "range": "± 3528",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_under_data_flood",
            "value": 1132,
            "range": "± 364",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/10",
            "value": 21624,
            "range": "± 3434",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/32",
            "value": 21949,
            "range": "± 3186",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/100",
            "value": 21424,
            "range": "± 3297",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_under_control_flood",
            "value": 1022,
            "range": "± 3768",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/1000",
            "value": 16969,
            "range": "± 65",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/1000",
            "value": 16354,
            "range": "± 84",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/10000",
            "value": 188058,
            "range": "± 436",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/10000",
            "value": 183916,
            "range": "± 1917",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/100000",
            "value": 1757708,
            "range": "± 35801",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/100000",
            "value": 1716719,
            "range": "± 22004",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/1000",
            "value": 169138,
            "range": "± 1875",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/1000",
            "value": 32231,
            "range": "± 69",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/10000",
            "value": 998954,
            "range": "± 48647",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/10000",
            "value": 349571,
            "range": "± 1419",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/100000",
            "value": 8678240,
            "range": "± 334622",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/100000",
            "value": 3332472,
            "range": "± 25439",
            "unit": "ns/iter"
          },
          {
            "name": "priority/data_uncontended",
            "value": 21215,
            "range": "± 599",
            "unit": "ns/iter"
          },
          {
            "name": "priority/data_under_control_flood",
            "value": 31961,
            "range": "± 958",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/10000",
            "value": 185206,
            "range": "± 3606",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/10000",
            "value": 103170,
            "range": "± 3920",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/100000",
            "value": 1575558,
            "range": "± 33921",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/100000",
            "value": 263366,
            "range": "± 1711",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/1000000",
            "value": 14537503,
            "range": "± 151955",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/1000000",
            "value": 5334611,
            "range": "± 180158",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/2",
            "value": 1555078,
            "range": "± 19816",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/2",
            "value": 792885,
            "range": "± 13143",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/4",
            "value": 4898776,
            "range": "± 101771",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/4",
            "value": 1456067,
            "range": "± 21554",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/8",
            "value": 10165712,
            "range": "± 253528",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/8",
            "value": 2756575,
            "range": "± 362813",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_ns/mpsc",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_ns/spsc",
            "value": 7,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_contended/mpsc/2",
            "value": 11,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_contended/sharded_spsc/2",
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_contended/mpsc/4",
            "value": 17,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_contended/sharded_spsc/4",
            "value": 5,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "roundtrip_latency/mpsc",
            "value": 23689,
            "range": "± 4342",
            "unit": "ns/iter"
          },
          {
            "name": "roundtrip_latency/spsc",
            "value": 894,
            "range": "± 77",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/1000",
            "value": 78443,
            "range": "± 2216",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/10000",
            "value": 547654,
            "range": "± 10237",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/100000",
            "value": 6101816,
            "range": "± 286864",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/100",
            "value": 78712,
            "range": "± 2216",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/1000",
            "value": 94143,
            "range": "± 4292",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/10000",
            "value": 481617,
            "range": "± 30339",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_lanes_10k_each",
            "value": 1595586,
            "range": "± 101546",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/1024",
            "value": 16317,
            "range": "± 246",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/1024",
            "value": 22104,
            "range": "± 180",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/1024",
            "value": 17128,
            "range": "± 36",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/1024",
            "value": 13184,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/1024",
            "value": 11373,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/1024",
            "value": 16812,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/4096",
            "value": 64696,
            "range": "± 131",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/4096",
            "value": 86615,
            "range": "± 430",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/4096",
            "value": 64583,
            "range": "± 155",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/4096",
            "value": 51211,
            "range": "± 361",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/4096",
            "value": 43585,
            "range": "± 67",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/4096",
            "value": 65013,
            "range": "± 661",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/16384",
            "value": 255352,
            "range": "± 380",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/16384",
            "value": 336645,
            "range": "± 2674",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/16384",
            "value": 250305,
            "range": "± 3238",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/16384",
            "value": 197936,
            "range": "± 2388",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/16384",
            "value": 171876,
            "range": "± 402",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/16384",
            "value": 263689,
            "range": "± 1280",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/65536",
            "value": 1020536,
            "range": "± 1402",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/65536",
            "value": 1338126,
            "range": "± 3753",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/65536",
            "value": 1013714,
            "range": "± 2477",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/65536",
            "value": 801557,
            "range": "± 24346",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/65536",
            "value": 699111,
            "range": "± 1843",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/65536",
            "value": 1041723,
            "range": "± 18251",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/alt_screen_random_write",
            "value": 814634,
            "range": "± 4468",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/scrolling",
            "value": 1026653,
            "range": "± 8097",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/unicode_random_write",
            "value": 692905,
            "range": "± 2462",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/osc_heavy",
            "value": 893527,
            "range": "± 9607",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_map_size_10",
            "value": 18,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_vec_size_10",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_binsearch_size_10",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_map_size_100",
            "value": 18,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_vec_size_100",
            "value": 133,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_binsearch_size_100",
            "value": 21,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_map_size_1000",
            "value": 19,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_vec_size_1000",
            "value": 1257,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_binsearch_size_1000",
            "value": 33,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_construction/zero_params",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_construction/one_param",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_construction/two_params",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_construction/with_block",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_construction/complex_expression",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_evaluation/zero_params_eval",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_evaluation/one_param_eval",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_evaluation/two_params_eval",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_evaluation/circle_sdf_eval",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "macro_vs_manual/macro_circle",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "macro_vs_manual/manual_circle",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "type_depth/depth_1",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "type_depth/depth_2",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "type_depth/depth_3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "type_depth/depth_4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_creation/from_f32_splat",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_creation/sequential",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_arithmetic/add",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_arithmetic/sub",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_arithmetic/mul",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_arithmetic/div",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_arithmetic/chained_mad",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_math/sqrt",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_math/abs",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_math/min",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_math/max",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_transcendental/log2_small",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_transcendental/log2_mid",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_transcendental/log2_large",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_transcendental/exp2_small",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_transcendental/exp2_mid",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_transcendental/exp2_large",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_transcendental/log2_exp2_roundtrip",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_comparisons/lt_manifold",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_comparisons/le_manifold",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_comparisons/gt_manifold",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_comparisons/ge_manifold",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_select/select_with_gt_ast",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_select/select_with_field_condition",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_select/select_gt_recompute_each_iter",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_bitwise/and_manifold",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_bitwise/or_manifold",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_constants/f32_constant",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_constants/X_variable",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_constants/Y_variable",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_simple/X_plus_Y",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_simple/X_mul_Y",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_simple/X_squared",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_simple/fma_X_mul_Y_plus_Z",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_simple/distance_squared",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_simple/distance_from_origin",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_circle/unit_circle_sdf",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_circle/circle_inside_test",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_select/simple_select",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_select/circle_select",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_select/nested_select",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_complex/polynomial_degree3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_complex/bilinear_interp",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_complex/min_max_chain",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_creation/x_seeded",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_creation/y_seeded",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_creation/constant",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_arithmetic/add",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_arithmetic/sub",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_arithmetic/mul",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_arithmetic/div",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_math/sqrt",
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_math/abs",
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_math/min",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_math/max",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_gradient/circle_sdf_gradient",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_gradient/polynomial_gradient",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fix_iteration/converge_fast_all_lanes",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fix_iteration/converge_10_iterations",
            "value": 10,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fix_iteration/converge_variable_lanes",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "evaluation_throughput/circle_sdf_64px",
            "value": 22,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "evaluation_throughput/circle_sdf_256px",
            "value": 85,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "evaluation_throughput/circle_sdf_1024px",
            "value": 336,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_mul_no_guard",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_mul_with_guard",
            "value": 17,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_div_no_guard",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_div_with_guard",
            "value": 17,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/manifold_denormal_heavy_no_guard",
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/manifold_denormal_heavy_with_guard",
            "value": 17,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/normal_mul_no_guard",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/normal_mul_with_guard",
            "value": 17,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_accumulation_no_guard",
            "value": 75,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_accumulation_with_guard",
            "value": 85,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/A_linear",
            "value": 11442,
            "range": "± 95",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/O_quadratic",
            "value": 108200,
            "range": "± 1388",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/S_complex",
            "value": 179480,
            "range": "± 3174",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/5",
            "value": 812879,
            "range": "± 7768",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/10",
            "value": 4415456,
            "range": "± 17778",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/26",
            "value": 63502676,
            "range": "± 240993",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/50",
            "value": 120077190,
            "range": "± 470265",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/uncached_HELLO",
            "value": 612911,
            "range": "± 6078",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/cached_HELLO",
            "value": 70553,
            "range": "± 509",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/cache_warmup_alphabet",
            "value": 5428818,
            "range": "± 76347",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial_optimization/manual_unfused",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial_optimization/kernel_optimized",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_construction/zero_params",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_construction/one_param",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_construction/two_params",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_construction/with_block",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_construction/complex_expression",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_evaluation/zero_params_eval",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_evaluation/one_param_eval",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_evaluation/two_params_eval",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_evaluation/circle_sdf_eval",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "macro_vs_manual/macro_circle",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "macro_vs_manual/manual_circle",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "type_depth/depth_1",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "type_depth/depth_2",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "type_depth/depth_3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "type_depth/depth_4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/add_xy",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/mul_xy",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/sub_xy",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/div_xy",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/add_mul",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/mul_add",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/chain3_add",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/chain3_mul",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/chain4_add",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/chain4_mul",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/wide2_add",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/wide2_mul",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/wide2_mix",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt_x",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt_xy",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt2_wide",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt3_wide",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt4_wide",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt2_deep",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt3_deep",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/div2_wide",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/div2_deep",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/div3_deep",
            "value": 7,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt_div_wide",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt_div_deep",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "distance_functions/dist2d",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "distance_functions/dist3d",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "distance_functions/dist4d",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "distance_functions/dist2d_sq",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "distance_functions/dist3d_sq",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "distance_functions/circle_sdf",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "distance_functions/sphere_sdf",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "distance_functions/box2d_sdf",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "distance_functions/normalize_x",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomials/linear",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomials/quadratic",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomials/cubic",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomials/quartic",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomials/quad2v",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomials/cubic2v",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomials/cross_xy",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomials/cross_xyz",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomials/full_quad2d",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/d2w4",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/d3w2_left",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/d3w2_right",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/d4w1",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/wide_sqrt4",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/deep_sqrt3",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/wide_div2",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/deep_div3",
            "value": 7,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "minmax/min_xy",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "minmax/max_xy",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "minmax/clamp",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "minmax/abs_via_max",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "minmax/sdf_union",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "minmax/sdf_intersect",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_raw/add_manual",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_raw/add_kernel_raw",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_raw/dist2d_manual",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_raw/dist2d_kernel_raw",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_raw/sdf_union_manual",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_raw/sdf_union_kernel_raw",
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_raw/fma_manual",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_raw/fma_kernel_raw",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/sin",
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/cos",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/sin_cos",
            "value": 7,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/exp",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/ln",
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/atan2",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/pow",
            "value": 12,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "166569057+jppittman@users.noreply.github.com",
            "name": "JP Pittman",
            "username": "jppittman"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e85c416b52882c7dad94112440beb1c6bc0368df",
          "message": "feat(ir): collapse full 2D lattice in JIT (#961)\n\n## Summary\n\n- replace the row-only collapse ABI with one 2D X/Y loop nest on SSE2,\nAVX2, AVX-512, and NEON\n- add two-level LICM: X/Y-invariant values execute once per plane and\nX-invariant values once per row\n- make `Lattice::bake` submit each Z/W plane's full-width region in one\nJIT call while preserving scalar SIMD tails\n- replace the ad-hoc timing example with a Criterion call-overhead\nbenchmark\n\n## Why\n\nThe existing collapse kernel removed the Rust-to-JIT boundary per SIMD\nbatch but still crossed it once per row and recomputed Z/W-only work in\nevery row prologue. The 2D ABI closes that remaining render-loop\nboundary while keeping tail writes explicit and safe.\n\n## Validation\n\n- `cargo test --workspace`\n- `cargo clippy --workspace --all-targets --all-features -- -D warnings`\n- `cargo fmt --all -- --check`\n- `cargo test -p pixelflow-ir --test collapse_loop`\n- `cargo test -p pixelflow-core --test kernel_bake`\n- `cargo bench -p pixelflow-ir --bench collapse_overhead -- --noplot`\n\nOn the SSE2 development host, the deliberately cheap 61,440-pixel\nbenchmark is effectively tied: 49.23µs for the Rust per-batch loop and\n49.63µs for one 2D collapse call. The benchmark is retained to catch\nregressions and to separate call-boundary cost from production LICM\nwins.\n\nCo-authored-by: JP Pittman <jppittman@jpptech.dev>",
          "timestamp": "2026-07-28T23:46:41-07:00",
          "tree_id": "b3d3662f37b5bbd41f46895c934be9f05c5edd73",
          "url": "https://github.com/jppittman/pixelflow/commit/e85c416b52882c7dad94112440beb1c6bc0368df"
        },
        "date": 1785313721776,
        "tool": "cargo",
        "benches": [
          {
            "name": "data_throughput_under_control_flood",
            "value": 60477729,
            "range": "± 22228",
            "unit": "ns/iter"
          },
          {
            "name": "burst_limit_vs_unlimited",
            "value": 50374538,
            "range": "± 33967",
            "unit": "ns/iter"
          },
          {
            "name": "four_control_flooders_vs_data",
            "value": 60546946,
            "range": "± 569910",
            "unit": "ns/iter"
          },
          {
            "name": "slow_receiver_backpressure",
            "value": 63495922,
            "range": "± 262385",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/10",
            "value": 22244,
            "range": "± 3642",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/32",
            "value": 23387,
            "range": "± 3172",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/100",
            "value": 23388,
            "range": "± 3486",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_under_data_flood",
            "value": 2340,
            "range": "± 707",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/10",
            "value": 23494,
            "range": "± 3566",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/32",
            "value": 23028,
            "range": "± 3550",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/100",
            "value": 24104,
            "range": "± 3447",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_under_control_flood",
            "value": 5575,
            "range": "± 3434",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/1000",
            "value": 17512,
            "range": "± 316",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/1000",
            "value": 16520,
            "range": "± 334",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/10000",
            "value": 195319,
            "range": "± 2766",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/10000",
            "value": 186895,
            "range": "± 3299",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/100000",
            "value": 1797625,
            "range": "± 41293",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/100000",
            "value": 1734795,
            "range": "± 31893",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/1000",
            "value": 175282,
            "range": "± 2825",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/1000",
            "value": 35159,
            "range": "± 1036",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/10000",
            "value": 1047983,
            "range": "± 58127",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/10000",
            "value": 358837,
            "range": "± 9986",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/100000",
            "value": 9647941,
            "range": "± 390475",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/100000",
            "value": 3377537,
            "range": "± 133050",
            "unit": "ns/iter"
          },
          {
            "name": "priority/data_uncontended",
            "value": 24280,
            "range": "± 592",
            "unit": "ns/iter"
          },
          {
            "name": "priority/data_under_control_flood",
            "value": 32720,
            "range": "± 194",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/10000",
            "value": 190190,
            "range": "± 3889",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/10000",
            "value": 114650,
            "range": "± 2380",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/100000",
            "value": 1457590,
            "range": "± 22392",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/100000",
            "value": 625358,
            "range": "± 14153",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/1000000",
            "value": 10042912,
            "range": "± 194705",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/1000000",
            "value": 4905875,
            "range": "± 132496",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/2",
            "value": 1705626,
            "range": "± 46430",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/2",
            "value": 751116,
            "range": "± 6286",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/4",
            "value": 5329022,
            "range": "± 162459",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/4",
            "value": 1471254,
            "range": "± 24017",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/8",
            "value": 10366814,
            "range": "± 458271",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/8",
            "value": 2847649,
            "range": "± 277215",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_ns/mpsc",
            "value": 15,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_ns/spsc",
            "value": 5,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_contended/mpsc/2",
            "value": 12,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_contended/sharded_spsc/2",
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_contended/mpsc/4",
            "value": 16,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_contended/sharded_spsc/4",
            "value": 6,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "roundtrip_latency/mpsc",
            "value": 25571,
            "range": "± 4232",
            "unit": "ns/iter"
          },
          {
            "name": "roundtrip_latency/spsc",
            "value": 899,
            "range": "± 139",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/1000",
            "value": 80957,
            "range": "± 4752",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/10000",
            "value": 280037,
            "range": "± 2871",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/100000",
            "value": 2312368,
            "range": "± 12455",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/100",
            "value": 83203,
            "range": "± 1826",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/1000",
            "value": 123180,
            "range": "± 5704",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/10000",
            "value": 606169,
            "range": "± 24113",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_lanes_10k_each",
            "value": 686922,
            "range": "± 11545",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/1024",
            "value": 17277,
            "range": "± 123",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/1024",
            "value": 24273,
            "range": "± 448",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/1024",
            "value": 17188,
            "range": "± 145",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/1024",
            "value": 13390,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/1024",
            "value": 11872,
            "range": "± 148",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/1024",
            "value": 17929,
            "range": "± 111",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/4096",
            "value": 68102,
            "range": "± 401",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/4096",
            "value": 93947,
            "range": "± 1242",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/4096",
            "value": 64780,
            "range": "± 579",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/4096",
            "value": 51408,
            "range": "± 1150",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/4096",
            "value": 44955,
            "range": "± 758",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/4096",
            "value": 69380,
            "range": "± 744",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/16384",
            "value": 270449,
            "range": "± 1439",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/16384",
            "value": 365061,
            "range": "± 7305",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/16384",
            "value": 254606,
            "range": "± 1531",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/16384",
            "value": 200183,
            "range": "± 853",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/16384",
            "value": 178101,
            "range": "± 2051",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/16384",
            "value": 281464,
            "range": "± 1753",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/65536",
            "value": 1098909,
            "range": "± 8195",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/65536",
            "value": 1385026,
            "range": "± 13879",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/65536",
            "value": 1030506,
            "range": "± 11298",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/65536",
            "value": 809593,
            "range": "± 1784",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/65536",
            "value": 727360,
            "range": "± 3994",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/65536",
            "value": 1108067,
            "range": "± 7910",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/alt_screen_random_write",
            "value": 824931,
            "range": "± 6310",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/scrolling",
            "value": 1092136,
            "range": "± 14743",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/unicode_random_write",
            "value": 725902,
            "range": "± 8729",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/osc_heavy",
            "value": 897929,
            "range": "± 3658",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_map_size_10",
            "value": 18,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_vec_size_10",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_binsearch_size_10",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_map_size_100",
            "value": 18,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_vec_size_100",
            "value": 133,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_binsearch_size_100",
            "value": 21,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_map_size_1000",
            "value": 19,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_vec_size_1000",
            "value": 1259,
            "range": "± 11",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_binsearch_size_1000",
            "value": 33,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_construction/zero_params",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_construction/one_param",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_construction/two_params",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_construction/with_block",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_construction/complex_expression",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_evaluation/zero_params_eval",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_evaluation/one_param_eval",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_evaluation/two_params_eval",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_evaluation/circle_sdf_eval",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "macro_vs_manual/macro_circle",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "macro_vs_manual/manual_circle",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "type_depth/depth_1",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "type_depth/depth_2",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "type_depth/depth_3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "type_depth/depth_4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_creation/from_f32_splat",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_creation/sequential",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_arithmetic/add",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_arithmetic/sub",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_arithmetic/mul",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_arithmetic/div",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_arithmetic/chained_mad",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_math/sqrt",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_math/abs",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_math/min",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_math/max",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_transcendental/log2_small",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_transcendental/log2_mid",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_transcendental/log2_large",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_transcendental/exp2_small",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_transcendental/exp2_mid",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_transcendental/exp2_large",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_transcendental/log2_exp2_roundtrip",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_comparisons/lt_manifold",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_comparisons/le_manifold",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_comparisons/gt_manifold",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_comparisons/ge_manifold",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_select/select_with_gt_ast",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_select/select_with_field_condition",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_select/select_gt_recompute_each_iter",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_bitwise/and_manifold",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_bitwise/or_manifold",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_constants/f32_constant",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_constants/X_variable",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_constants/Y_variable",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_simple/X_plus_Y",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_simple/X_mul_Y",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_simple/X_squared",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_simple/fma_X_mul_Y_plus_Z",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_simple/distance_squared",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_simple/distance_from_origin",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_circle/unit_circle_sdf",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_circle/circle_inside_test",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_select/simple_select",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_select/circle_select",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_select/nested_select",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_complex/polynomial_degree3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_complex/bilinear_interp",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_complex/min_max_chain",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_creation/x_seeded",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_creation/y_seeded",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_creation/constant",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_arithmetic/add",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_arithmetic/sub",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_arithmetic/mul",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_arithmetic/div",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_math/sqrt",
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_math/abs",
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_math/min",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_math/max",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_gradient/circle_sdf_gradient",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_gradient/polynomial_gradient",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fix_iteration/converge_fast_all_lanes",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fix_iteration/converge_10_iterations",
            "value": 10,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fix_iteration/converge_variable_lanes",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "evaluation_throughput/circle_sdf_64px",
            "value": 22,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "evaluation_throughput/circle_sdf_256px",
            "value": 85,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "evaluation_throughput/circle_sdf_1024px",
            "value": 336,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_mul_no_guard",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_mul_with_guard",
            "value": 17,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_div_no_guard",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_div_with_guard",
            "value": 17,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/manifold_denormal_heavy_no_guard",
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/manifold_denormal_heavy_with_guard",
            "value": 17,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/normal_mul_no_guard",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/normal_mul_with_guard",
            "value": 17,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_accumulation_no_guard",
            "value": 75,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_accumulation_with_guard",
            "value": 85,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/A_linear",
            "value": 10906,
            "range": "± 89",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/O_quadratic",
            "value": 109582,
            "range": "± 927",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/S_complex",
            "value": 173432,
            "range": "± 1670",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/5",
            "value": 803645,
            "range": "± 2513",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/10",
            "value": 4446864,
            "range": "± 25728",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/26",
            "value": 62554446,
            "range": "± 186067",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/50",
            "value": 118380416,
            "range": "± 226651",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/uncached_HELLO",
            "value": 614117,
            "range": "± 3585",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/cached_HELLO",
            "value": 77937,
            "range": "± 355",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/cache_warmup_alphabet",
            "value": 5374083,
            "range": "± 68951",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial_optimization/manual_unfused",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial_optimization/kernel_optimized",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jit_collapse_call_overhead/rust_per_batch_loop/4",
            "value": 102721,
            "range": "± 188",
            "unit": "ns/iter"
          },
          {
            "name": "jit_collapse_call_overhead/one_2d_collapse_call/4",
            "value": 57678,
            "range": "± 220",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_construction/zero_params",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_construction/one_param",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_construction/two_params",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_construction/with_block",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_construction/complex_expression",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_evaluation/zero_params_eval",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_evaluation/one_param_eval",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_evaluation/two_params_eval",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_evaluation/circle_sdf_eval",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "macro_vs_manual/macro_circle",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "macro_vs_manual/manual_circle",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "type_depth/depth_1",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "type_depth/depth_2",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "type_depth/depth_3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "type_depth/depth_4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/add_xy",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/mul_xy",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/sub_xy",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/div_xy",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/add_mul",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/mul_add",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/chain3_add",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/chain3_mul",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/chain4_add",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/chain4_mul",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/wide2_add",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/wide2_mul",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/wide2_mix",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt_x",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt_xy",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt2_wide",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt3_wide",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt4_wide",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt2_deep",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt3_deep",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/div2_wide",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/div2_deep",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/div3_deep",
            "value": 7,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt_div_wide",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt_div_deep",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "distance_functions/dist2d",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "distance_functions/dist3d",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "distance_functions/dist4d",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "distance_functions/dist2d_sq",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "distance_functions/dist3d_sq",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "distance_functions/circle_sdf",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "distance_functions/sphere_sdf",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "distance_functions/box2d_sdf",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "distance_functions/normalize_x",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomials/linear",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomials/quadratic",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomials/cubic",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomials/quartic",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomials/quad2v",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomials/cubic2v",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomials/cross_xy",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomials/cross_xyz",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomials/full_quad2d",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/d2w4",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/d3w2_left",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/d3w2_right",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/d4w1",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/wide_sqrt4",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/deep_sqrt3",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/wide_div2",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/deep_div3",
            "value": 7,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "minmax/min_xy",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "minmax/max_xy",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "minmax/clamp",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "minmax/abs_via_max",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "minmax/sdf_union",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "minmax/sdf_intersect",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_raw/add_manual",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_raw/add_kernel_raw",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_raw/dist2d_manual",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_raw/dist2d_kernel_raw",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_raw/sdf_union_manual",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_raw/sdf_union_kernel_raw",
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_raw/fma_manual",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_raw/fma_kernel_raw",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/sin",
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/cos",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/sin_cos",
            "value": 7,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/exp",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/ln",
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/atan2",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/pow",
            "value": 12,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}