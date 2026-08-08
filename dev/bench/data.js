window.BENCHMARK_DATA = {
  "lastUpdate": 1786230865505,
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
          "id": "b9b3fdf1a0a0e5c11ad6e1298d5d885a7013486c",
          "message": "ci: require ABI and render behavior contracts (#963)\n\n## What changed\n- cross-compile the public `kernel!` consumer parity suite for the\nsupported aarch64 JIT ABI\n- make default-vs-arena routing parity and JIT-vs-interpreter glyph\ngoldens explicit CI contracts\n- make scene color/reflection/antialiasing and ray/surface render\ncontracts fail loudly when deleted or renamed\n\n## Why\nPR #959 passed the broad workspace suite while deleting or weakening\nbehavior coverage and changing architecture-sensitive routing. These\nnamed gates turn those behaviors into merge requirements instead of\nrelying on test discovery alone.\n\n## Local validation\n- `cargo check -p pixelflow-compiler --test kernel_routing_parity\n--target aarch64-unknown-linux-gnu`\n- default and `arena-backend` kernel routing parity: 10/10 passed\n- JIT/interpreter glyph goldens: 4/4 passed\n- selected scene behavior contracts: 3/3 passed\n- ray/surface composition contracts: 3/3 passed\n- `cargo fmt --all -- --check`\n- workflow YAML parse and `git diff --check`\n\n---------\n\nCo-authored-by: JP Pittman <jppittman@jpptech.dev>",
          "timestamp": "2026-07-29T07:43:23Z",
          "tree_id": "9cb3c94fcea759e6a32815d19f554b3856f2c7fc",
          "url": "https://github.com/jppittman/pixelflow/commit/b9b3fdf1a0a0e5c11ad6e1298d5d885a7013486c"
        },
        "date": 1785317365816,
        "tool": "cargo",
        "benches": [
          {
            "name": "data_throughput_under_control_flood",
            "value": 60455941,
            "range": "± 35136",
            "unit": "ns/iter"
          },
          {
            "name": "burst_limit_vs_unlimited",
            "value": 50350399,
            "range": "± 24788",
            "unit": "ns/iter"
          },
          {
            "name": "four_control_flooders_vs_data",
            "value": 60541935,
            "range": "± 1561565",
            "unit": "ns/iter"
          },
          {
            "name": "slow_receiver_backpressure",
            "value": 63472195,
            "range": "± 93011",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/10",
            "value": 12121,
            "range": "± 3884",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/32",
            "value": 11616,
            "range": "± 4142",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/100",
            "value": 11161,
            "range": "± 4805",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_under_data_flood",
            "value": 683,
            "range": "± 203",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/10",
            "value": 11588,
            "range": "± 4172",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/32",
            "value": 12629,
            "range": "± 3961",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/100",
            "value": 11803,
            "range": "± 4392",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_under_control_flood",
            "value": 762,
            "range": "± 270",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/1000",
            "value": 17251,
            "range": "± 39",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/1000",
            "value": 15196,
            "range": "± 61",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/10000",
            "value": 192542,
            "range": "± 2595",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/10000",
            "value": 181284,
            "range": "± 1131",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/100000",
            "value": 1783204,
            "range": "± 16578",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/100000",
            "value": 1676227,
            "range": "± 25723",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/1000",
            "value": 152506,
            "range": "± 1401",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/1000",
            "value": 31005,
            "range": "± 100",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/10000",
            "value": 1011798,
            "range": "± 52327",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/10000",
            "value": 341437,
            "range": "± 1112",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/100000",
            "value": 9157792,
            "range": "± 313377",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/100000",
            "value": 3240131,
            "range": "± 25689",
            "unit": "ns/iter"
          },
          {
            "name": "priority/data_uncontended",
            "value": 17687,
            "range": "± 69",
            "unit": "ns/iter"
          },
          {
            "name": "priority/data_under_control_flood",
            "value": 26296,
            "range": "± 1062",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/10000",
            "value": 186906,
            "range": "± 6225",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/10000",
            "value": 99824,
            "range": "± 1458",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/100000",
            "value": 1390987,
            "range": "± 41144",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/100000",
            "value": 696075,
            "range": "± 4390",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/1000000",
            "value": 11163911,
            "range": "± 172636",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/1000000",
            "value": 6508148,
            "range": "± 104780",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/2",
            "value": 1509608,
            "range": "± 70506",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/2",
            "value": 789967,
            "range": "± 6853",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/4",
            "value": 5554541,
            "range": "± 76533",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/4",
            "value": 1544914,
            "range": "± 29314",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/8",
            "value": 11807852,
            "range": "± 700237",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/8",
            "value": 3085957,
            "range": "± 324567",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_ns/mpsc",
            "value": 18,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_ns/spsc",
            "value": 6,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_contended/mpsc/2",
            "value": 13,
            "range": "± 3",
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
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_contended/sharded_spsc/4",
            "value": 5,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "roundtrip_latency/mpsc",
            "value": 14274,
            "range": "± 4276",
            "unit": "ns/iter"
          },
          {
            "name": "roundtrip_latency/spsc",
            "value": 1099,
            "range": "± 119",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/1000",
            "value": 69158,
            "range": "± 1342",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/10000",
            "value": 466778,
            "range": "± 16312",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/100000",
            "value": 5325793,
            "range": "± 317268",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/100",
            "value": 69080,
            "range": "± 979",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/1000",
            "value": 153004,
            "range": "± 9276",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/10000",
            "value": 1080488,
            "range": "± 135376",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_lanes_10k_each",
            "value": 3715780,
            "range": "± 224944",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/1024",
            "value": 15424,
            "range": "± 260",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/1024",
            "value": 22360,
            "range": "± 281",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/1024",
            "value": 17327,
            "range": "± 101",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/1024",
            "value": 13198,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/1024",
            "value": 10687,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/1024",
            "value": 16642,
            "range": "± 223",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/4096",
            "value": 60471,
            "range": "± 478",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/4096",
            "value": 85607,
            "range": "± 208",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/4096",
            "value": 62593,
            "range": "± 133",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/4096",
            "value": 48177,
            "range": "± 1189",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/4096",
            "value": 40502,
            "range": "± 104",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/4096",
            "value": 63301,
            "range": "± 621",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/16384",
            "value": 238246,
            "range": "± 316",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/16384",
            "value": 334201,
            "range": "± 1026",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/16384",
            "value": 243586,
            "range": "± 10330",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/16384",
            "value": 187151,
            "range": "± 1267",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/16384",
            "value": 159014,
            "range": "± 323",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/16384",
            "value": 251121,
            "range": "± 741",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/65536",
            "value": 975861,
            "range": "± 10652",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/65536",
            "value": 1341080,
            "range": "± 3708",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/65536",
            "value": 988409,
            "range": "± 3950",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/65536",
            "value": 768200,
            "range": "± 3794",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/65536",
            "value": 650059,
            "range": "± 736",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/65536",
            "value": 988633,
            "range": "± 3386",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/alt_screen_random_write",
            "value": 790949,
            "range": "± 21408",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/scrolling",
            "value": 974976,
            "range": "± 739",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/unicode_random_write",
            "value": 652026,
            "range": "± 1691",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/osc_heavy",
            "value": 843525,
            "range": "± 4172",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_map_size_10",
            "value": 19,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_vec_size_10",
            "value": 13,
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
            "value": 19,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_vec_size_100",
            "value": 121,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_binsearch_size_100",
            "value": 23,
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
            "value": 1223,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_binsearch_size_1000",
            "value": 38,
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
            "value": 1,
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
            "value": 1,
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
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_arithmetic/sub",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_arithmetic/mul",
            "value": 5,
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
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_math/max",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_gradient/circle_sdf_gradient",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_gradient/polynomial_gradient",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fix_iteration/converge_fast_all_lanes",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fix_iteration/converge_10_iterations",
            "value": 11,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fix_iteration/converge_variable_lanes",
            "value": 10,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "evaluation_throughput/circle_sdf_64px",
            "value": 24,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "evaluation_throughput/circle_sdf_256px",
            "value": 92,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "evaluation_throughput/circle_sdf_1024px",
            "value": 365,
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
            "value": 21,
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
            "value": 21,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/manifold_denormal_heavy_no_guard",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/manifold_denormal_heavy_with_guard",
            "value": 19,
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
            "value": 21,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_accumulation_no_guard",
            "value": 69,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_accumulation_with_guard",
            "value": 80,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/A_linear",
            "value": 10891,
            "range": "± 123",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/O_quadratic",
            "value": 96072,
            "range": "± 463",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/S_complex",
            "value": 165092,
            "range": "± 683",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/5",
            "value": 794193,
            "range": "± 3224",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/10",
            "value": 4149728,
            "range": "± 13675",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/26",
            "value": 22287805,
            "range": "± 91987",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/50",
            "value": 41958799,
            "range": "± 148923",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/uncached_HELLO",
            "value": 603431,
            "range": "± 3753",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/cached_HELLO",
            "value": 69315,
            "range": "± 364",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/cache_warmup_alphabet",
            "value": 5118959,
            "range": "± 60244",
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
            "value": 114576,
            "range": "± 926",
            "unit": "ns/iter"
          },
          {
            "name": "jit_collapse_call_overhead/one_2d_collapse_call/4",
            "value": 63697,
            "range": "± 248",
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
            "value": 1,
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
            "value": 1,
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
            "value": 3,
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
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/div2_wide",
            "value": 2,
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
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt_div_wide",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt_div_deep",
            "value": 1,
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
            "value": 3,
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
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/deep_sqrt3",
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/wide_div2",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/deep_div3",
            "value": 3,
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
            "value": 3,
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
            "value": 3,
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
            "value": 9,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/cos",
            "value": 8,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/sin_cos",
            "value": 11,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/exp",
            "value": 10,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/ln",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/atan2",
            "value": 7,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/pow",
            "value": 17,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jppittman@jpptech.dev",
            "name": "JP Pittman"
          },
          "committer": {
            "email": "166569057+jppittman@users.noreply.github.com",
            "name": "JP Pittman",
            "username": "jppittman"
          },
          "distinct": true,
          "id": "1e0b5000095423325759b27e1b1400695fad3fd2",
          "message": "chore(ci): enforce CL metadata tags",
          "timestamp": "2026-07-29T15:15:29-07:00",
          "tree_id": "d9359918f5e877a3ab386472499d311b483bbdfc",
          "url": "https://github.com/jppittman/pixelflow/commit/1e0b5000095423325759b27e1b1400695fad3fd2"
        },
        "date": 1785366955107,
        "tool": "cargo",
        "benches": [
          {
            "name": "data_throughput_under_control_flood",
            "value": 60473332,
            "range": "± 28979",
            "unit": "ns/iter"
          },
          {
            "name": "burst_limit_vs_unlimited",
            "value": 50405624,
            "range": "± 28883",
            "unit": "ns/iter"
          },
          {
            "name": "four_control_flooders_vs_data",
            "value": 60534240,
            "range": "± 1191574",
            "unit": "ns/iter"
          },
          {
            "name": "slow_receiver_backpressure",
            "value": 63488224,
            "range": "± 209586",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/10",
            "value": 11723,
            "range": "± 4405",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/32",
            "value": 12453,
            "range": "± 4464",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/100",
            "value": 11805,
            "range": "± 4577",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_under_data_flood",
            "value": 683,
            "range": "± 327",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/10",
            "value": 13112,
            "range": "± 4182",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/32",
            "value": 14281,
            "range": "± 4422",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/100",
            "value": 13657,
            "range": "± 4101",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_under_control_flood",
            "value": 872,
            "range": "± 394",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/1000",
            "value": 17191,
            "range": "± 55",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/1000",
            "value": 15217,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/10000",
            "value": 192485,
            "range": "± 1357",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/10000",
            "value": 177795,
            "range": "± 738",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/100000",
            "value": 1778813,
            "range": "± 32292",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/100000",
            "value": 1638608,
            "range": "± 27345",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/1000",
            "value": 153641,
            "range": "± 1344",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/1000",
            "value": 31035,
            "range": "± 137",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/10000",
            "value": 1063757,
            "range": "± 53421",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/10000",
            "value": 342199,
            "range": "± 1106",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/100000",
            "value": 9380344,
            "range": "± 341258",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/100000",
            "value": 3276055,
            "range": "± 12837",
            "unit": "ns/iter"
          },
          {
            "name": "priority/data_uncontended",
            "value": 17439,
            "range": "± 465",
            "unit": "ns/iter"
          },
          {
            "name": "priority/data_under_control_flood",
            "value": 26279,
            "range": "± 203",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/10000",
            "value": 187862,
            "range": "± 6714",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/10000",
            "value": 99096,
            "range": "± 1968",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/100000",
            "value": 1372102,
            "range": "± 30367",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/100000",
            "value": 645027,
            "range": "± 5465",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/1000000",
            "value": 13228489,
            "range": "± 173308",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/1000000",
            "value": 5796224,
            "range": "± 44697",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/2",
            "value": 1458919,
            "range": "± 34130",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/2",
            "value": 817695,
            "range": "± 27959",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/4",
            "value": 5100866,
            "range": "± 97696",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/4",
            "value": 1553594,
            "range": "± 23083",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/8",
            "value": 9746521,
            "range": "± 1476480",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/8",
            "value": 3047254,
            "range": "± 321574",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_ns/mpsc",
            "value": 15,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_ns/spsc",
            "value": 7,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_contended/mpsc/2",
            "value": 12,
            "range": "± 3",
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
            "range": "± 9",
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
            "value": 13759,
            "range": "± 3816",
            "unit": "ns/iter"
          },
          {
            "name": "roundtrip_latency/spsc",
            "value": 1162,
            "range": "± 120",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/1000",
            "value": 69439,
            "range": "± 1066",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/10000",
            "value": 670175,
            "range": "± 22590",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/100000",
            "value": 7902890,
            "range": "± 415436",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/100",
            "value": 67595,
            "range": "± 1389",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/1000",
            "value": 93014,
            "range": "± 6597",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/10000",
            "value": 371650,
            "range": "± 10913",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_lanes_10k_each",
            "value": 3027114,
            "range": "± 111021",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/1024",
            "value": 15931,
            "range": "± 84",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/1024",
            "value": 22561,
            "range": "± 174",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/1024",
            "value": 16789,
            "range": "± 165",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/1024",
            "value": 12888,
            "range": "± 67",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/1024",
            "value": 10609,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/1024",
            "value": 16522,
            "range": "± 113",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/4096",
            "value": 63223,
            "range": "± 660",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/4096",
            "value": 87454,
            "range": "± 251",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/4096",
            "value": 63352,
            "range": "± 248",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/4096",
            "value": 49271,
            "range": "± 294",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/4096",
            "value": 40893,
            "range": "± 277",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/4096",
            "value": 63178,
            "range": "± 338",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/16384",
            "value": 242863,
            "range": "± 839",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/16384",
            "value": 337073,
            "range": "± 486",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/16384",
            "value": 244440,
            "range": "± 8844",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/16384",
            "value": 188560,
            "range": "± 15765",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/16384",
            "value": 160919,
            "range": "± 204",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/16384",
            "value": 255489,
            "range": "± 523",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/65536",
            "value": 984066,
            "range": "± 1231",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/65536",
            "value": 1346961,
            "range": "± 2439",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/65536",
            "value": 992839,
            "range": "± 4344",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/65536",
            "value": 767393,
            "range": "± 3249",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/65536",
            "value": 654273,
            "range": "± 2945",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/65536",
            "value": 1003407,
            "range": "± 1053",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/alt_screen_random_write",
            "value": 796331,
            "range": "± 3548",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/scrolling",
            "value": 1020208,
            "range": "± 15287",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/unicode_random_write",
            "value": 661464,
            "range": "± 1239",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/osc_heavy",
            "value": 835686,
            "range": "± 3968",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_map_size_10",
            "value": 19,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_vec_size_10",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_binsearch_size_10",
            "value": 15,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_map_size_100",
            "value": 19,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_vec_size_100",
            "value": 121,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_binsearch_size_100",
            "value": 23,
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
            "value": 1212,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_binsearch_size_1000",
            "value": 38,
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
            "value": 1,
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
            "value": 1,
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
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_arithmetic/sub",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_arithmetic/mul",
            "value": 5,
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
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_math/max",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_gradient/circle_sdf_gradient",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_gradient/polynomial_gradient",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fix_iteration/converge_fast_all_lanes",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fix_iteration/converge_10_iterations",
            "value": 11,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fix_iteration/converge_variable_lanes",
            "value": 10,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "evaluation_throughput/circle_sdf_64px",
            "value": 24,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "evaluation_throughput/circle_sdf_256px",
            "value": 92,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "evaluation_throughput/circle_sdf_1024px",
            "value": 365,
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
            "value": 21,
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
            "value": 21,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/manifold_denormal_heavy_no_guard",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/manifold_denormal_heavy_with_guard",
            "value": 19,
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
            "value": 21,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_accumulation_no_guard",
            "value": 69,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_accumulation_with_guard",
            "value": 80,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/A_linear",
            "value": 10712,
            "range": "± 99",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/O_quadratic",
            "value": 95597,
            "range": "± 478",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/S_complex",
            "value": 159524,
            "range": "± 1838",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/5",
            "value": 781213,
            "range": "± 4504",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/10",
            "value": 3995910,
            "range": "± 15541",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/26",
            "value": 21617952,
            "range": "± 47671",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/50",
            "value": 40661310,
            "range": "± 55487",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/uncached_HELLO",
            "value": 580268,
            "range": "± 3429",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/cached_HELLO",
            "value": 69219,
            "range": "± 239",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/cache_warmup_alphabet",
            "value": 5106411,
            "range": "± 79022",
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
            "value": 114478,
            "range": "± 157",
            "unit": "ns/iter"
          },
          {
            "name": "jit_collapse_call_overhead/one_2d_collapse_call/4",
            "value": 63871,
            "range": "± 279",
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
            "value": 1,
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
            "value": 1,
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
            "value": 3,
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
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/div2_wide",
            "value": 2,
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
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt_div_wide",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt_div_deep",
            "value": 1,
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
            "value": 3,
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
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/deep_sqrt3",
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/wide_div2",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/deep_div3",
            "value": 3,
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
            "value": 3,
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
            "value": 3,
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
            "value": 9,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/cos",
            "value": 8,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/sin_cos",
            "value": 11,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/exp",
            "value": 12,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/ln",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/atan2",
            "value": 7,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/pow",
            "value": 17,
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
          "id": "67acccf10d509f82ded47ad128582db6cae6ae13",
          "message": "feat(actor-scheduler): one runtime primitive — step_os on Transducer, DedicatedThread driver (#967)\n\n## Summary\n\nSlice 1 of unifying the classic `Actor` API and the green `Transducer`\nAPI: the mealy `Transducer` + `Node` becomes the **single runtime\nprimitive**, and placement — dedicated OS thread vs hosted green —\nbecomes a driver choice, making the design doc's §5 placement table\nliteral.\n\n- **`Transducer::step_os`** — the OS-bridge hook in effects-as-values\nform, defaulted to silent + `Idle`. Called only by a dedicated-thread\ndriver when lanes are quiet, never by a `Host` (a green actor may not\nblock; placement is exactly the choice of which driver runs you).\n- **`Node::poll_os`** — runs `step_os` through the same dispatch\ndiscipline as the lanes (outbox flushed first; step → continuation →\nflush → park) via `finish_step`, the shared tail split out of `dispatch`\nso the two paths cannot drift. Continuation-first holds across both\nentry points, and a stored continuation or freshly parked outbox\noverrides an `Idle` hint to `Busy` — deferred work only another poll can\nadvance.\n- **`DedicatedThread`** — the mirror of `GreenThread`: one `Node` on its\nown thread, doorbell-blocking loop. Built only by\n`ActorBuilder::build_node` over the same three sharded lanes `build()`\nwould seal — so **every existing `ActorHandle` keeps working\nunchanged**. The load-bearing bridge is `impl mealy::Inbox for\nShardedInbox`.\n- **Driver-loop semantics hardened across three review rounds** (all\nfindings were real; each has a named regression test):\n- Lane draining is bounded per sweep (`sweep_burst`, the classic\nper-wake burst budget) so a flooded lane can't starve `Shutdown` or\n`step_os`.\n- A step whose output parks still reports `Ran` (`Host::sweep`'s busy\nflag depends on it); the park is the next poll's news.\n- A dead wiring target exits as `Exit::Failed` — no supervisor exists\nhere, so retrying forever just burns a core; the undelivered payload\nstays retained in the node.\n- **Terminality belongs to the doorbell**: dead lanes alone are not\ncompletion. A held `Waker` keeps the doorbell connected, so an idle OS\nbridge sleeps at 0% CPU and processes late push-then-wake events;\n`Exit::Completed` requires `Shutdown` or full doorbell disconnection\nplus an empty final drain.\n- Untouched this slice: `Actor`, `ActorScheduler::run`, `poll_once`,\n`troupe!`, all production actors. Slice 2 dissolves the shells\n(EngineHandler `flush` → `Wiring`, RasterizerActor,\nPtyParser/TerminalApp → `Transducer`); slice 3 moves the OS bridges onto\n`step_os` and retires `Actor`/`poll_once`.\n\n## Test plan\n\n- [x] 6 `poll_os` unit tests: flush-through-wiring, `Busy` propagation,\ncontinuation slot, parked-outbox deferral, fresh-park Busy override,\npending-continuation deferral of `step_os`\n- [x] `tests/dedicated_thread.rs` (9 tests): priority order over real\n`ActorHandle`s; OS-bridge drain + clean shutdown;\n`the_same_transducer_runs_dedicated_or_hosted` (the unification claim);\nflood-vs-shutdown; final `step_os` turn after handle drop;\ncontinuation-without-wake; parked-final-word retention; dead-peer fails\nloudly; idle-bridge-survives-live-waker\n- [x] `Host` regression: a parked first output keeps the host busy\n- [x] `cargo test --workspace` — 1700 passed, 0 failed; driver suite\nre-run 12× for timing stability\n- [x] `cargo fmt --check` and `cargo clippy --workspace --all-targets --\n-D warnings` — clean\n- [x] Branch updated onto current `main`\n\n🤖 Generated with [Claude Code](https://claude.ai/code)\n\nhttps://claude.ai/code/session_015Y5cNzR9PnASfGfjBh5kJc\n\n---------\n\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-07-29T20:11:41-07:00",
          "tree_id": "ed9787adcda526ff5a8fd91d4c6ea3aaa3e2ab2a",
          "url": "https://github.com/jppittman/pixelflow/commit/67acccf10d509f82ded47ad128582db6cae6ae13"
        },
        "date": 1785384792983,
        "tool": "cargo",
        "benches": [
          {
            "name": "data_throughput_under_control_flood",
            "value": 60478219,
            "range": "± 27054",
            "unit": "ns/iter"
          },
          {
            "name": "burst_limit_vs_unlimited",
            "value": 50404292,
            "range": "± 40709",
            "unit": "ns/iter"
          },
          {
            "name": "four_control_flooders_vs_data",
            "value": 60598909,
            "range": "± 966344",
            "unit": "ns/iter"
          },
          {
            "name": "slow_receiver_backpressure",
            "value": 63333595,
            "range": "± 295259",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/10",
            "value": 21952,
            "range": "± 3070",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/32",
            "value": 22242,
            "range": "± 2600",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/100",
            "value": 22288,
            "range": "± 3294",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_under_data_flood",
            "value": 966,
            "range": "± 377",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/10",
            "value": 21898,
            "range": "± 3274",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/32",
            "value": 21890,
            "range": "± 3396",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/100",
            "value": 22544,
            "range": "± 2575",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_under_control_flood",
            "value": 1171,
            "range": "± 3476",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/1000",
            "value": 16955,
            "range": "± 70",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/1000",
            "value": 16532,
            "range": "± 76",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/10000",
            "value": 188543,
            "range": "± 928",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/10000",
            "value": 186155,
            "range": "± 638",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/100000",
            "value": 1766986,
            "range": "± 14318",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/100000",
            "value": 1739788,
            "range": "± 11865",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/1000",
            "value": 149723,
            "range": "± 2458",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/1000",
            "value": 32213,
            "range": "± 148",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/10000",
            "value": 963283,
            "range": "± 19688",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/10000",
            "value": 350044,
            "range": "± 798",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/100000",
            "value": 8634277,
            "range": "± 359143",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/100000",
            "value": 3335051,
            "range": "± 8933",
            "unit": "ns/iter"
          },
          {
            "name": "priority/data_uncontended",
            "value": 24068,
            "range": "± 42",
            "unit": "ns/iter"
          },
          {
            "name": "priority/data_under_control_flood",
            "value": 31476,
            "range": "± 298",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/10000",
            "value": 183545,
            "range": "± 3843",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/10000",
            "value": 105909,
            "range": "± 1703",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/100000",
            "value": 1525410,
            "range": "± 33612",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/100000",
            "value": 595600,
            "range": "± 3479",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/1000000",
            "value": 14802606,
            "range": "± 255274",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/1000000",
            "value": 5605327,
            "range": "± 207440",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/2",
            "value": 1588060,
            "range": "± 53547",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/2",
            "value": 780770,
            "range": "± 6764",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/4",
            "value": 4884670,
            "range": "± 113568",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/4",
            "value": 1447350,
            "range": "± 34532",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/8",
            "value": 10064499,
            "range": "± 212865",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/8",
            "value": 2795306,
            "range": "± 377247",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_ns/mpsc",
            "value": 14,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_ns/spsc",
            "value": 7,
            "range": "± 0",
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
            "value": 4,
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
            "value": 4,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "roundtrip_latency/mpsc",
            "value": 23379,
            "range": "± 3488",
            "unit": "ns/iter"
          },
          {
            "name": "roundtrip_latency/spsc",
            "value": 917,
            "range": "± 118",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/1000",
            "value": 76286,
            "range": "± 3203",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/10000",
            "value": 277150,
            "range": "± 2812",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/100000",
            "value": 2360445,
            "range": "± 16072",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/100",
            "value": 77632,
            "range": "± 1896",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/1000",
            "value": 111993,
            "range": "± 9354",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/10000",
            "value": 536425,
            "range": "± 24181",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_lanes_10k_each",
            "value": 659797,
            "range": "± 10934",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/1024",
            "value": 16842,
            "range": "± 112",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/1024",
            "value": 22687,
            "range": "± 63",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/1024",
            "value": 17335,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/1024",
            "value": 13394,
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/1024",
            "value": 11566,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/1024",
            "value": 17429,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/4096",
            "value": 66126,
            "range": "± 221",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/4096",
            "value": 87715,
            "range": "± 2750",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/4096",
            "value": 65463,
            "range": "± 128",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/4096",
            "value": 50906,
            "range": "± 67",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/4096",
            "value": 44090,
            "range": "± 128",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/4096",
            "value": 66392,
            "range": "± 180",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/16384",
            "value": 261733,
            "range": "± 1246",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/16384",
            "value": 339147,
            "range": "± 964",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/16384",
            "value": 253099,
            "range": "± 642",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/16384",
            "value": 197975,
            "range": "± 1967",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/16384",
            "value": 173337,
            "range": "± 937",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/16384",
            "value": 271934,
            "range": "± 7679",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/65536",
            "value": 1064864,
            "range": "± 2549",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/65536",
            "value": 1363630,
            "range": "± 4330",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/65536",
            "value": 1021729,
            "range": "± 3427",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/65536",
            "value": 812045,
            "range": "± 1331",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/65536",
            "value": 716295,
            "range": "± 1989",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/65536",
            "value": 1070922,
            "range": "± 2438",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/alt_screen_random_write",
            "value": 817074,
            "range": "± 4589",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/scrolling",
            "value": 1090741,
            "range": "± 5783",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/unicode_random_write",
            "value": 727944,
            "range": "± 3909",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/osc_heavy",
            "value": 898032,
            "range": "± 2486",
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
            "value": 17,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_vec_size_1000",
            "value": 1258,
            "range": "± 32",
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
            "value": 10937,
            "range": "± 39",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/O_quadratic",
            "value": 109295,
            "range": "± 871",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/S_complex",
            "value": 171885,
            "range": "± 1408",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/5",
            "value": 795977,
            "range": "± 4603",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/10",
            "value": 4476614,
            "range": "± 19595",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/26",
            "value": 62376931,
            "range": "± 191965",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/50",
            "value": 118034858,
            "range": "± 362952",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/uncached_HELLO",
            "value": 606926,
            "range": "± 5270",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/cached_HELLO",
            "value": 68368,
            "range": "± 1439",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/cache_warmup_alphabet",
            "value": 5424156,
            "range": "± 79982",
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
            "value": 102804,
            "range": "± 120",
            "unit": "ns/iter"
          },
          {
            "name": "jit_collapse_call_overhead/one_2d_collapse_call/4",
            "value": 59130,
            "range": "± 280",
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
            "value": 11,
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
          "id": "5e201258e643d63fdc6afca64236babf1e992fc7",
          "message": "refactor(actor-scheduler): delete supervision and shedding; the crate owns send policy (#968)\n\n## Summary\n\nThree commits removing machinery that described capabilities the runtime\ndoes not have. **Net ≈ −1000 lines.**\n\n### 1. Supervision deleted; handler failures panic\nThere was never a supervisor. `Exit::Failed`,\n`HandlerError::Recoverable`, `Supervision`/`Stuck`/`NodeId`,\n`Host::exits()`, `HostOut::supervision`, `EngineControl::GreenStuck` —\nall scaffolding for one that was never built, and with `panic = \"abort\"`\nin every profile a \"recoverable\" error could never have been recovered\nfrom: nothing unwinds, nothing restarts, nothing resumes.\n\n`lifecycle.rs` goes entirely (with failures panicking, `Exit` has one\ninhabitant, which is `()`); `HandlerError` collapses to a wrapper with\none panic site; `Step::Disconnected` goes, and with it **`GreenThread`'s\nwiring type parameter** — `HostOut` now carries only its continuation,\nwhich never reaches a `Wiring`.\n\n### 2. The backpressure protocol, written down\nNew: `docs/designs/actor-scheduler-backpressure.md`. The send path looks\nlike it's made of local decisions and isn't — every \"this queue is full,\nwhat should *this* caller do?\" has a local answer that's individually\nplausible and collectively fatal, which is why shedding, spinning, and\nunbounded blocking have each been written and reverted here before.\nStates the invariant once (everyone backs off on the same ladder; the\nproperties hold only because that's true of everyone) and derives the\nconsequences: the heaviest sender is provably the one timed out *with no\nper-sender accounting anywhere*, because the space light senders find\nwas freed while the heavy one slept; a backing-off sender has yielded\nits slot rather than gone idle; and ~8.6s is a liveness assertion three\norders of magnitude past any live consumer's drain time, so blowing it\nis a diagnosis, not backpressure.\n\n### 3. Shedding deleted; `try_send` privatized\n`Delivery::Droppable`: zero production users, ever. `Topology` (which it\nserves as cycle-closer): zero. The macro's `[drop]`: zero. The one place\nproduction shed hand-rolled it through public `try_send`, bypassing the\nabstraction meant to govern it.\n\n`try_send` was public only because cross-crate `Wiring` impls needed it\n— and they needed it because the crate never gave them an\n`ActorHandle`/`GreenSender` analogue of `send_port`. One `PortTarget`\ntrait (one method, three impls) lets `send_port` serve all three, both\n`try_send`s drop to `pub(crate)`, and the hand-rolled `TrySendError`\nmatch arms delete themselves. **The caller names a target; the crate\nnames the policy.**\n\n**Deliberate behavior change:** the FPS-telemetry port parks instead of\nshedding. Its ring holds 128 and samples arrive one per presented frame,\nso a full ring means vsync hasn't run in ~2s — a wedged node, not a\ntransient.\n\n## Known risk (deliberate, not papered over)\n\nA gone flush target panics unconditionally, and the quit cascade sends\n`Shutdown` to the green host before the driver without synchronizing\nthem. If the driver's receiver drops before the host stops stepping its\nnodes, a coordinator flush would panic during an otherwise clean quit.\nNo test hits it — including every shutdown-cascade path — but that's\nevidence, not proof. The honest fix is ordering the cascade so the host\nprovably drains first, not a \"shutting down\" flag.\n\n## Open question for review\n\n`Topology` has zero production users and its DAG rule is unexercised.\nDeleting it would be consistent with everything else here; keeping it is\ndefensible as the design's stated answer to artificial deadlock, at zero\nruntime cost. Deliberately not decided.\n\n## Test plan\n\n- [x] `cargo test --workspace` — 1692 passed, 0 failed\n- [x] `cargo fmt --check` and `cargo clippy --workspace --all-targets --\n-D warnings` — clean\n- [x] 4 tests converted to `#[should_panic]` rather than deleted; 5\ndeleted whose behavior no longer exists\n\n🤖 Generated with [Claude Code](https://claude.ai/code)\n\nhttps://claude.ai/code/session_015Y5cNzR9PnASfGfjBh5kJc\n\n---------\n\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-07-30T05:07:24Z",
          "tree_id": "91507c7d1672b055cf253c6acb83651eef73b3c5",
          "url": "https://github.com/jppittman/pixelflow/commit/5e201258e643d63fdc6afca64236babf1e992fc7"
        },
        "date": 1785391732278,
        "tool": "cargo",
        "benches": [
          {
            "name": "data_throughput_under_control_flood",
            "value": 60531574,
            "range": "± 43794",
            "unit": "ns/iter"
          },
          {
            "name": "burst_limit_vs_unlimited",
            "value": 50426085,
            "range": "± 46649",
            "unit": "ns/iter"
          },
          {
            "name": "four_control_flooders_vs_data",
            "value": 60651758,
            "range": "± 967086",
            "unit": "ns/iter"
          },
          {
            "name": "slow_receiver_backpressure",
            "value": 63487149,
            "range": "± 518340",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/10",
            "value": 21688,
            "range": "± 3459",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/32",
            "value": 22397,
            "range": "± 3516",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/100",
            "value": 22117,
            "range": "± 2920",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_under_data_flood",
            "value": 1128,
            "range": "± 258",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/10",
            "value": 22211,
            "range": "± 3156",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/32",
            "value": 22621,
            "range": "± 3226",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/100",
            "value": 21990,
            "range": "± 3124",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_under_control_flood",
            "value": 2229,
            "range": "± 2701",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/1000",
            "value": 17194,
            "range": "± 67",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/1000",
            "value": 17315,
            "range": "± 79",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/10000",
            "value": 190601,
            "range": "± 6259",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/10000",
            "value": 194194,
            "range": "± 944",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/100000",
            "value": 1781379,
            "range": "± 25023",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/100000",
            "value": 1824181,
            "range": "± 23074",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/1000",
            "value": 160574,
            "range": "± 4248",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/1000",
            "value": 29411,
            "range": "± 173",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/10000",
            "value": 942236,
            "range": "± 22150",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/10000",
            "value": 346286,
            "range": "± 1622",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/100000",
            "value": 8479296,
            "range": "± 378201",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/100000",
            "value": 3056582,
            "range": "± 57662",
            "unit": "ns/iter"
          },
          {
            "name": "priority/data_uncontended",
            "value": 25934,
            "range": "± 614",
            "unit": "ns/iter"
          },
          {
            "name": "priority/data_under_control_flood",
            "value": 32111,
            "range": "± 422",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/10000",
            "value": 185498,
            "range": "± 4691",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/10000",
            "value": 98279,
            "range": "± 5035",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/100000",
            "value": 1524864,
            "range": "± 25748",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/100000",
            "value": 537470,
            "range": "± 14439",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/1000000",
            "value": 15442093,
            "range": "± 234315",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/1000000",
            "value": 4442625,
            "range": "± 55214",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/2",
            "value": 1671696,
            "range": "± 37981",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/2",
            "value": 790501,
            "range": "± 14118",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/4",
            "value": 5239958,
            "range": "± 129104",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/4",
            "value": 1484889,
            "range": "± 13883",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/8",
            "value": 10289685,
            "range": "± 387979",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/8",
            "value": 2861645,
            "range": "± 291558",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_ns/mpsc",
            "value": 14,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_ns/spsc",
            "value": 7,
            "range": "± 0",
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
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_contended/mpsc/4",
            "value": 14,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_contended/sharded_spsc/4",
            "value": 3,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "roundtrip_latency/mpsc",
            "value": 25237,
            "range": "± 4106",
            "unit": "ns/iter"
          },
          {
            "name": "roundtrip_latency/spsc",
            "value": 863,
            "range": "± 82",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/1000",
            "value": 81760,
            "range": "± 2706",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/10000",
            "value": 253125,
            "range": "± 13759",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/100000",
            "value": 1971734,
            "range": "± 21065",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/100",
            "value": 82262,
            "range": "± 3542",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/1000",
            "value": 125301,
            "range": "± 8883",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/10000",
            "value": 648062,
            "range": "± 26449",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_lanes_10k_each",
            "value": 675090,
            "range": "± 13254",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/1024",
            "value": 16454,
            "range": "± 36",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/1024",
            "value": 22711,
            "range": "± 450",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/1024",
            "value": 17272,
            "range": "± 55",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/1024",
            "value": 12988,
            "range": "± 97",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/1024",
            "value": 11521,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/1024",
            "value": 17155,
            "range": "± 55",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/4096",
            "value": 64973,
            "range": "± 985",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/4096",
            "value": 88290,
            "range": "± 1790",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/4096",
            "value": 65020,
            "range": "± 215",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/4096",
            "value": 50474,
            "range": "± 358",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/4096",
            "value": 43717,
            "range": "± 97",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/4096",
            "value": 65676,
            "range": "± 247",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/16384",
            "value": 258450,
            "range": "± 3051",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/16384",
            "value": 339815,
            "range": "± 4573",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/16384",
            "value": 254457,
            "range": "± 922",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/16384",
            "value": 199627,
            "range": "± 881",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/16384",
            "value": 172454,
            "range": "± 1426",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/16384",
            "value": 266519,
            "range": "± 973",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/65536",
            "value": 1046598,
            "range": "± 8367",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/65536",
            "value": 1349952,
            "range": "± 10129",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/65536",
            "value": 1018955,
            "range": "± 9641",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/65536",
            "value": 812748,
            "range": "± 10099",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/65536",
            "value": 708594,
            "range": "± 11388",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/65536",
            "value": 1050702,
            "range": "± 3689",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/alt_screen_random_write",
            "value": 818388,
            "range": "± 6556",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/scrolling",
            "value": 1053092,
            "range": "± 23826",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/unicode_random_write",
            "value": 716466,
            "range": "± 5797",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/osc_heavy",
            "value": 900310,
            "range": "± 11796",
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
            "value": 18,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_vec_size_1000",
            "value": 1259,
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
            "value": 10961,
            "range": "± 166",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/O_quadratic",
            "value": 106189,
            "range": "± 1863",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/S_complex",
            "value": 168483,
            "range": "± 2251",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/5",
            "value": 787677,
            "range": "± 12348",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/10",
            "value": 4484865,
            "range": "± 22639",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/26",
            "value": 62771117,
            "range": "± 324825",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/50",
            "value": 118511436,
            "range": "± 513771",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/uncached_HELLO",
            "value": 611548,
            "range": "± 4584",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/cached_HELLO",
            "value": 78165,
            "range": "± 444",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/cache_warmup_alphabet",
            "value": 5734532,
            "range": "± 154450",
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
            "value": 102755,
            "range": "± 178",
            "unit": "ns/iter"
          },
          {
            "name": "jit_collapse_call_overhead/one_2d_collapse_call/4",
            "value": 64708,
            "range": "± 261",
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
          "id": "b916f44d090076ba00a8cb782e9d2056c8447cf4",
          "message": "feat(render): kernel-native frame lane with a collapse-baked cell grid (#966)\n\nThe next cut after #956 and the 2D collapse (#961): the render path\nitself goes through the JIT's internal loop. The rasterizer's input is\nnow `pixelflow_graphics::render::scene::Scene`, with two lanes:\n\n- **`Scene::Surface`** — an arbitrary color manifold, dense per-batch\nevaluation through the `Manifold` trait. The generic lane; the runtime\nexamples and core-term's no-snapshot background use it, and the\ncoordinator's DPI contramap applies here.\n- **`Scene::CellGrid`** — a `CellGridFrame` rendered by **one\n2D-collapse call per channel per stripe**: the pixel loop lives inside\nthe JIT with both LICM prologue scopes active (per-call and per-row),\nthen the four planes pack via `Pixel::from_rgba`, stripe-parallel across\nthe worker pool. No per-batch `extern \"C\"` boundary, no virtual\ndispatch, no combinator evaluation.\n\n## What changed\n\n- **pixelflow-core**: `CellGridProgram` compiles collapse-mode kernels\nonly; `CellGridFrame::bake_channel_rows` bakes a channel plane at\npadded-batch stride (one `call_collapse` per invocation). The per-batch\n`CellGridChannel` manifolds are **deleted** — one implementation per\njob. Cell-grid tests re-anchor on the plane-bake path; the\nmisaligned-sample apron test moves to a fractional density probe (the\nbake API owns the pixel-center origin).\n- **pixelflow-graphics**: `render::scene::Scene` + the stripe-parallel\nbake-and-pack; `RenderRequest.manifold` → `RenderRequest.scene`. New\ntests: packed-channel blending against scalar oracles, and thread-count\ninvariance of the stripe split.\n- **pixelflow-runtime**: the coordinator's `Scene` alias becomes the\ngraphics `Scene`; the DPI contramap (`At { x: X·points/px }`) wraps only\n`Surface` scenes — **`Scene::CellGrid` is device-pixel space by\ncontract**, carrying its scale in its geometry.\n`AppData::RenderSurfaceU32` deleted per subtract-before-add: documented\nas u32-coordinate rendering, treated identically to `RenderSurface` by\nthe engine, zero senders.\n- **core-term**: `build_scene` returns `Scene::CellGrid` with\ndevice-pixel geometry (cell extents × density; atlas texels ≡ device\npixels, sample density 1). The `At`+`ColorCube` frame-root composition\nis gone — the terminal's render path now contains no combinator\nevaluation at all.\n\n## Testing\n\n- pixelflow-core `cell_grid`: 7/7 on the plane-bake path (lossless\naligned reads, fractional-density apron fade, neighbor isolation,\nout-of-grid background, oversized-buffer refusal).\n- pixelflow-graphics: full lib suite (149) including the new scene\ntests; rasterizer actor suite on the `Scene` shape.\n- pixelflow-runtime: full suite including the coordinator's 1:1/Retina\ncontramap identity tests restated on `Scene::Surface`.\n- core-term: full suite; the end-to-end scene test now renders a blank\nscreen to the default background **through the collapse lane** and still\nproves resize-recompiles.\n- Full workspace test + clippy `-D warnings` run in flight at time of\nopening; will confirm on this PR.\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nhttps://claude.ai/code/session_01PHPBWpn4HqSDpo1DoqcTWW\n\n---\n_Generated by [Claude\nCode](https://claude.ai/code/session_01PHPBWpn4HqSDpo1DoqcTWW)_\n\n---------\n\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-07-30T14:28:04-07:00",
          "tree_id": "ef0aa344799838c70858eaeb50e610bf8be12162",
          "url": "https://github.com/jppittman/pixelflow/commit/b916f44d090076ba00a8cb782e9d2056c8447cf4"
        },
        "date": 1785450566615,
        "tool": "cargo",
        "benches": [
          {
            "name": "data_throughput_under_control_flood",
            "value": 60412931,
            "range": "± 27467",
            "unit": "ns/iter"
          },
          {
            "name": "burst_limit_vs_unlimited",
            "value": 50329430,
            "range": "± 35600",
            "unit": "ns/iter"
          },
          {
            "name": "four_control_flooders_vs_data",
            "value": 60593503,
            "range": "± 331234",
            "unit": "ns/iter"
          },
          {
            "name": "slow_receiver_backpressure",
            "value": 63369828,
            "range": "± 207310",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/10",
            "value": 21463,
            "range": "± 3093",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/32",
            "value": 22216,
            "range": "± 3074",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/100",
            "value": 21927,
            "range": "± 3432",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_under_data_flood",
            "value": 1312,
            "range": "± 506",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/10",
            "value": 22039,
            "range": "± 3234",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/32",
            "value": 22260,
            "range": "± 3302",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/100",
            "value": 21779,
            "range": "± 3008",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_under_control_flood",
            "value": 10158,
            "range": "± 3434",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/1000",
            "value": 17391,
            "range": "± 71",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/1000",
            "value": 16812,
            "range": "± 393",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/10000",
            "value": 194008,
            "range": "± 935",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/10000",
            "value": 189345,
            "range": "± 457",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/100000",
            "value": 1801896,
            "range": "± 19332",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/100000",
            "value": 1810437,
            "range": "± 27091",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/1000",
            "value": 169503,
            "range": "± 1978",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/1000",
            "value": 30931,
            "range": "± 79",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/10000",
            "value": 974146,
            "range": "± 27841",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/10000",
            "value": 364237,
            "range": "± 1356",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/100000",
            "value": 8799059,
            "range": "± 371675",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/100000",
            "value": 3199639,
            "range": "± 17506",
            "unit": "ns/iter"
          },
          {
            "name": "priority/data_uncontended",
            "value": 23454,
            "range": "± 518",
            "unit": "ns/iter"
          },
          {
            "name": "priority/data_under_control_flood",
            "value": 31178,
            "range": "± 48",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/10000",
            "value": 181896,
            "range": "± 3695",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/10000",
            "value": 106458,
            "range": "± 1843",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/100000",
            "value": 1445331,
            "range": "± 19053",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/100000",
            "value": 590618,
            "range": "± 6293",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/1000000",
            "value": 14165179,
            "range": "± 128092",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/1000000",
            "value": 5515549,
            "range": "± 24021",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/2",
            "value": 1537462,
            "range": "± 35639",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/2",
            "value": 782276,
            "range": "± 6894",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/4",
            "value": 4814987,
            "range": "± 92577",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/4",
            "value": 1460319,
            "range": "± 19772",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/8",
            "value": 9141791,
            "range": "± 290158",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/8",
            "value": 2765772,
            "range": "± 331031",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_ns/mpsc",
            "value": 14,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_ns/spsc",
            "value": 7,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_contended/mpsc/2",
            "value": 11,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_contended/sharded_spsc/2",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_contended/mpsc/4",
            "value": 22,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_contended/sharded_spsc/4",
            "value": 5,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "roundtrip_latency/mpsc",
            "value": 23832,
            "range": "± 4026",
            "unit": "ns/iter"
          },
          {
            "name": "roundtrip_latency/spsc",
            "value": 940,
            "range": "± 110",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/1000",
            "value": 79517,
            "range": "± 2751",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/10000",
            "value": 441809,
            "range": "± 14194",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/100000",
            "value": 4107865,
            "range": "± 190077",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/100",
            "value": 79319,
            "range": "± 1980",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/1000",
            "value": 115623,
            "range": "± 7400",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/10000",
            "value": 648674,
            "range": "± 26734",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_lanes_10k_each",
            "value": 655511,
            "range": "± 11905",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/1024",
            "value": 16766,
            "range": "± 105",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/1024",
            "value": 22638,
            "range": "± 695",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/1024",
            "value": 17319,
            "range": "± 438",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/1024",
            "value": 13264,
            "range": "± 56",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/1024",
            "value": 11473,
            "range": "± 52",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/1024",
            "value": 17170,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/4096",
            "value": 65957,
            "range": "± 9360",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/4096",
            "value": 86687,
            "range": "± 180",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/4096",
            "value": 64716,
            "range": "± 112",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/4096",
            "value": 50249,
            "range": "± 643",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/4096",
            "value": 43942,
            "range": "± 74",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/4096",
            "value": 66214,
            "range": "± 142",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/16384",
            "value": 261255,
            "range": "± 6579",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/16384",
            "value": 340034,
            "range": "± 2381",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/16384",
            "value": 253385,
            "range": "± 504",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/16384",
            "value": 198560,
            "range": "± 22312",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/16384",
            "value": 173388,
            "range": "± 8613",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/16384",
            "value": 270986,
            "range": "± 639",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/65536",
            "value": 1058013,
            "range": "± 3713",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/65536",
            "value": 1355196,
            "range": "± 2387",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/65536",
            "value": 1019444,
            "range": "± 2142",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/65536",
            "value": 807822,
            "range": "± 9008",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/65536",
            "value": 716811,
            "range": "± 3948",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/65536",
            "value": 1063792,
            "range": "± 15604",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/alt_screen_random_write",
            "value": 818478,
            "range": "± 8326",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/scrolling",
            "value": 1100050,
            "range": "± 10041",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/unicode_random_write",
            "value": 744585,
            "range": "± 7814",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/osc_heavy",
            "value": 899275,
            "range": "± 13535",
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
            "value": 18,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_vec_size_1000",
            "value": 1258,
            "range": "± 21",
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
            "range": "± 38",
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
            "value": 11186,
            "range": "± 162",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/O_quadratic",
            "value": 106604,
            "range": "± 742",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/S_complex",
            "value": 169016,
            "range": "± 913",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/5",
            "value": 788221,
            "range": "± 3319",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/10",
            "value": 4461242,
            "range": "± 8740",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/26",
            "value": 63225378,
            "range": "± 150327",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/50",
            "value": 120047604,
            "range": "± 354884",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/uncached_HELLO",
            "value": 621364,
            "range": "± 4994",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/cached_HELLO",
            "value": 69086,
            "range": "± 2043",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/cache_warmup_alphabet",
            "value": 5525259,
            "range": "± 84760",
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
            "value": 102712,
            "range": "± 533",
            "unit": "ns/iter"
          },
          {
            "name": "jit_collapse_call_overhead/one_2d_collapse_call/4",
            "value": 64612,
            "range": "± 306",
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
          "id": "6592d4be3723d4f2e3d6e39c351edb19357f263a",
          "message": "fix(pixelflow-runtime): remove bit-rotted headless driver; document/fix WASM gaps (#970)\n\n## Summary\n\nStarted as one specific case of \"green CI but the change wasn't actually\nsafe\" and grew into a broader audit: every place CI's\n`--all-features`/single-target-matrix habit could paper over a code path\nnobody actually exercises. Found and fixed four real bugs this way,\nclosed one class of the underlying failure mode with a blocking check,\nand added visibility for the two gaps too deep to fix blind.\n\n### 1. Headless driver — dead code, deleted\n\n`display/drivers/headless.rs` was gated behind\n`#[cfg(use_headless_display)]`. Driver-selection priority in `build.rs`\nmeans X11 always wins on Linux and Cocoa always wins on macOS — every CI\nrunner — so this cfg has never once been active in CI. Building it\ndirectly (`DISPLAY_DRIVER=headless cargo check -p pixelflow-runtime`)\nproduced **22 compile errors**: it imports `pixelflow_render`, a crate\nrenamed to `pixelflow-graphics` long ago, plus several removed message\ntypes. Deleted the driver, its Cargo features, and the build.rs branches\nthat could select it; made the fallback panic loudly instead of silently\nresolving to a driver that can't compile. (Confirmed with the repo owner\nbefore deleting.)\n\n### 2. Structural guard against that exact failure recurring\n\nAdded a **blocking** `driver-cfg-coverage` job\n(`scripts/check-driver-cfg-coverage.sh`) that fails CI the moment a new\n`use_*_display` cfg is introduced without a matching entry recording how\nit's covered. This is the mechanism, not just the fix: the headless\ndriver could rot silently because nothing asserted every cfg gate has an\nowner; now something does.\n\n### 3. Web/WASM driver — real code, but nothing in the chain supports\nwasm32 yet\n\nChecking the same way for `DISPLAY_DRIVER=web` turned up two more\nissues, neither caught by CI since no job ever targets\n`wasm32-unknown-unknown`:\n- `pixelflow-core` → `pixelflow-search` → `rand`/`getrandom` needs the\n`js` feature to build for wasm32 at all. **Fixed.**\n- `pixelflow-ir`'s JIT emitter has no wasm32 backend at all — a real,\nsubstantial gap, not something to paper over. Added a **non-blocking**\n`wasm32-status` job that reports this honestly instead of continuing to\nclaim Web WASM is on equal footing with Cocoa/X11.\n\n### 4. `pixelflow-graphics`'s `serde` feature — broken in isolation,\ninvisible under `--all-features`\n\nAuditing every feature-bearing crate individually (`cargo check -p\n<crate> --no-default-features --features <each-feature>`) rather than\nrelying on workspace `--all-features` found that `pixelflow-graphics\n--features serde` doesn't compile alone: `bitflags`'s `serde` feature\nwas never wired up to fire when `pixelflow-graphics`'s own `serde`\nfeature is enabled. Cargo's feature unification papered over it in every\nexisting CI job, since something else in the workspace always turned\n`bitflags/serde` on anyway. **Fixed** — one line in\n`pixelflow-graphics/Cargo.toml`.\n\n### 5. Two deeper gaps — too architecturally unfamiliar to fix blind,\nmade visible instead\n\n- `pixelflow-ir`'s `no_std` support: doesn't compile (218 errors) —\nnever exercised by any CI job.\n- `pixelflow-search`'s `std`-off build: doesn't compile — same story.\n\nBoth are surfaced by a new **non-blocking** `std-off-status` job rather\nthan fixed outright — I don't have enough context on the intended no_std\narchitecture to guess at fixes without risking a plausible-looking but\nwrong change. Tracked and visible beats silently dark.\n\n### 6. Dependency-advisory scanning — nothing checked this at all\n\nAdded `cargo-deny` (advisories only — `licenses`/`bans`/`sources`\ndeliberately out of scope, see `deny.toml`'s header comment) as a new\n**blocking** `Dependency advisories` job. It immediately found real,\nlive RUSTSEC advisories in the existing dependency tree:\n- `rand` 0.8.5 and 0.9.2 — both in unsound version ranges\n(RUSTSEC-2026-0097). Bumped to 0.8.6 / 0.9.3.\n- `anyhow` 1.0.98 — unsound `downcast_mut` (RUSTSEC-2026-0190). Bumped\nto 1.0.104.\n- `crossbeam-epoch` 0.9.18 — pointer-format vulnerability\n(RUSTSEC-2026-0204). Bumped to 0.9.20.\n- `spin` 0.10.0 — yanked. Bumped to 0.10.1.\n- `bincode` 1.3.3 — unmaintained, no safe upgrade; only reachable via\nthe wasm32 IPC path that doesn't build yet regardless. Documented and\nignored in `deny.toml`.\n- `quick-xml` 0.26.0 (via `inferno`→`pprof`) — two DoS advisories\n(RUSTSEC-2026-0194/0195) that appeared in the advisory database *during\nthis PR*, caught on the very next push. No upgrade path exists (`pprof`\n0.15.0, latest, hard-pins `inferno ^0.11`, which hard-pins `quick-xml\n^0.26`); the affected code only parses profiling data the process\nproduces itself, behind the optional non-default `profiling` feature, so\nthe advisories' actual threat model (untrusted network XML) doesn't\napply. Documented and ignored in `deny.toml`.\n\n## Changes\n\n- Delete `pixelflow-runtime/src/display/drivers/headless.rs` and all\nwiring/features/build.rs branches for it.\n- Add `scripts/check-driver-cfg-coverage.sh` + blocking\n`driver-cfg-coverage` CI job.\n- Add non-blocking `wasm32-status` and `std-off-status` CI jobs.\n- Fix `pixelflow-search/Cargo.toml` (`getrandom` wasm32 `js` feature)\nand `pixelflow-graphics/Cargo.toml` (`bitflags/serde` wiring).\n- Add `deny.toml` + blocking `dependency-advisories` CI job; bump\n`rand`, `anyhow`, `crossbeam-epoch`, `spin` to patched versions;\ndocument two intentionally-ignored advisories with no viable fix.\n- Update `CLAUDE.md` and `.claude/agents/pixelflow-runtime.md` to\ndescribe driver status accurately.\n\n## Test plan\n\n- [x] `cargo build --workspace` (default features) succeeds\n- [x] `cargo clippy --workspace --all-targets --all-features -- -D\nwarnings` clean\n- [x] `cargo fmt --all -- --check` clean\n- [x] `cargo nextest run` across affected crates with `--all-features` —\nall passing\n- [x] `cargo deny check advisories` clean (two ignores documented with\njustification)\n- [x] Verified no remaining references to the deleted headless driver\nanywhere in the tree\n- [x] **All 12 CI checks green on the final commit (945b95a)**: CL\nmetadata, Rustfmt, Clippy, Cross-target ABI (aarch64), Test on\nubuntu-latest, Test on macos-latest, Behavior contracts, ISA matrix\n(SSE2/AVX2/AVX-512), Display driver CI coverage, Dependency advisories,\nWASM target status (informational), no_std/std-off status\n(informational)\n\n---------\n\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-07-31T15:28:14-07:00",
          "tree_id": "d6bdfc4a4f8c2d57779f5813a97f7c32fc91b399",
          "url": "https://github.com/jppittman/pixelflow/commit/6592d4be3723d4f2e3d6e39c351edb19357f263a"
        },
        "date": 1785540432099,
        "tool": "cargo",
        "benches": [
          {
            "name": "data_throughput_under_control_flood",
            "value": 60523577,
            "range": "± 31853",
            "unit": "ns/iter"
          },
          {
            "name": "burst_limit_vs_unlimited",
            "value": 50428627,
            "range": "± 40215",
            "unit": "ns/iter"
          },
          {
            "name": "four_control_flooders_vs_data",
            "value": 60612107,
            "range": "± 173270",
            "unit": "ns/iter"
          },
          {
            "name": "slow_receiver_backpressure",
            "value": 63536586,
            "range": "± 209342",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/10",
            "value": 23088,
            "range": "± 3094",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/32",
            "value": 23493,
            "range": "± 3218",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/100",
            "value": 23175,
            "range": "± 3573",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_under_data_flood",
            "value": 1900,
            "range": "± 827",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/10",
            "value": 23732,
            "range": "± 3395",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/32",
            "value": 23151,
            "range": "± 3635",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/100",
            "value": 22948,
            "range": "± 2844",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_under_control_flood",
            "value": 1523,
            "range": "± 3477",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/1000",
            "value": 15609,
            "range": "± 149",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/1000",
            "value": 15383,
            "range": "± 235",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/10000",
            "value": 163645,
            "range": "± 583",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/10000",
            "value": 158450,
            "range": "± 1201",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/100000",
            "value": 1571903,
            "range": "± 16074",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/100000",
            "value": 1521084,
            "range": "± 25076",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/1000",
            "value": 171296,
            "range": "± 3722",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/1000",
            "value": 28458,
            "range": "± 152",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/10000",
            "value": 1031165,
            "range": "± 34935",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/10000",
            "value": 295735,
            "range": "± 2690",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/100000",
            "value": 8760984,
            "range": "± 558752",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/100000",
            "value": 2878217,
            "range": "± 15481",
            "unit": "ns/iter"
          },
          {
            "name": "priority/data_uncontended",
            "value": 25176,
            "range": "± 78",
            "unit": "ns/iter"
          },
          {
            "name": "priority/data_under_control_flood",
            "value": 33381,
            "range": "± 693",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/10000",
            "value": 187776,
            "range": "± 2716",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/10000",
            "value": 109199,
            "range": "± 1899",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/100000",
            "value": 1452395,
            "range": "± 29829",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/100000",
            "value": 465658,
            "range": "± 4589",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/1000000",
            "value": 14061480,
            "range": "± 112670",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/1000000",
            "value": 3694757,
            "range": "± 81466",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/2",
            "value": 1767162,
            "range": "± 59711",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/2",
            "value": 778393,
            "range": "± 7739",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/4",
            "value": 5303854,
            "range": "± 153591",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/4",
            "value": 1473381,
            "range": "± 21467",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/8",
            "value": 10555830,
            "range": "± 428619",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/8",
            "value": 2835692,
            "range": "± 243251",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_ns/mpsc",
            "value": 14,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_ns/spsc",
            "value": 8,
            "range": "± 0",
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
            "value": 6,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "roundtrip_latency/mpsc",
            "value": 24978,
            "range": "± 3405",
            "unit": "ns/iter"
          },
          {
            "name": "roundtrip_latency/spsc",
            "value": 958,
            "range": "± 152",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/1000",
            "value": 80752,
            "range": "± 3108",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/10000",
            "value": 287283,
            "range": "± 4877",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/100000",
            "value": 2359112,
            "range": "± 14894",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/100",
            "value": 83608,
            "range": "± 4069",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/1000",
            "value": 117148,
            "range": "± 22583",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/10000",
            "value": 514887,
            "range": "± 27857",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_lanes_10k_each",
            "value": 675233,
            "range": "± 25843",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/1024",
            "value": 17151,
            "range": "± 245",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/1024",
            "value": 21593,
            "range": "± 60",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/1024",
            "value": 16326,
            "range": "± 59",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/1024",
            "value": 12396,
            "range": "± 122",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/1024",
            "value": 11349,
            "range": "± 153",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/1024",
            "value": 17620,
            "range": "± 152",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/4096",
            "value": 67640,
            "range": "± 402",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/4096",
            "value": 84180,
            "range": "± 815",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/4096",
            "value": 62233,
            "range": "± 540",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/4096",
            "value": 46857,
            "range": "± 356",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/4096",
            "value": 42962,
            "range": "± 334",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/4096",
            "value": 67986,
            "range": "± 429",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/16384",
            "value": 268690,
            "range": "± 2456",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/16384",
            "value": 329242,
            "range": "± 1409",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/16384",
            "value": 239634,
            "range": "± 1112",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/16384",
            "value": 184553,
            "range": "± 1209",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/16384",
            "value": 170170,
            "range": "± 2201",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/16384",
            "value": 270972,
            "range": "± 2520",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/65536",
            "value": 1074745,
            "range": "± 8335",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/65536",
            "value": 1277848,
            "range": "± 2909",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/65536",
            "value": 950420,
            "range": "± 3036",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/65536",
            "value": 750404,
            "range": "± 8240",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/65536",
            "value": 683879,
            "range": "± 4682",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/65536",
            "value": 1094193,
            "range": "± 11185",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/alt_screen_random_write",
            "value": 762492,
            "range": "± 2893",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/scrolling",
            "value": 1061297,
            "range": "± 22537",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/unicode_random_write",
            "value": 676138,
            "range": "± 7496",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/osc_heavy",
            "value": 847994,
            "range": "± 2746",
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
            "range": "± 1",
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
            "value": 18,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_vec_size_1000",
            "value": 1258,
            "range": "± 33",
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
            "value": 10493,
            "range": "± 175",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/O_quadratic",
            "value": 105467,
            "range": "± 784",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/S_complex",
            "value": 167560,
            "range": "± 1373",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/5",
            "value": 813825,
            "range": "± 4892",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/10",
            "value": 4471797,
            "range": "± 13582",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/26",
            "value": 62620642,
            "range": "± 306255",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/50",
            "value": 118439638,
            "range": "± 384338",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/uncached_HELLO",
            "value": 592117,
            "range": "± 3312",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/cached_HELLO",
            "value": 73255,
            "range": "± 772",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/cache_warmup_alphabet",
            "value": 5364379,
            "range": "± 122045",
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
            "value": 102922,
            "range": "± 92",
            "unit": "ns/iter"
          },
          {
            "name": "jit_collapse_call_overhead/one_2d_collapse_call/4",
            "value": 64943,
            "range": "± 197",
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
            "value": 4,
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
          "id": "eb03ee9f801362f28d0378338488830fe90de9d3",
          "message": "test: fix STYLE.md violations and mutation-test gaps (2026-08-01 audit) (#971)\n\n## Summary\n\nScheduled test-quality-control pass, continuing\n`docs/bugs/2026-07-28-test-quality-audit-followup.md`. Full writeup:\n`docs/bugs/2026-08-01-test-quality-audit-followup.md`.\n\n- Static-audited everything changed since the last pass\n(`d7d33b8..HEAD`, 33 commits) against `docs/STYLE.md`'s \"Test Public\nAPI\" rule and fixed 4 genuine violations:\n- `actor-scheduler/src/mealy.rs`: removed ten redundant reads of\n`Node`'s private `continuation` field, replacing them with (or\nconfirming they were already subsumed by) public\n`poll()`/`poll_os()`/`actor()` assertions. Also killed a literal\ntautology assertion (`is_none() || is_some()`).\n- `actor-scheduler/src/host.rs` + `src/lib.rs`: this delta's send-policy\nrefactor demoted `try_send` to `pub(crate)` on both `GreenSender` and\n`ActorHandle`; updated tests to go through the new public\n`send()`/`send_port()` paths instead.\n- `core-term/src/terminal_app.rs`: a resize-recompile test called the\nprivate `build_scene()` and read the private `program` field's geometry.\nRewrote to drive the actor's real `RequestFrame` entry point and observe\nrendered output via a new `EngineProbe` test double.\n- `pixelflow-graphics/src/render/scene.rs`: one case judged an\nacceptable Flexibility-clause exception rather than fixed (the private\nfunction under test is shared production logic, not a test seam) —\ndocumented in a comment instead.\n- Mutation-tested `pixelflow-core/src/lattice/cell_grid.rs` (brand new\nthis delta, never audited before) with `cargo-mutants`: 9/47 mutants\nmissed initially, all real coverage gaps. Added 6 tests; re-run confirms\n0 missed.\n\n## Test plan\n\n- [x] `cargo test --workspace --lib` — all 11 crates pass\n- [x] `cargo clippy --workspace --lib --tests` — clean\n- [x] `cargo fmt --check` on every touched crate — clean\n- [x] `cargo mutants -p pixelflow-core --file\npixelflow-core/src/lattice/cell_grid.rs` — 0 missed (was 9)\n- [x] `cargo test -p actor-scheduler --lib mealy:: / host:: /\ntry_send_tests::` — 36/15/3 passing\n- [x] `cargo test -p core-term --lib terminal_app::` — 4/4 passing\n\n🤖 Generated with [Claude Code](https://claude.ai/code)\n\nhttps://claude.ai/code/session_01AW5FTT1BREJHwjgCzVqBm5\n\n---\n_Generated by [Claude\nCode](https://claude.ai/code/session_01AW5FTT1BREJHwjgCzVqBm5)_\n\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-08-01T22:17:08-07:00",
          "tree_id": "5e53b3cfaab8f453bb8d17bd6c03d715f478b8ee",
          "url": "https://github.com/jppittman/pixelflow/commit/eb03ee9f801362f28d0378338488830fe90de9d3"
        },
        "date": 1785651328368,
        "tool": "cargo",
        "benches": [
          {
            "name": "data_throughput_under_control_flood",
            "value": 60424658,
            "range": "± 27332",
            "unit": "ns/iter"
          },
          {
            "name": "burst_limit_vs_unlimited",
            "value": 50377984,
            "range": "± 42780",
            "unit": "ns/iter"
          },
          {
            "name": "four_control_flooders_vs_data",
            "value": 60565089,
            "range": "± 1578269",
            "unit": "ns/iter"
          },
          {
            "name": "slow_receiver_backpressure",
            "value": 63368368,
            "range": "± 222019",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/10",
            "value": 21599,
            "range": "± 2697",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/32",
            "value": 21800,
            "range": "± 3048",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/100",
            "value": 20852,
            "range": "± 3176",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_under_data_flood",
            "value": 2009,
            "range": "± 1007",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/10",
            "value": 21391,
            "range": "± 3118",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/32",
            "value": 21373,
            "range": "± 3223",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/100",
            "value": 21927,
            "range": "± 3466",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_under_control_flood",
            "value": 9960,
            "range": "± 3677",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/1000",
            "value": 15637,
            "range": "± 175",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/1000",
            "value": 14802,
            "range": "± 39",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/10000",
            "value": 165643,
            "range": "± 846",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/10000",
            "value": 160542,
            "range": "± 385",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/100000",
            "value": 1572938,
            "range": "± 23698",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/100000",
            "value": 1520388,
            "range": "± 34738",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/1000",
            "value": 166554,
            "range": "± 1869",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/1000",
            "value": 31258,
            "range": "± 65",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/10000",
            "value": 842331,
            "range": "± 14585",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/10000",
            "value": 327258,
            "range": "± 840",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/100000",
            "value": 7749103,
            "range": "± 333390",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/100000",
            "value": 3153224,
            "range": "± 12865",
            "unit": "ns/iter"
          },
          {
            "name": "priority/data_uncontended",
            "value": 26228,
            "range": "± 437",
            "unit": "ns/iter"
          },
          {
            "name": "priority/data_under_control_flood",
            "value": 31186,
            "range": "± 421",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/10000",
            "value": 188391,
            "range": "± 2533",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/10000",
            "value": 107922,
            "range": "± 1538",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/100000",
            "value": 1566929,
            "range": "± 42248",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/100000",
            "value": 616576,
            "range": "± 3585",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/1000000",
            "value": 15605907,
            "range": "± 238894",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/1000000",
            "value": 5574754,
            "range": "± 29686",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/2",
            "value": 1698550,
            "range": "± 41288",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/2",
            "value": 756440,
            "range": "± 5439",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/4",
            "value": 5270165,
            "range": "± 126929",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/4",
            "value": 1463423,
            "range": "± 51960",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/8",
            "value": 10397697,
            "range": "± 226894",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/8",
            "value": 2816464,
            "range": "± 262536",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_ns/mpsc",
            "value": 13,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_ns/spsc",
            "value": 8,
            "range": "± 0",
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
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_contended/mpsc/4",
            "value": 13,
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
            "value": 23466,
            "range": "± 3880",
            "unit": "ns/iter"
          },
          {
            "name": "roundtrip_latency/spsc",
            "value": 1017,
            "range": "± 163",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/1000",
            "value": 79281,
            "range": "± 4552",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/10000",
            "value": 275736,
            "range": "± 3662",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/100000",
            "value": 2301111,
            "range": "± 211479",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/100",
            "value": 79324,
            "range": "± 2393",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/1000",
            "value": 112965,
            "range": "± 7533",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/10000",
            "value": 585964,
            "range": "± 10818",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_lanes_10k_each",
            "value": 661768,
            "range": "± 12431",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/1024",
            "value": 16743,
            "range": "± 30",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/1024",
            "value": 22184,
            "range": "± 42",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/1024",
            "value": 16277,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/1024",
            "value": 12600,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/1024",
            "value": 11064,
            "range": "± 40",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/1024",
            "value": 17140,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/4096",
            "value": 66453,
            "range": "± 56",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/4096",
            "value": 84470,
            "range": "± 164",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/4096",
            "value": 61866,
            "range": "± 475",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/4096",
            "value": 47996,
            "range": "± 1673",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/4096",
            "value": 42367,
            "range": "± 272",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/4096",
            "value": 66196,
            "range": "± 354",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/16384",
            "value": 260403,
            "range": "± 874",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/16384",
            "value": 331929,
            "range": "± 1465",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/16384",
            "value": 242097,
            "range": "± 529",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/16384",
            "value": 188862,
            "range": "± 2301",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/16384",
            "value": 166844,
            "range": "± 289",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/16384",
            "value": 263737,
            "range": "± 481",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/65536",
            "value": 1046743,
            "range": "± 3075",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/65536",
            "value": 1275453,
            "range": "± 9691",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/65536",
            "value": 981031,
            "range": "± 2628",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/65536",
            "value": 773265,
            "range": "± 8202",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/65536",
            "value": 669717,
            "range": "± 2041",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/65536",
            "value": 1059094,
            "range": "± 4010",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/alt_screen_random_write",
            "value": 774548,
            "range": "± 3914",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/scrolling",
            "value": 1029687,
            "range": "± 19604",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/unicode_random_write",
            "value": 665220,
            "range": "± 2077",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/osc_heavy",
            "value": 848222,
            "range": "± 3310",
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
            "value": 18,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_vec_size_1000",
            "value": 1257,
            "range": "± 2",
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
            "value": 10476,
            "range": "± 44",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/O_quadratic",
            "value": 104683,
            "range": "± 657",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/S_complex",
            "value": 168367,
            "range": "± 785",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/5",
            "value": 806579,
            "range": "± 3254",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/10",
            "value": 4390341,
            "range": "± 10483",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/26",
            "value": 62580815,
            "range": "± 163913",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/50",
            "value": 118561528,
            "range": "± 303419",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/uncached_HELLO",
            "value": 594159,
            "range": "± 3357",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/cached_HELLO",
            "value": 68494,
            "range": "± 184",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/cache_warmup_alphabet",
            "value": 5374320,
            "range": "± 61231",
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
            "value": 102672,
            "range": "± 200",
            "unit": "ns/iter"
          },
          {
            "name": "jit_collapse_call_overhead/one_2d_collapse_call/4",
            "value": 59026,
            "range": "± 532",
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
            "value": 14,
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
          "id": "ab17bab14d0d90a5e98ad591030ee10c46c66645",
          "message": "ci: run benchmark regression checks on pull requests (#972)\n\n## Summary\n- Adds a `pull_request` trigger to the benchmark regression workflow so\nPRs get a benchmark comparison, without mutating the `gh-pages` baseline\n(only `push` events auto-push/open issues; PR events comment on the\ncheck instead).\n\n## Test plan\n- [x] Rebased cleanly onto latest `main`\n- [x] `cargo build --workspace` passes\n- [x] Manually verified window resize behavior still works correctly\n(grow/shrink/extreme-shrink, no corruption) on a release build\n\n## Note\nWhile manually verifying the app for this PR, found an unrelated\npre-existing crash on `main`: the bundled release app crashes\nintermittently on launch (AppKit's internal main-event-queue check\ntripping on `MetalOps`'s hand-rolled event pump, since it skips the\nstandard `NSApplication run()`/`NSApplicationMain` bootstrap). Not\ncaused by anything in this diff and not introduced by recent\nactor-scheduler changes — tracking as separate follow-up work.\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n---------\n\nCo-authored-by: JP Pittman <jppittman@jpptech.dev>\nCo-authored-by: Claude Opus 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-04T04:45:50Z",
          "tree_id": "efdaec1718fd27d1021f284a786dd3d1d275e98b",
          "url": "https://github.com/jppittman/pixelflow/commit/ab17bab14d0d90a5e98ad591030ee10c46c66645"
        },
        "date": 1785822238689,
        "tool": "cargo",
        "benches": [
          {
            "name": "data_throughput_under_control_flood",
            "value": 60398480,
            "range": "± 15442",
            "unit": "ns/iter"
          },
          {
            "name": "burst_limit_vs_unlimited",
            "value": 50311858,
            "range": "± 33187",
            "unit": "ns/iter"
          },
          {
            "name": "four_control_flooders_vs_data",
            "value": 60848269,
            "range": "± 1534862",
            "unit": "ns/iter"
          },
          {
            "name": "slow_receiver_backpressure",
            "value": 63390187,
            "range": "± 204432",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/10",
            "value": 21921,
            "range": "± 2819",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/32",
            "value": 21787,
            "range": "± 3196",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/100",
            "value": 22938,
            "range": "± 3253",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_under_data_flood",
            "value": 2202,
            "range": "± 837",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/10",
            "value": 22804,
            "range": "± 2767",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/32",
            "value": 22172,
            "range": "± 2607",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/100",
            "value": 23153,
            "range": "± 3378",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_under_control_flood",
            "value": 2216,
            "range": "± 4187",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/1000",
            "value": 15554,
            "range": "± 682",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/1000",
            "value": 14799,
            "range": "± 73",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/10000",
            "value": 162713,
            "range": "± 2588",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/10000",
            "value": 157722,
            "range": "± 1172",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/100000",
            "value": 1565702,
            "range": "± 12300",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/100000",
            "value": 1517713,
            "range": "± 9939",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/1000",
            "value": 152416,
            "range": "± 1304",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/1000",
            "value": 28396,
            "range": "± 561",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/10000",
            "value": 928567,
            "range": "± 25370",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/10000",
            "value": 323261,
            "range": "± 9119",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/100000",
            "value": 8338091,
            "range": "± 312923",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/100000",
            "value": 2872429,
            "range": "± 9069",
            "unit": "ns/iter"
          },
          {
            "name": "priority/data_uncontended",
            "value": 22860,
            "range": "± 297",
            "unit": "ns/iter"
          },
          {
            "name": "priority/data_under_control_flood",
            "value": 33314,
            "range": "± 617",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/10000",
            "value": 186785,
            "range": "± 2616",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/10000",
            "value": 105346,
            "range": "± 1266",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/100000",
            "value": 1626111,
            "range": "± 54956",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/100000",
            "value": 526764,
            "range": "± 4272",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/1000000",
            "value": 13492091,
            "range": "± 316602",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/1000000",
            "value": 3972482,
            "range": "± 97108",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/2",
            "value": 1705027,
            "range": "± 52370",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/2",
            "value": 751113,
            "range": "± 5347",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/4",
            "value": 5223771,
            "range": "± 126393",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/4",
            "value": 1428278,
            "range": "± 306004",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/8",
            "value": 10214087,
            "range": "± 289391",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/8",
            "value": 2779440,
            "range": "± 289230",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_ns/mpsc",
            "value": 13,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_ns/spsc",
            "value": 8,
            "range": "± 0",
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
            "value": 16,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_contended/sharded_spsc/4",
            "value": 3,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "roundtrip_latency/mpsc",
            "value": 24325,
            "range": "± 4159",
            "unit": "ns/iter"
          },
          {
            "name": "roundtrip_latency/spsc",
            "value": 1027,
            "range": "± 101",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/1000",
            "value": 79850,
            "range": "± 1413",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/10000",
            "value": 257342,
            "range": "± 2965",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/100000",
            "value": 2115658,
            "range": "± 14367",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/100",
            "value": 79865,
            "range": "± 2191",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/1000",
            "value": 108778,
            "range": "± 11016",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/10000",
            "value": 682231,
            "range": "± 32302",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_lanes_10k_each",
            "value": 653665,
            "range": "± 6571",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/1024",
            "value": 16053,
            "range": "± 84",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/1024",
            "value": 21587,
            "range": "± 154",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/1024",
            "value": 16147,
            "range": "± 33",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/1024",
            "value": 12434,
            "range": "± 163",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/1024",
            "value": 11056,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/1024",
            "value": 16645,
            "range": "± 45",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/4096",
            "value": 63218,
            "range": "± 84",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/4096",
            "value": 83605,
            "range": "± 379",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/4096",
            "value": 61568,
            "range": "± 201",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/4096",
            "value": 48071,
            "range": "± 186",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/4096",
            "value": 41696,
            "range": "± 128",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/4096",
            "value": 63666,
            "range": "± 1047",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/16384",
            "value": 249893,
            "range": "± 417",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/16384",
            "value": 328315,
            "range": "± 606",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/16384",
            "value": 239752,
            "range": "± 860",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/16384",
            "value": 188594,
            "range": "± 506",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/16384",
            "value": 163907,
            "range": "± 406",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/16384",
            "value": 252135,
            "range": "± 389",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/65536",
            "value": 997484,
            "range": "± 3401",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/65536",
            "value": 1262072,
            "range": "± 16244",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/65536",
            "value": 949339,
            "range": "± 1567",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/65536",
            "value": 767455,
            "range": "± 1087",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/65536",
            "value": 655383,
            "range": "± 1398",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/65536",
            "value": 1018471,
            "range": "± 4864",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/alt_screen_random_write",
            "value": 768578,
            "range": "± 8845",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/scrolling",
            "value": 1004999,
            "range": "± 15326",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/unicode_random_write",
            "value": 659140,
            "range": "± 1432",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/osc_heavy",
            "value": 836639,
            "range": "± 3158",
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
            "range": "± 7",
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
            "value": 10647,
            "range": "± 391",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/O_quadratic",
            "value": 106793,
            "range": "± 729",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/S_complex",
            "value": 165034,
            "range": "± 501",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/5",
            "value": 780518,
            "range": "± 2866",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/10",
            "value": 4460924,
            "range": "± 21390",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/26",
            "value": 62571110,
            "range": "± 176599",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/50",
            "value": 118463094,
            "range": "± 280243",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/uncached_HELLO",
            "value": 594059,
            "range": "± 3143",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/cached_HELLO",
            "value": 69083,
            "range": "± 311",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/cache_warmup_alphabet",
            "value": 5296372,
            "range": "± 78452",
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
            "value": 102695,
            "range": "± 110",
            "unit": "ns/iter"
          },
          {
            "name": "jit_collapse_call_overhead/one_2d_collapse_call/4",
            "value": 57643,
            "range": "± 126",
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
            "value": 3,
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
          "id": "eeec0ce5424c927ebe6ea5d229fd2e7efa0366de",
          "message": "refactor(pixelflow): split codegen out of the IR, and stop rendering when nothing changed (#974)\n\nFifteen commits, each independently revertible, all verified green at\nevery step (96 test binaries, clippy clean, all three x86 ISA levels\nchecked since most of the edited emitter code is `cfg`'d out on an arm64\nhost).\n\n## The two measured outcomes\n\n**Idle CPU: 100% of a core → 0.9%.** Measured as CPU-time delta over a\nfixed wall window, same machine, back to back. `AppData::Skipped`\nalready existed for exactly this and was constructed nowhere in\ncore-term; the emulator's per-line dirty tracking already existed and\nwas computed then discarded.\n\n**Cost-model correctness.** 13 of 50 ops were reading a neighbour's\ncycle count — `Dwrt` came back 10 instead of 1000, cheap enough for\nextraction to select an unlowered derivative, which is what the codegen\npanic exists to make impossible. `Shr`, which `expand_log2` emits, came\nback 1000.\n\n## Structure\n\n`pixelflow-ir` was two crates in a trenchcoat. The defs half must sit\nbelow the e-graph; the codegen half always wanted to sit above it.\nWelding them forced the whole thing under, and the API distortion\nfollowed — `optimize_runtime_arena` had to be public because the compile\nentry couldn't call it, so every caller had to remember to optimize\nfirst, and two of three production sites didn't.\n\n```\npixelflow-ir       the language: arena, ops, passes, Kernel   (libm — nothing else)\n  → pixelflow-search    what an expression should be\n    → pixelflow-codegen   what it becomes: emitters, exec memory, cache\n      → pixelflow-core      the standard library\n```\n\n`pixelflow-core` no longer depends on the e-graph at all; it cannot skip\noptimization because it cannot name it.\n\n## Deletions\n\n~5,000 lines, including things nothing compiled (`math.rs`, `patch.rs` —\nno `mod` declaration anywhere), a blanket impl over the empty set\n(`Primitives`/`Compounds`), a 3,400-line SIMD intrinsics library that\nwas in the IR crate on loan, an executable-memory type whose benchmark\nmeasured the opposite of its documented claim, and\n`rand`/`serde`/`serde_json` from `pixelflow-search` — unused, but\ndefault-on, so the terminal's dependency graph carried an RNG and a\nserialization framework to render text.\n\n## Two bugs worth reading the commits for\n\n- `OpKind::from_index(17)` was undefined behaviour: bounds-checked\nagainst `COUNT`, then transmuted, while the discriminants had gaps.\n- `CompileWorkspace` reimplemented `arena_to_schedule` inline without\nthe guard that refuses a surviving `Dwrt`, and ran no lowering passes at\nall.\n\n## Verified by hand\n\nBuilt the bundle and drove it: first frame draws, typed input draws,\ncommand output draws, resize recompiles and redraws, and while idle zero\nframes are presented and the rasterizer's per-frame thread fan-out\ndisappears from the process entirely.\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n---------\n\nCo-authored-by: JP Pittman <jppittman@jpptech.dev>\nCo-authored-by: Claude Opus 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-04T05:36:36Z",
          "tree_id": "fb7a17b68747be1463ff7eeef5656a3bfd890153",
          "url": "https://github.com/jppittman/pixelflow/commit/eeec0ce5424c927ebe6ea5d229fd2e7efa0366de"
        },
        "date": 1785825756676,
        "tool": "cargo",
        "benches": [
          {
            "name": "data_throughput_under_control_flood",
            "value": 60471488,
            "range": "± 38095",
            "unit": "ns/iter"
          },
          {
            "name": "burst_limit_vs_unlimited",
            "value": 50366170,
            "range": "± 37570",
            "unit": "ns/iter"
          },
          {
            "name": "four_control_flooders_vs_data",
            "value": 60844766,
            "range": "± 1693905",
            "unit": "ns/iter"
          },
          {
            "name": "slow_receiver_backpressure",
            "value": 63407627,
            "range": "± 257065",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/10",
            "value": 22303,
            "range": "± 3813",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/32",
            "value": 22358,
            "range": "± 3069",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/100",
            "value": 23179,
            "range": "± 3021",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_under_data_flood",
            "value": 1577,
            "range": "± 1098",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/10",
            "value": 21966,
            "range": "± 3196",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/32",
            "value": 22546,
            "range": "± 3499",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/100",
            "value": 22081,
            "range": "± 3236",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_under_control_flood",
            "value": 1239,
            "range": "± 3977",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/1000",
            "value": 15539,
            "range": "± 72",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/1000",
            "value": 14672,
            "range": "± 166",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/10000",
            "value": 166845,
            "range": "± 3892",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/10000",
            "value": 156543,
            "range": "± 514",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/100000",
            "value": 1600546,
            "range": "± 29764",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/100000",
            "value": 1505225,
            "range": "± 21496",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/1000",
            "value": 168092,
            "range": "± 2420",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/1000",
            "value": 31177,
            "range": "± 931",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/10000",
            "value": 989766,
            "range": "± 13437",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/10000",
            "value": 322982,
            "range": "± 13553",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/100000",
            "value": 8873200,
            "range": "± 346781",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/100000",
            "value": 2875630,
            "range": "± 19329",
            "unit": "ns/iter"
          },
          {
            "name": "priority/data_uncontended",
            "value": 25012,
            "range": "± 218",
            "unit": "ns/iter"
          },
          {
            "name": "priority/data_under_control_flood",
            "value": 33347,
            "range": "± 494",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/10000",
            "value": 189498,
            "range": "± 3166",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/10000",
            "value": 108085,
            "range": "± 3515",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/100000",
            "value": 1496644,
            "range": "± 25809",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/100000",
            "value": 638662,
            "range": "± 3867",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/1000000",
            "value": 14777036,
            "range": "± 254648",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/1000000",
            "value": 6157892,
            "range": "± 21212",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/2",
            "value": 1621817,
            "range": "± 46969",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/2",
            "value": 794176,
            "range": "± 11259",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/4",
            "value": 5127884,
            "range": "± 99141",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/4",
            "value": 1492864,
            "range": "± 17010",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/8",
            "value": 10613019,
            "range": "± 246144",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/8",
            "value": 2822162,
            "range": "± 623073",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_ns/mpsc",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_ns/spsc",
            "value": 8,
            "range": "± 0",
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
            "value": 4,
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
            "value": 6,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "roundtrip_latency/mpsc",
            "value": 24225,
            "range": "± 3769",
            "unit": "ns/iter"
          },
          {
            "name": "roundtrip_latency/spsc",
            "value": 1024,
            "range": "± 101",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/1000",
            "value": 78376,
            "range": "± 1910",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/10000",
            "value": 260673,
            "range": "± 14439",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/100000",
            "value": 2141736,
            "range": "± 54117",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/100",
            "value": 80962,
            "range": "± 2067",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/1000",
            "value": 115328,
            "range": "± 10420",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/10000",
            "value": 614456,
            "range": "± 12102",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_lanes_10k_each",
            "value": 647864,
            "range": "± 29148",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/1024",
            "value": 16433,
            "range": "± 67",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/1024",
            "value": 18800,
            "range": "± 78",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/1024",
            "value": 16507,
            "range": "± 161",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/1024",
            "value": 12523,
            "range": "± 64",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/1024",
            "value": 11213,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/1024",
            "value": 16941,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/4096",
            "value": 64666,
            "range": "± 213",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/4096",
            "value": 85351,
            "range": "± 242",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/4096",
            "value": 63308,
            "range": "± 148",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/4096",
            "value": 48094,
            "range": "± 142",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/4096",
            "value": 43276,
            "range": "± 167",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/4096",
            "value": 64749,
            "range": "± 157",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/16384",
            "value": 255974,
            "range": "± 4858",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/16384",
            "value": 333277,
            "range": "± 1016",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/16384",
            "value": 246951,
            "range": "± 675",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/16384",
            "value": 188854,
            "range": "± 1359",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/16384",
            "value": 169330,
            "range": "± 290",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/16384",
            "value": 258961,
            "range": "± 7013",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/65536",
            "value": 1027056,
            "range": "± 3351",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/65536",
            "value": 1291868,
            "range": "± 5058",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/65536",
            "value": 967853,
            "range": "± 3188",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/65536",
            "value": 773940,
            "range": "± 4493",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/65536",
            "value": 690959,
            "range": "± 2273",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/65536",
            "value": 1042276,
            "range": "± 11062",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/alt_screen_random_write",
            "value": 779710,
            "range": "± 6716",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/scrolling",
            "value": 1034932,
            "range": "± 9229",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/unicode_random_write",
            "value": 687322,
            "range": "± 2312",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/osc_heavy",
            "value": 820355,
            "range": "± 4557",
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
            "value": 18,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_vec_size_1000",
            "value": 1257,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_binsearch_size_1000",
            "value": 33,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jit_collapse_call_overhead/rust_per_batch_loop/4",
            "value": 102739,
            "range": "± 65",
            "unit": "ns/iter"
          },
          {
            "name": "jit_collapse_call_overhead/one_2d_collapse_call/4",
            "value": 64653,
            "range": "± 289",
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
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/A_linear",
            "value": 8268,
            "range": "± 77",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/O_quadratic",
            "value": 77015,
            "range": "± 646",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/S_complex",
            "value": 120902,
            "range": "± 962",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/5",
            "value": 699245,
            "range": "± 3464",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/10",
            "value": 4259731,
            "range": "± 47868",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/26",
            "value": 62001072,
            "range": "± 1002253",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/50",
            "value": 117699781,
            "range": "± 396903",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/uncached_HELLO",
            "value": 549295,
            "range": "± 3942",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/cached_HELLO",
            "value": 66473,
            "range": "± 2304",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/cache_warmup_alphabet",
            "value": 4939371,
            "range": "± 50248",
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
            "value": 1,
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
          "id": "d9e4f7d5e1abeeb46c12f2cdcf7b185e3a24bd8a",
          "message": "fix(pixelflow-pipeline): resolve training-feature build break, remove duplicate OpKind lookups (#976)\n\n## Summary\n- `pixelflow-pipeline/src/training/factored.rs`'s test-only S-expression\nparser referenced `OpKind::Fract`/`OpKind::Hypot`, deleted from\n`pixelflow-ir::OpKind` when those ops were converted from IR primitives\ninto compound functions built from other ops (see\n`pixelflow-ir/src/kernel.rs`). This broke `cargo build --features\ntraining` and thus every `cargo test -p pixelflow-pipeline --features\ntraining` with E0599.\n- `parse_op_kind` no longer maps `\"fract\"`/`\"hypot\"` to the nonexistent\nvariants. `parse_expr_into` now special-cases those two names and builds\nthe same primitive subgraph the runtime uses (`fract(x) = x - floor(x)`,\n`hypot(x,y) = sqrt(x*x + y*y)`, matching\n`pixelflow-ir/src/backend/compounds.rs`), so old logged repros using\nthat syntax still parse and evaluate correctly.\n- Added two regression tests\n(`parse_expr_fract_builds_sub_floor_compound`,\n`parse_expr_hypot_builds_sqrt_mul_add_compound`) since nothing\npreviously exercised this path.\n- Swept `pixelflow-search`, `pixelflow-pipeline`, and\n`pixelflow-compiler` for other hand-rolled `OpKind` name/arity tables\nthat duplicate the canonical `OpKind::name()`/`from_name()`\n(`pixelflow-ir/src/kind.rs`). Found and removed one more:\n`pixelflow-search`'s `CostModel::cost_by_name` had a 38-line hand-copied\nstring table (dead code, zero callers workspace-wide, already flagged as\na defect in `docs/COMPILER_ANALYSIS.md`) — replaced with a direct call\nto `OpKind::from_name`, matching the pattern the rest of that file\nalready uses.\n- `pixelflow-compiler/src/ir_bridge.rs`'s method-call dispatch table\nlooks similar but was deliberately left alone: it doubles as an\nallowlist of which `OpKind`s are reachable from user kernel source, and\nseveral `OpKind::from_name`-recognized names (`tuple`, `iadd`, `shl`,\n`dwrt`, `buffer`, `raw_gather`, `reduce`) are internal/raw ops that must\nnever be publicly callable. Collapsing it to a generic `from_name`\ndispatch would have silently exposed them.\n\n## Test plan\n- [x] `cargo test -p pixelflow-pipeline --lib --features training` —\nbuilds clean; 25/34 pass (the remaining 9 fail for two pre-existing,\nunrelated reasons masked until now by the build break:\n`OpKind::eval_binary` has no arm for `Pow`, and a stack overflow in\n`numerical_gradient_check_value` — tracked separately, not touched here)\n- [x] `cargo test -p pixelflow-search --lib` — 102/102 pass\n- [x] `cargo build -p pixelflow-search -p pixelflow-compiler -p\npixelflow-pipeline --lib` — clean\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n---------\n\nCo-authored-by: JP Pittman <jppittman@jpptech.dev>\nCo-authored-by: Claude Sonnet 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-04T14:10:39Z",
          "tree_id": "3a99f3fae7b67fd39e7ee4eaf3051370d096fad8",
          "url": "https://github.com/jppittman/pixelflow/commit/d9e4f7d5e1abeeb46c12f2cdcf7b185e3a24bd8a"
        },
        "date": 1785856117911,
        "tool": "cargo",
        "benches": [
          {
            "name": "data_throughput_under_control_flood",
            "value": 60496664,
            "range": "± 24900",
            "unit": "ns/iter"
          },
          {
            "name": "burst_limit_vs_unlimited",
            "value": 50391308,
            "range": "± 39871",
            "unit": "ns/iter"
          },
          {
            "name": "four_control_flooders_vs_data",
            "value": 60587927,
            "range": "± 1712221",
            "unit": "ns/iter"
          },
          {
            "name": "slow_receiver_backpressure",
            "value": 63553905,
            "range": "± 157962",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/10",
            "value": 13376,
            "range": "± 4546",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/32",
            "value": 14769,
            "range": "± 4555",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/100",
            "value": 14892,
            "range": "± 4216",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_under_data_flood",
            "value": 809,
            "range": "± 111",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/10",
            "value": 14400,
            "range": "± 4996",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/32",
            "value": 12530,
            "range": "± 5196",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/100",
            "value": 14810,
            "range": "± 4805",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_under_control_flood",
            "value": 796,
            "range": "± 295",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/1000",
            "value": 15232,
            "range": "± 90",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/1000",
            "value": 14016,
            "range": "± 34",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/10000",
            "value": 161352,
            "range": "± 458",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/10000",
            "value": 150786,
            "range": "± 1035",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/100000",
            "value": 1541657,
            "range": "± 21934",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/100000",
            "value": 1436722,
            "range": "± 24280",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/1000",
            "value": 165178,
            "range": "± 1486",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/1000",
            "value": 30035,
            "range": "± 99",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/10000",
            "value": 918845,
            "range": "± 16211",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/10000",
            "value": 313825,
            "range": "± 23078",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/100000",
            "value": 8825071,
            "range": "± 312233",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/100000",
            "value": 3039551,
            "range": "± 81685",
            "unit": "ns/iter"
          },
          {
            "name": "priority/data_uncontended",
            "value": 20923,
            "range": "± 180",
            "unit": "ns/iter"
          },
          {
            "name": "priority/data_under_control_flood",
            "value": 29038,
            "range": "± 473",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/10000",
            "value": 192494,
            "range": "± 5184",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/10000",
            "value": 96694,
            "range": "± 1611",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/100000",
            "value": 1439346,
            "range": "± 85332",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/100000",
            "value": 606272,
            "range": "± 6246",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/1000000",
            "value": 13131691,
            "range": "± 347961",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/1000000",
            "value": 6070869,
            "range": "± 369321",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/2",
            "value": 1478785,
            "range": "± 31412",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/2",
            "value": 808959,
            "range": "± 12229",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/4",
            "value": 4818763,
            "range": "± 72821",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/4",
            "value": 1599667,
            "range": "± 19627",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/8",
            "value": 9701829,
            "range": "± 243650",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/8",
            "value": 3068893,
            "range": "± 301159",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_ns/mpsc",
            "value": 20,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_ns/spsc",
            "value": 7,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_contended/mpsc/2",
            "value": 12,
            "range": "± 3",
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
            "value": 15,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_contended/sharded_spsc/4",
            "value": 6,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "roundtrip_latency/mpsc",
            "value": 14879,
            "range": "± 3917",
            "unit": "ns/iter"
          },
          {
            "name": "roundtrip_latency/spsc",
            "value": 1153,
            "range": "± 113",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/1000",
            "value": 72255,
            "range": "± 1629",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/10000",
            "value": 262873,
            "range": "± 6883",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/100000",
            "value": 2397665,
            "range": "± 156835",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/100",
            "value": 71255,
            "range": "± 1264",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/1000",
            "value": 99772,
            "range": "± 7881",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/10000",
            "value": 340325,
            "range": "± 9361",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_lanes_10k_each",
            "value": 660456,
            "range": "± 17876",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/1024",
            "value": 15908,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/1024",
            "value": 19355,
            "range": "± 46",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/1024",
            "value": 16461,
            "range": "± 91",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/1024",
            "value": 12587,
            "range": "± 102",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/1024",
            "value": 10638,
            "range": "± 37",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/1024",
            "value": 16387,
            "range": "± 35",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/4096",
            "value": 63272,
            "range": "± 750",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/4096",
            "value": 86596,
            "range": "± 95",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/4096",
            "value": 63936,
            "range": "± 179",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/4096",
            "value": 47629,
            "range": "± 250",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/4096",
            "value": 40862,
            "range": "± 494",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/4096",
            "value": 62468,
            "range": "± 70",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/16384",
            "value": 247076,
            "range": "± 342",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/16384",
            "value": 341612,
            "range": "± 548",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/16384",
            "value": 248666,
            "range": "± 2758",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/16384",
            "value": 185756,
            "range": "± 784",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/16384",
            "value": 158515,
            "range": "± 381",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/16384",
            "value": 246927,
            "range": "± 2897",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/65536",
            "value": 991907,
            "range": "± 1522",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/65536",
            "value": 1314457,
            "range": "± 6421",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/65536",
            "value": 986486,
            "range": "± 1454",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/65536",
            "value": 758711,
            "range": "± 7482",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/65536",
            "value": 641793,
            "range": "± 851",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/65536",
            "value": 1004051,
            "range": "± 7923",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/alt_screen_random_write",
            "value": 785697,
            "range": "± 4076",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/scrolling",
            "value": 1002826,
            "range": "± 14978",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/unicode_random_write",
            "value": 649054,
            "range": "± 5771",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/osc_heavy",
            "value": 815973,
            "range": "± 1589",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_map_size_10",
            "value": 19,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_vec_size_10",
            "value": 13,
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
            "value": 19,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_vec_size_100",
            "value": 121,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_binsearch_size_100",
            "value": 23,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_map_size_1000",
            "value": 20,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_vec_size_1000",
            "value": 1211,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_binsearch_size_1000",
            "value": 38,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jit_collapse_call_overhead/rust_per_batch_loop/4",
            "value": 114824,
            "range": "± 984",
            "unit": "ns/iter"
          },
          {
            "name": "jit_collapse_call_overhead/one_2d_collapse_call/4",
            "value": 63845,
            "range": "± 804",
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
            "value": 1,
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
            "value": 1,
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
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_arithmetic/sub",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_arithmetic/mul",
            "value": 5,
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
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_math/max",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_gradient/circle_sdf_gradient",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_gradient/polynomial_gradient",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fix_iteration/converge_fast_all_lanes",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fix_iteration/converge_10_iterations",
            "value": 11,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fix_iteration/converge_variable_lanes",
            "value": 10,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "evaluation_throughput/circle_sdf_64px",
            "value": 24,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "evaluation_throughput/circle_sdf_256px",
            "value": 92,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "evaluation_throughput/circle_sdf_1024px",
            "value": 365,
            "range": "± 1",
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
            "value": 21,
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
            "value": 21,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/manifold_denormal_heavy_no_guard",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/manifold_denormal_heavy_with_guard",
            "value": 19,
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
            "value": 21,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_accumulation_no_guard",
            "value": 69,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_accumulation_with_guard",
            "value": 80,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/A_linear",
            "value": 7552,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/O_quadratic",
            "value": 73516,
            "range": "± 738",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/S_complex",
            "value": 116687,
            "range": "± 1541",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/5",
            "value": 711430,
            "range": "± 2068",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/10",
            "value": 3907766,
            "range": "± 19479",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/26",
            "value": 21486418,
            "range": "± 104129",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/50",
            "value": 40728312,
            "range": "± 749231",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/uncached_HELLO",
            "value": 550992,
            "range": "± 6200",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/cached_HELLO",
            "value": 69154,
            "range": "± 186",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/cache_warmup_alphabet",
            "value": 4565574,
            "range": "± 81861",
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
            "value": 1,
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
            "value": 1,
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
            "value": 3,
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
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/div2_wide",
            "value": 2,
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
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt_div_wide",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt_div_deep",
            "value": 1,
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
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/deep_sqrt3",
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/wide_div2",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/deep_div3",
            "value": 3,
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
            "value": 3,
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
            "value": 3,
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
            "value": 8,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/cos",
            "value": 8,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/sin_cos",
            "value": 11,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/exp",
            "value": 12,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/ln",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/atan2",
            "value": 7,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/pow",
            "value": 17,
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
          "id": "a2cf8f8a99d051a5d96534c67a4c41c1d622365f",
          "message": "test(pixelflow-pipeline): give corpus round-trip tests per-process temp paths (#979)\n\n## Summary\n- `pixelflow-pipeline/src/training/corpus.rs`'s round-trip tests\n(`round_trip_empty`, `round_trip_simple`,\n`round_trip_with_const_and_unary`, `round_trip_ternary`,\n`round_trip_nary`, `round_trip_multiple_entries`) all wrote to fixed\nshared paths under `std::env::temp_dir()` (e.g. `corpus_rt_simple.bin`).\n- Two concurrent `cargo test` invocations (two agents, or a user running\ntests alongside CI) could race on these paths: one process's\n`remove_file` deletes the file while another is between `write_corpus`\nand `read_corpus`, producing spurious `NotFound` failures. Observed\n2026-08-03 during a parallel workspace test run.\n- Added a `unique_tmp(name)` helper that suffixes each path with\n`std::process::id()`, so concurrent test processes never collide. Tests\nare otherwise unchanged.\n\n## Test plan\n- [x] `cargo test -p pixelflow-pipeline --lib --features training\ntraining::corpus` — all 8 tests pass\n- [x] Ran two `cargo test` invocations concurrently against the same\ntree — both completed cleanly with no `NotFound` errors, confirming the\nrace is resolved\n\nNote: the `training` feature must be enabled to build this module at all\n(`--features training`); without it the module is compiled out and the\ntest filter silently matches zero tests.\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-authored-by: JP Pittman <jppittman@jpptech.dev>\nCo-authored-by: Claude Sonnet 5 <noreply@anthropic.com>",
          "timestamp": "2026-08-06T01:36:11Z",
          "tree_id": "0319c88978a74b7753867632721240231d3bb45c",
          "url": "https://github.com/jppittman/pixelflow/commit/a2cf8f8a99d051a5d96534c67a4c41c1d622365f"
        },
        "date": 1785983489081,
        "tool": "cargo",
        "benches": [
          {
            "name": "data_throughput_under_control_flood",
            "value": 60403825,
            "range": "± 37436",
            "unit": "ns/iter"
          },
          {
            "name": "burst_limit_vs_unlimited",
            "value": 50317174,
            "range": "± 21148",
            "unit": "ns/iter"
          },
          {
            "name": "four_control_flooders_vs_data",
            "value": 60469717,
            "range": "± 1465287",
            "unit": "ns/iter"
          },
          {
            "name": "slow_receiver_backpressure",
            "value": 63243459,
            "range": "± 102818",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/10",
            "value": 12498,
            "range": "± 3101",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/32",
            "value": 13681,
            "range": "± 2925",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/100",
            "value": 13680,
            "range": "± 2759",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_under_data_flood",
            "value": 1676,
            "range": "± 496",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/10",
            "value": 13864,
            "range": "± 2851",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/32",
            "value": 13295,
            "range": "± 2966",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/100",
            "value": 12829,
            "range": "± 2914",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_under_control_flood",
            "value": 596,
            "range": "± 164",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/1000",
            "value": 11862,
            "range": "± 115",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/1000",
            "value": 10858,
            "range": "± 46",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/10000",
            "value": 125977,
            "range": "± 345",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/10000",
            "value": 117111,
            "range": "± 729",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/100000",
            "value": 1206116,
            "range": "± 7442",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/100000",
            "value": 1115262,
            "range": "± 6435",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/1000",
            "value": 127875,
            "range": "± 1230",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/1000",
            "value": 23283,
            "range": "± 84",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/10000",
            "value": 783016,
            "range": "± 11839",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/10000",
            "value": 243214,
            "range": "± 1263",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/100000",
            "value": 7160197,
            "range": "± 247687",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/100000",
            "value": 2359885,
            "range": "± 2974",
            "unit": "ns/iter"
          },
          {
            "name": "priority/data_uncontended",
            "value": 15914,
            "range": "± 692",
            "unit": "ns/iter"
          },
          {
            "name": "priority/data_under_control_flood",
            "value": 21930,
            "range": "± 182",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/10000",
            "value": 144542,
            "range": "± 3763",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/10000",
            "value": 79056,
            "range": "± 962",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/100000",
            "value": 1189164,
            "range": "± 60191",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/100000",
            "value": 350976,
            "range": "± 3813",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/1000000",
            "value": 10707556,
            "range": "± 209057",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/1000000",
            "value": 4603903,
            "range": "± 41846",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/2",
            "value": 1136176,
            "range": "± 26654",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/2",
            "value": 619743,
            "range": "± 10612",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/4",
            "value": 3804654,
            "range": "± 54401",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/4",
            "value": 1221018,
            "range": "± 218657",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/8",
            "value": 7554108,
            "range": "± 160582",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/8",
            "value": 2412744,
            "range": "± 212661",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_ns/mpsc",
            "value": 24,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_ns/spsc",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_contended/mpsc/2",
            "value": 9,
            "range": "± 2",
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
            "value": 12,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_contended/sharded_spsc/4",
            "value": 4,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "roundtrip_latency/mpsc",
            "value": 12420,
            "range": "± 3017",
            "unit": "ns/iter"
          },
          {
            "name": "roundtrip_latency/spsc",
            "value": 939,
            "range": "± 96",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/1000",
            "value": 55107,
            "range": "± 828",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/10000",
            "value": 370302,
            "range": "± 7742",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/100000",
            "value": 4554219,
            "range": "± 184712",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/100",
            "value": 53831,
            "range": "± 953",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/1000",
            "value": 70397,
            "range": "± 4284",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/10000",
            "value": 265246,
            "range": "± 9514",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_lanes_10k_each",
            "value": 505529,
            "range": "± 14878",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/1024",
            "value": 12018,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/1024",
            "value": 14920,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/1024",
            "value": 12668,
            "range": "± 237",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/1024",
            "value": 9662,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/1024",
            "value": 8099,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/1024",
            "value": 12097,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/4096",
            "value": 47373,
            "range": "± 100",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/4096",
            "value": 66843,
            "range": "± 554",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/4096",
            "value": 48994,
            "range": "± 318",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/4096",
            "value": 36993,
            "range": "± 109",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/4096",
            "value": 31141,
            "range": "± 58",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/4096",
            "value": 47540,
            "range": "± 2538",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/16384",
            "value": 186202,
            "range": "± 97",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/16384",
            "value": 263193,
            "range": "± 274",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/16384",
            "value": 191379,
            "range": "± 209",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/16384",
            "value": 142685,
            "range": "± 611",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/16384",
            "value": 122241,
            "range": "± 162",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/16384",
            "value": 185486,
            "range": "± 239",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/65536",
            "value": 747891,
            "range": "± 1227",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/65536",
            "value": 1020225,
            "range": "± 1187",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/65536",
            "value": 776240,
            "range": "± 1017",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/65536",
            "value": 583792,
            "range": "± 566",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/65536",
            "value": 491539,
            "range": "± 1330",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/65536",
            "value": 753303,
            "range": "± 972",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/alt_screen_random_write",
            "value": 610631,
            "range": "± 1925",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/scrolling",
            "value": 743978,
            "range": "± 1661",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/unicode_random_write",
            "value": 495167,
            "range": "± 1276",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/osc_heavy",
            "value": 633755,
            "range": "± 1458",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_map_size_10",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_vec_size_10",
            "value": 10,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_binsearch_size_10",
            "value": 10,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_map_size_100",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_vec_size_100",
            "value": 93,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_binsearch_size_100",
            "value": 18,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_map_size_1000",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_vec_size_1000",
            "value": 958,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_binsearch_size_1000",
            "value": 29,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jit_collapse_call_overhead/rust_per_batch_loop/4",
            "value": 89058,
            "range": "± 49",
            "unit": "ns/iter"
          },
          {
            "name": "jit_collapse_call_overhead/one_2d_collapse_call/4",
            "value": 54399,
            "range": "± 381",
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
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_evaluation/circle_sdf_eval",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "macro_vs_manual/macro_circle",
            "value": 1,
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
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_math/abs",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_math/min",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_math/max",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_transcendental/log2_small",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_transcendental/log2_mid",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_transcendental/log2_large",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_transcendental/exp2_small",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_transcendental/exp2_mid",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_transcendental/exp2_large",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_transcendental/log2_exp2_roundtrip",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_comparisons/lt_manifold",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_comparisons/le_manifold",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_comparisons/gt_manifold",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_comparisons/ge_manifold",
            "value": 0,
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
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_bitwise/or_manifold",
            "value": 0,
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
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_creation/y_seeded",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_creation/constant",
            "value": 1,
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
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_math/sqrt",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_math/abs",
            "value": 2,
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
            "value": 8,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fix_iteration/converge_variable_lanes",
            "value": 8,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "evaluation_throughput/circle_sdf_64px",
            "value": 19,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "evaluation_throughput/circle_sdf_256px",
            "value": 73,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "evaluation_throughput/circle_sdf_1024px",
            "value": 287,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_mul_no_guard",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_mul_with_guard",
            "value": 16,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_div_no_guard",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_div_with_guard",
            "value": 16,
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
            "value": 15,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/normal_mul_no_guard",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/normal_mul_with_guard",
            "value": 16,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_accumulation_no_guard",
            "value": 54,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_accumulation_with_guard",
            "value": 62,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/A_linear",
            "value": 6209,
            "range": "± 36",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/O_quadratic",
            "value": 54601,
            "range": "± 137",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/S_complex",
            "value": 88272,
            "range": "± 324",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/5",
            "value": 543626,
            "range": "± 1444",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/10",
            "value": 3029044,
            "range": "± 98826",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/26",
            "value": 16810165,
            "range": "± 42793",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/50",
            "value": 31891745,
            "range": "± 130753",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/uncached_HELLO",
            "value": 412090,
            "range": "± 1210",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/cached_HELLO",
            "value": 55600,
            "range": "± 163",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/cache_warmup_alphabet",
            "value": 3486994,
            "range": "± 54053",
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
            "value": 1,
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
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_evaluation/circle_sdf_eval",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "macro_vs_manual/macro_circle",
            "value": 1,
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
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/chain4_mul",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/wide2_add",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/wide2_mul",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/wide2_mix",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt_x",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt_xy",
            "value": 1,
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
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/div2_wide",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/div2_deep",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/div3_deep",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt_div_wide",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt_div_deep",
            "value": 1,
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
            "value": 1,
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
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomials/quadratic",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomials/cubic",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomials/quartic",
            "value": 0,
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
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/d3w2_left",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/d3w2_right",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/d4w1",
            "value": 0,
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
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/wide_div2",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/deep_div3",
            "value": 2,
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
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "minmax/abs_via_max",
            "value": 0,
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
            "value": 1,
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
            "value": 1,
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
            "value": 1,
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
            "value": 2,
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
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/sin",
            "value": 6,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/cos",
            "value": 6,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/sin_cos",
            "value": 8,
            "range": "± 1",
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
            "value": 14,
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
          "id": "f676b4b8e738aa72e43de06ee86e9dbb36733f41",
          "message": "test: fix STYLE.md violations and mutation-test gaps (2026-08-08 audit) (#985)\n\n## Summary\n\nScheduled test-quality-control pass, continuing\n`docs/bugs/2026-08-01-test-quality-audit-followup.md`. Full writeup:\n`docs/bugs/2026-08-08-test-quality-audit-followup.md`.\n\n- Static-audited everything changed since the last pass\n(`eb03ee9..HEAD`, 4\n  commits, mostly the `pixelflow-codegen` crate-split refactor) against\n`docs/STYLE.md`'s \"Test Public API\" rule and fixed 2 genuine violations:\n  - `pixelflow-pipeline/src/training/corpus.rs`:\n    `v1_corpus_is_refused_with_regeneration_hint` now goes through the\npublic `read_corpus()` via a temp file instead of calling the private\n    `read_corpus_bytes()` directly.\n- `pixelflow-pipeline/tests/judge_weights_load.rs`: this delta bumped\nthe\nNNUE weights-file magic from `TRID` to `TRIE`, but the test's own name,\n    module doc, and several comments still said `TRID`. Renamed\n`judge_weights_round_trip_via_trid` →\n`judge_weights_round_trip_via_trie`\n    and updated every stale reference.\n- Mutation-tested (`cargo-mutants`) the delta's genuinely new logic\nrather\n  than sweeping pre-existing, untouched code:\n  - `pixelflow-ir/src/kind.rs`'s new `OpMap<T>` total-map type,\n    `known_method_names()`, and the `libm`-swapped `eval_unary` arms\n    (`Ceil`/`Round`/`IntToFloat`/`Recip`) — 9 tests added, all surfaced\n    gaps closed.\n  - `pixelflow-search/src/egraph/cost.rs`'s `CostModel::zero()` (this\n    delta's array→`OpMap::splat(0)` migration) — 1 test added\n(`zero_prices_every_op_at_zero_cost`) after the model's own test suite\n    proved too slow (110s+ baseline) for a full sweep in one pass.\n\nSee the writeup doc for the full mutant-by-mutant breakdown and\nrecommended\nnext steps (untested pre-existing backlog in `kind.rs`, the rest of\n`cost.rs`, and the newly-relocated `pixelflow-codegen`/\n`pixelflow-core/src/backend` surfaces).\n\n## Test plan\n\n- [x] `cargo test --workspace --lib` — all 12 crates pass, 0 failures\n- [x] `cargo test -p pixelflow-pipeline --test judge_weights_load` — 1/1\n- [x] `cargo clippy -p pixelflow-ir --lib --tests` — clean\n- [x] `cargo clippy -p pixelflow-pipeline --lib --tests --all-features`\n— clean\n- [x] `cargo clippy -p pixelflow-search --lib --tests --all-features` —\nclean\n- [x] `cargo fmt --check` on every touched crate — clean\n\n🤖 Generated with [Claude Code](https://claude.ai/code)\n\n\n---\n_Generated by [Claude\nCode](https://claude.ai/code/session_01XNju2bMSoQUBBgnHVtkmgh)_\n\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-08-08T05:05:44-07:00",
          "tree_id": "a1077c2acabf46cdac90958a81225867642aec10",
          "url": "https://github.com/jppittman/pixelflow/commit/f676b4b8e738aa72e43de06ee86e9dbb36733f41"
        },
        "date": 1786193983624,
        "tool": "cargo",
        "benches": [
          {
            "name": "data_throughput_under_control_flood",
            "value": 60297419,
            "range": "± 30563",
            "unit": "ns/iter"
          },
          {
            "name": "burst_limit_vs_unlimited",
            "value": 50378356,
            "range": "± 1366169",
            "unit": "ns/iter"
          },
          {
            "name": "four_control_flooders_vs_data",
            "value": 61741311,
            "range": "± 369761",
            "unit": "ns/iter"
          },
          {
            "name": "slow_receiver_backpressure",
            "value": 64021498,
            "range": "± 235283",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/10",
            "value": 21038,
            "range": "± 7028",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/32",
            "value": 21809,
            "range": "± 6286",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/100",
            "value": 21132,
            "range": "± 5405",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_under_data_flood",
            "value": 2427,
            "range": "± 1100",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/10",
            "value": 22082,
            "range": "± 6523",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/32",
            "value": 24521,
            "range": "± 9027",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/100",
            "value": 21704,
            "range": "± 5534",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_under_control_flood",
            "value": 3555,
            "range": "± 3942",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/1000",
            "value": 12538,
            "range": "± 988",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/1000",
            "value": 8230,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/10000",
            "value": 130578,
            "range": "± 4505",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/10000",
            "value": 89362,
            "range": "± 2754",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/100000",
            "value": 1252445,
            "range": "± 6595",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/100000",
            "value": 877482,
            "range": "± 37214",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/1000",
            "value": 104766,
            "range": "± 2564",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/1000",
            "value": 18232,
            "range": "± 62",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/10000",
            "value": 1132539,
            "range": "± 89614",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/10000",
            "value": 187540,
            "range": "± 7050",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/100000",
            "value": 9790906,
            "range": "± 749702",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/100000",
            "value": 1795726,
            "range": "± 62774",
            "unit": "ns/iter"
          },
          {
            "name": "priority/data_uncontended",
            "value": 14102,
            "range": "± 763",
            "unit": "ns/iter"
          },
          {
            "name": "priority/data_under_control_flood",
            "value": 19553,
            "range": "± 303",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/10000",
            "value": 301344,
            "range": "± 17232",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/10000",
            "value": 75217,
            "range": "± 2146",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/100000",
            "value": 2272186,
            "range": "± 321592",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/100000",
            "value": 512787,
            "range": "± 3598",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/1000000",
            "value": 25572625,
            "range": "± 1270362",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/1000000",
            "value": 4799670,
            "range": "± 98999",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/2",
            "value": 7934980,
            "range": "± 256867",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/2",
            "value": 532223,
            "range": "± 38151",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/4",
            "value": 21817913,
            "range": "± 541584",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/4",
            "value": 1188615,
            "range": "± 53729",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/8",
            "value": 31561073,
            "range": "± 4690309",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/8",
            "value": 2141647,
            "range": "± 68436",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_ns/mpsc",
            "value": 15,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_ns/spsc",
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_contended/mpsc/2",
            "value": 139,
            "range": "± 30",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_contended/sharded_spsc/2",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_contended/mpsc/4",
            "value": 23,
            "range": "± 72",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_contended/sharded_spsc/4",
            "value": 3,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "roundtrip_latency/mpsc",
            "value": 17852,
            "range": "± 10635",
            "unit": "ns/iter"
          },
          {
            "name": "roundtrip_latency/spsc",
            "value": 394,
            "range": "± 99",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/1000",
            "value": 57487,
            "range": "± 8142",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/10000",
            "value": 434481,
            "range": "± 10272",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/100000",
            "value": 4221066,
            "range": "± 101434",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/100",
            "value": 58184,
            "range": "± 16950",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/1000",
            "value": 128739,
            "range": "± 12788",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/10000",
            "value": 622619,
            "range": "± 13562",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_lanes_10k_each",
            "value": 886495,
            "range": "± 33914",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/1024",
            "value": 9839,
            "range": "± 84",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/1024",
            "value": 14592,
            "range": "± 55",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/1024",
            "value": 10811,
            "range": "± 56",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/1024",
            "value": 7909,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/1024",
            "value": 6887,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/1024",
            "value": 10401,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/4096",
            "value": 38515,
            "range": "± 1615",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/4096",
            "value": 57540,
            "range": "± 2066",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/4096",
            "value": 40669,
            "range": "± 444",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/4096",
            "value": 29262,
            "range": "± 82",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/4096",
            "value": 26618,
            "range": "± 1035",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/4096",
            "value": 41829,
            "range": "± 1402",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/16384",
            "value": 151297,
            "range": "± 13034",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/16384",
            "value": 224358,
            "range": "± 585",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/16384",
            "value": 165493,
            "range": "± 7928",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/16384",
            "value": 113870,
            "range": "± 475",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/16384",
            "value": 103857,
            "range": "± 3315",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/16384",
            "value": 155814,
            "range": "± 7779",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/65536",
            "value": 636774,
            "range": "± 3111",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/65536",
            "value": 870490,
            "range": "± 3821",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/65536",
            "value": 624616,
            "range": "± 710",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/65536",
            "value": 490580,
            "range": "± 30184",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/65536",
            "value": 415849,
            "range": "± 15981",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/65536",
            "value": 670635,
            "range": "± 33968",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/alt_screen_random_write",
            "value": 481317,
            "range": "± 15679",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/scrolling",
            "value": 655511,
            "range": "± 33281",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/unicode_random_write",
            "value": 419268,
            "range": "± 13834",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/osc_heavy",
            "value": 548159,
            "range": "± 41725",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_map_size_10",
            "value": 10,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_vec_size_10",
            "value": 9,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_binsearch_size_10",
            "value": 8,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_map_size_100",
            "value": 10,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_vec_size_100",
            "value": 101,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_binsearch_size_100",
            "value": 15,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_map_size_1000",
            "value": 10,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_vec_size_1000",
            "value": 848,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_binsearch_size_1000",
            "value": 23,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jit_collapse_call_overhead/rust_per_batch_loop/4",
            "value": 51493,
            "range": "± 8317",
            "unit": "ns/iter"
          },
          {
            "name": "jit_collapse_call_overhead/one_2d_collapse_call/4",
            "value": 29782,
            "range": "± 1214",
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
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_evaluation/one_param_eval",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_evaluation/two_params_eval",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_evaluation/circle_sdf_eval",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "macro_vs_manual/macro_circle",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "macro_vs_manual/manual_circle",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "type_depth/depth_1",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "type_depth/depth_2",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "type_depth/depth_3",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "type_depth/depth_4",
            "value": 0,
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
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_arithmetic/sub",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_arithmetic/mul",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_arithmetic/div",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_arithmetic/chained_mad",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_math/sqrt",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_math/abs",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_math/min",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_math/max",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_transcendental/log2_small",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_transcendental/log2_mid",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_transcendental/log2_large",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_transcendental/exp2_small",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_transcendental/exp2_mid",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_transcendental/exp2_large",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_transcendental/log2_exp2_roundtrip",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_comparisons/lt_manifold",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_comparisons/le_manifold",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_comparisons/gt_manifold",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_comparisons/ge_manifold",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_select/select_with_gt_ast",
            "value": 0,
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
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_bitwise/and_manifold",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "field_bitwise/or_manifold",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_constants/f32_constant",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_constants/X_variable",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_constants/Y_variable",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_simple/X_plus_Y",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_simple/X_mul_Y",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_simple/X_squared",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_simple/fma_X_mul_Y_plus_Z",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_simple/distance_squared",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_simple/distance_from_origin",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_circle/unit_circle_sdf",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_circle/circle_inside_test",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_select/simple_select",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_select/circle_select",
            "value": 0,
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
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_complex/bilinear_interp",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "manifold_complex/min_max_chain",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_creation/x_seeded",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_creation/y_seeded",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_creation/constant",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_arithmetic/add",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_arithmetic/sub",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_arithmetic/mul",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_arithmetic/div",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_math/sqrt",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_math/abs",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_math/min",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_math/max",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_gradient/circle_sdf_gradient",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_gradient/polynomial_gradient",
            "value": 0,
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
            "value": 9,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fix_iteration/converge_variable_lanes",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "evaluation_throughput/circle_sdf_64px",
            "value": 17,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "evaluation_throughput/circle_sdf_256px",
            "value": 72,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "evaluation_throughput/circle_sdf_1024px",
            "value": 282,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_mul_no_guard",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_mul_with_guard",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_div_no_guard",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_div_with_guard",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/manifold_denormal_heavy_no_guard",
            "value": 32,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/manifold_denormal_heavy_with_guard",
            "value": 10,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/normal_mul_no_guard",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/normal_mul_with_guard",
            "value": 15,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_accumulation_no_guard",
            "value": 27,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_accumulation_with_guard",
            "value": 29,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/A_linear",
            "value": 5487,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/O_quadratic",
            "value": 43757,
            "range": "± 1126",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/S_complex",
            "value": 67027,
            "range": "± 138",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/5",
            "value": 325928,
            "range": "± 12554",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/10",
            "value": 2239115,
            "range": "± 130535",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/26",
            "value": 11445903,
            "range": "± 19716",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/50",
            "value": 21675253,
            "range": "± 79823",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/uncached_HELLO",
            "value": 343686,
            "range": "± 16914",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/cached_HELLO",
            "value": 50851,
            "range": "± 3574",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/cache_warmup_alphabet",
            "value": 3094570,
            "range": "± 46871",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial_optimization/manual_unfused",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial_optimization/kernel_optimized",
            "value": 1,
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
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_evaluation/one_param_eval",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_evaluation/two_params_eval",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_evaluation/circle_sdf_eval",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "macro_vs_manual/macro_circle",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "macro_vs_manual/manual_circle",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "type_depth/depth_1",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "type_depth/depth_2",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "type_depth/depth_3",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "type_depth/depth_4",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/add_xy",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/mul_xy",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/sub_xy",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/div_xy",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/add_mul",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/mul_add",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/chain3_add",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/chain3_mul",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/chain4_add",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/chain4_mul",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/wide2_add",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/wide2_mul",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basic_arithmetic/wide2_mix",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt_x",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt_xy",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt2_wide",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt3_wide",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt4_wide",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt2_deep",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt3_deep",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/div2_wide",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/div2_deep",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/div3_deep",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt_div_wide",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt_div_deep",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "distance_functions/dist2d",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "distance_functions/dist3d",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "distance_functions/dist4d",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "distance_functions/dist2d_sq",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "distance_functions/dist3d_sq",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "distance_functions/circle_sdf",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "distance_functions/sphere_sdf",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "distance_functions/box2d_sdf",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "distance_functions/normalize_x",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomials/linear",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomials/quadratic",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomials/cubic",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomials/quartic",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomials/quad2v",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomials/cubic2v",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomials/cross_xy",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomials/cross_xyz",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomials/full_quad2d",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/d2w4",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/d3w2_left",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/d3w2_right",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/d4w1",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/wide_sqrt4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/deep_sqrt3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/wide_div2",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/deep_div3",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "minmax/min_xy",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "minmax/max_xy",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "minmax/clamp",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "minmax/abs_via_max",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "minmax/sdf_union",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "minmax/sdf_intersect",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_raw/add_manual",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_raw/add_kernel_raw",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_raw/dist2d_manual",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_raw/dist2d_kernel_raw",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_raw/sdf_union_manual",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_raw/sdf_union_kernel_raw",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_raw/fma_manual",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kernel_raw/fma_kernel_raw",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/sin",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/cos",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/sin_cos",
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/exp",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/ln",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/atan2",
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/pow",
            "value": 5,
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
          "id": "4c4c900ce7aa9b905c2ec88a13a8e9e788cf69c5",
          "message": "test: close mutation-testing gaps in mealy/coordinator_node/engine_troupe (2026-08-07 audit) (#983)\n\n## Summary\n\nScheduled test-quality-control pass, continuing\n`docs/bugs/2026-08-01-test-quality-audit-followup.md`. Full write-up:\n`docs/bugs/2026-08-07-test-quality-audit-followup.md`.\n\nMutation-tested (`cargo-mutants`) the three surfaces the 2026-08-01 pass\nflagged as never tested, and closed every gap that was a real one:\n\n- **`actor-scheduler/src/mealy.rs`** — 6→4 missed. `Node::poll`'s\nper-lane\n`slot_progress` counter was only proven for the Control lane; `+= 1` →\n`*= 1`\nsurvived on the Management and Data arms. Since the counter starts at 0\nthat\nmutation pins it there, `advance_if_exhausted` never fires, and a\nbacklogged\nlane monopolises the node past its configured burst limit — the same bug\nclass\n  as 2026-07-28's `Host::sweep` finding. The remaining 4 are documented\nequivalent/timeout mutants (trait-default bodies, and\n`Topology::find_cycle`\nguard mutations that provably can't change `validate()`'s public\n`Result`).\n- **`pixelflow-runtime/src/coordinator_node.rs`** — 2→0 missed. The\nFPS-telemetry frame counter was never asserted on its actual value, so a\nmutation leaving it stuck at 0 forever went unnoticed; the `Debug` impl\nwas\n  never exercised.\n- **`pixelflow-runtime/src/engine_troupe.rs`** — 4→2 missed.\n`EngineHandler::send_vsync_control` and `shut_down` were unconditional\nno-ops\nin every existing test, because the `Rig` fixture left their target\nhandles\n`None`. The remaining 2 are `Troupe::with_config`'s bootstrap\narithmetic,\nwhich spawns real OS threads and needs a live platform waker — not\nreachable\n  from a unit test.\n\n6 new regression tests. **No production code changed.**\n\n## Scope changes since this PR was opened\n\n- **The benchmark-workflow change moved to #986.** It was a CI-policy\nchange\nriding along in a test-quality PR; it's now on its own branch cut from\n`main`.\n- **The `corpus.rs` STYLE.md fix is gone from this diff.** #985 (the\n2026-08-08\naudit) swept an overlapping window, found the same violation, and landed\na\nfunctionally identical fix first. This branch is rebased onto that and\nkeeps\n`main`'s version. The write-up records the finding and adds a\nrecommended\nnext step: the audit window should key off merged history rather than\nthe\nprevious audit doc's commit, or successive passes will keep re-auditing\nthe\n  same delta whenever one is still in review when the next fires.\n\n## Test plan\n\n- [x] `cargo test -p actor-scheduler --lib mealy::` (38/38)\n- [x] `cargo test -p pixelflow-runtime --lib coordinator_node::` (13/13)\n- [x] `cargo test -p pixelflow-runtime --lib engine_troupe::` (10/10)\n- [x] `cargo test -p pixelflow-pipeline --features training --lib\ntraining::corpus::` (9/9, against #985's version)\n- [x] `cargo test --workspace --lib` (all crates green)\n- [x] `cargo clippy --workspace --lib --tests` (clean)\n- [x] `cargo fmt --all -- --check` (clean)\n- [x] `cargo mutants` re-run on all three files, confirming the gaps\nclosed\n\n---------\n\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-08-08T22:16:59Z",
          "tree_id": "5531456e2e43a0a9994487fc1e273fc8ee8fc4a1",
          "url": "https://github.com/jppittman/pixelflow/commit/4c4c900ce7aa9b905c2ec88a13a8e9e788cf69c5"
        },
        "date": 1786230865002,
        "tool": "cargo",
        "benches": [
          {
            "name": "data_throughput_under_control_flood",
            "value": 60450962,
            "range": "± 295454",
            "unit": "ns/iter"
          },
          {
            "name": "burst_limit_vs_unlimited",
            "value": 50346348,
            "range": "± 28593",
            "unit": "ns/iter"
          },
          {
            "name": "four_control_flooders_vs_data",
            "value": 60532530,
            "range": "± 70523",
            "unit": "ns/iter"
          },
          {
            "name": "slow_receiver_backpressure",
            "value": 63511603,
            "range": "± 297885",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/10",
            "value": 13666,
            "range": "± 4325",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/32",
            "value": 11692,
            "range": "± 3990",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_steady_state/buffer/100",
            "value": 13496,
            "range": "± 4938",
            "unit": "ns/iter"
          },
          {
            "name": "control_latency_under_data_flood",
            "value": 683,
            "range": "± 183",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/10",
            "value": 10625,
            "range": "± 4586",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/32",
            "value": 13834,
            "range": "± 4617",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_steady_state/buffer/100",
            "value": 13398,
            "range": "± 4457",
            "unit": "ns/iter"
          },
          {
            "name": "management_latency_under_control_flood",
            "value": 973,
            "range": "± 190",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/1000",
            "value": 15278,
            "range": "± 213",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/1000",
            "value": 13973,
            "range": "± 351",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/10000",
            "value": 161233,
            "range": "± 1351",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/10000",
            "value": 151444,
            "range": "± 1538",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/actor_sends/100000",
            "value": 1545294,
            "range": "± 24017",
            "unit": "ns/iter"
          },
          {
            "name": "dispatch/transducer_returns/100000",
            "value": 1438520,
            "range": "± 40038",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/1000",
            "value": 151209,
            "range": "± 1665",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/1000",
            "value": 30032,
            "range": "± 143",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/10000",
            "value": 915628,
            "range": "± 13941",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/10000",
            "value": 314235,
            "range": "± 2437",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/three_threads/100000",
            "value": 8221532,
            "range": "± 224861",
            "unit": "ns/iter"
          },
          {
            "name": "pipeline_3_stage/one_thread/100000",
            "value": 3043111,
            "range": "± 13224",
            "unit": "ns/iter"
          },
          {
            "name": "priority/data_uncontended",
            "value": 24348,
            "range": "± 1461",
            "unit": "ns/iter"
          },
          {
            "name": "priority/data_under_control_flood",
            "value": 30807,
            "range": "± 118",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/10000",
            "value": 185176,
            "range": "± 6211",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/10000",
            "value": 99503,
            "range": "± 1351",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/100000",
            "value": 1475610,
            "range": "± 59894",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/100000",
            "value": 514850,
            "range": "± 13246",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/mpsc/1000000",
            "value": 13126011,
            "range": "± 169922",
            "unit": "ns/iter"
          },
          {
            "name": "single_producer_throughput/spsc/1000000",
            "value": 4493688,
            "range": "± 127679",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/2",
            "value": 1474675,
            "range": "± 25362",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/2",
            "value": 814172,
            "range": "± 7780",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/4",
            "value": 4787902,
            "range": "± 71024",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/4",
            "value": 1586510,
            "range": "± 24661",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/mpsc/8",
            "value": 9470030,
            "range": "± 204910",
            "unit": "ns/iter"
          },
          {
            "name": "multi_producer_throughput/sharded_spsc/8",
            "value": 3077425,
            "range": "± 380505",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_ns/mpsc",
            "value": 22,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_ns/spsc",
            "value": 6,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "send_latency_contended/mpsc/2",
            "value": 12,
            "range": "± 3",
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
            "value": 15,
            "range": "± 8",
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
            "value": 13212,
            "range": "± 4304",
            "unit": "ns/iter"
          },
          {
            "name": "roundtrip_latency/spsc",
            "value": 1221,
            "range": "± 111",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/1000",
            "value": 71239,
            "range": "± 1370",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/10000",
            "value": 273111,
            "range": "± 11779",
            "unit": "ns/iter"
          },
          {
            "name": "data_throughput/messages/100000",
            "value": 3631628,
            "range": "± 193369",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/100",
            "value": 70230,
            "range": "± 891",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/1000",
            "value": 82146,
            "range": "± 3083",
            "unit": "ns/iter"
          },
          {
            "name": "control_throughput/messages/10000",
            "value": 300298,
            "range": "± 8536",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_lanes_10k_each",
            "value": 635311,
            "range": "± 10108",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/1024",
            "value": 16137,
            "range": "± 393",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/1024",
            "value": 21817,
            "range": "± 140",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/1024",
            "value": 16822,
            "range": "± 83",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/1024",
            "value": 12557,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/1024",
            "value": 10709,
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/1024",
            "value": 16440,
            "range": "± 45",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/4096",
            "value": 63141,
            "range": "± 535",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/4096",
            "value": 86588,
            "range": "± 343",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/4096",
            "value": 63427,
            "range": "± 691",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/4096",
            "value": 47229,
            "range": "± 256",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/4096",
            "value": 40763,
            "range": "± 89",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/4096",
            "value": 63075,
            "range": "± 616",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/16384",
            "value": 250515,
            "range": "± 356",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/16384",
            "value": 345354,
            "range": "± 1260",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/16384",
            "value": 247657,
            "range": "± 458",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/16384",
            "value": 185483,
            "range": "± 5183",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/16384",
            "value": 161594,
            "range": "± 3244",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/16384",
            "value": 250357,
            "range": "± 321",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/ascii_text/65536",
            "value": 1005695,
            "range": "± 3324",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/csi_heavy/65536",
            "value": 1319673,
            "range": "± 17425",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/sgr_256_colors/65536",
            "value": 982308,
            "range": "± 8263",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/cursor_movement/65536",
            "value": 752977,
            "range": "± 2483",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/unicode_heavy/65536",
            "value": 646514,
            "range": "± 15175",
            "unit": "ns/iter"
          },
          {
            "name": "ansi_parser/scrolling/65536",
            "value": 1013922,
            "range": "± 7465",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/alt_screen_random_write",
            "value": 787828,
            "range": "± 27043",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/scrolling",
            "value": 988461,
            "range": "± 9028",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/unicode_random_write",
            "value": 646665,
            "range": "± 2253",
            "unit": "ns/iter"
          },
          {
            "name": "vtebench_scenarios/osc_heavy",
            "value": 819943,
            "range": "± 10649",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_map_size_10",
            "value": 19,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_vec_size_10",
            "value": 13,
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
            "value": 19,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_vec_size_100",
            "value": 121,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_binsearch_size_100",
            "value": 23,
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
            "value": 1244,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "keybindings/lookup_binsearch_size_1000",
            "value": 38,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jit_collapse_call_overhead/rust_per_batch_loop/4",
            "value": 114635,
            "range": "± 87",
            "unit": "ns/iter"
          },
          {
            "name": "jit_collapse_call_overhead/one_2d_collapse_call/4",
            "value": 63928,
            "range": "± 820",
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
            "value": 1,
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
            "value": 1,
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
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_arithmetic/sub",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_arithmetic/mul",
            "value": 5,
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
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_math/max",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_gradient/circle_sdf_gradient",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jet2_gradient/polynomial_gradient",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fix_iteration/converge_fast_all_lanes",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fix_iteration/converge_10_iterations",
            "value": 11,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fix_iteration/converge_variable_lanes",
            "value": 10,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "evaluation_throughput/circle_sdf_64px",
            "value": 24,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "evaluation_throughput/circle_sdf_256px",
            "value": 92,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "evaluation_throughput/circle_sdf_1024px",
            "value": 365,
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
            "value": 21,
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
            "value": 21,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/manifold_denormal_heavy_no_guard",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/manifold_denormal_heavy_with_guard",
            "value": 19,
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
            "value": 21,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_accumulation_no_guard",
            "value": 69,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "fastmath_denormals/denormal_accumulation_with_guard",
            "value": 80,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/A_linear",
            "value": 7615,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/O_quadratic",
            "value": 73348,
            "range": "± 269",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_single_char/S_complex",
            "value": 114208,
            "range": "± 1917",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/5",
            "value": 705926,
            "range": "± 2882",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/10",
            "value": 3860311,
            "range": "± 80825",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/26",
            "value": 21296742,
            "range": "± 187793",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_text_sizes/50",
            "value": 40364087,
            "range": "± 89261",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/uncached_HELLO",
            "value": 527218,
            "range": "± 3440",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/cached_HELLO",
            "value": 69340,
            "range": "± 954",
            "unit": "ns/iter"
          },
          {
            "name": "pixelflow_caching/cache_warmup_alphabet",
            "value": 4530338,
            "range": "± 69096",
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
            "value": 1,
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
            "value": 1,
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
            "value": 3,
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
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/div2_wide",
            "value": 2,
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
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt_div_wide",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expensive_ops/sqrt_div_deep",
            "value": 1,
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
            "value": 3,
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
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/deep_sqrt3",
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/wide_div2",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "depth_vs_width/deep_div3",
            "value": 3,
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
            "value": 3,
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
            "value": 3,
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
            "value": 8,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/cos",
            "value": 8,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/sin_cos",
            "value": 11,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/exp",
            "value": 13,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/ln",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/atan2",
            "value": 7,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "transcendental/pow",
            "value": 17,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}