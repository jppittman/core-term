# Project Documentation Index

Auto-generated TOC of `cargo doc` output. Use this to find types, traits, and modules.

---

## Crates

### actor_scheduler_macros

---

### actor_scheduler

**Structs:** ActorHandle,ActorScheduler

**Traits:** Actor,ActorTypes,TroupeActor,WakeHandler

**Enums:** Message,ParkHint,SendError

**Functions:** create_actor

---

### core_term

**Modules:**
- `ansi`
- `ansi/commands`
- `color`
- `config`
- `glyph`
- `io`
- `io/event`
- `io/event_monitor_actor`
- `io/kqueue`
- `io/pty`
- `io/traits`
- `keys`
- `messages`
- `surface`
- `surface/manifold`
- `surface/terminal`
- `term`
- `term/action`
- `term/charset`
- `term/cursor`
- `term/layout`
- `term/modes`
- `term/screen`
- `term/snapshot`
- `term/unicode`
- `terminal_app`

**Structs:** AnsiProcessor,AppearanceConfig,AttrFlags,Attributes,BehaviorConfig,Cell,CellChannel,ColorScheme,Config,ConstCoverage,ContentCell,Cursor,CursorConfig,CursorController,CursorRenderState,EventMonitor,EventMonitorActor,FontConfig,Keybinding,KeybindingsConfig,KqueueEvent,KqueueFlags,Layout,LocalCoords,Modifiers,MouseConfig,NixPty,Point,PtyConfig,RenderRequest,ScreenContext,Selection,SelectionRange,ShellConfig,SnapshotLine,TerminalApp,TerminalEmulator,TerminalSnapshot,TerminalSurface

**Traits:** AnsiParser,CellFactory,EventSource,PtyChannel,TerminalInterface

**Enums:** AltScreenClear,AnsiCommand,AppEvent,Attribute,C0Control,CharacterSet,Color,ControlEvent,CsiCommand,CursorShape,EmulatorAction,EmulatorInput,EscCommand,Glyph,KeySymbol,Mode,ModeAction,NamedColor,ScrollHistory,SelectionMode,StandardModeConstant,TabClearMode,UserInputAction

**Functions:** build_grid,get_char_display_width,map_key_event_to_action,map_to_dec_line_drawing,spawn_terminal_app

---

### pixelflow_core

**Modules:**
- `backend`
- `backend/arm`
- `backend/scalar`
- `combinators`
- `combinators/at`
- `combinators/fix`
- `combinators/map`
- `combinators/pack`
- `combinators/project`
- `combinators/select`
- `combinators/spherical`
- `combinators/texture`
- `ext`
- `jet`
- `manifold`
- `ops`
- `ops/binary`
- `ops/compare`
- `ops/logic`
- `ops/unary`
- `ops/vector`
- `variables`

**Structs:** Abs,Add,AddMasked,And,At,BNot,BoxedManifold,Div,F32x4,Fix,Floor,Ge,Gt,Le,Lt,Map,Mask4,MaskScalar,Max,Min,Mul,MulAdd,MulRecip,MulRsqrt,Neon,Or,Pack,Project,Rsqrt,Scalar,ScalarF32,ScalarU32,Scale,Select,Sh2Field,ShCoeffs,ShProject,ShReconstruct,Sin,SoftGt,SoftLt,SoftSelect,SphericalHarmonic,Sqrt,Sub,Texture,U32x4,W,X,Y,Z,ZonalHarmonic

**Traits:** Backend,Computational,Dimension,FieldCondition,Manifold,ManifoldExt,MaskOps,Projectable,SimdOps,SimdU32Ops,Vector

**Enums:** Axis

**Functions:** cosine_lobe_sh2,materialize,materialize_discrete,materialize_discrete_fields,sh2_basis_at,sh2_multiply,sh2_multiply_field,sh2_multiply_static_field

---

### pixelflow_graphics

**Modules:**
- `baked`
- `fonts`
- `fonts/cache`
- `fonts/combinators`
- `fonts/loader`
- `fonts/text`
- `fonts/ttf`
- `image`
- `render`
- `render/aa`
- `render/color`
- `render/frame`
- `render/pixel`
- `render/rasterizer`
- `render/rasterizer/parallel`
- `render/rasterizer/pool`
- `scene3d`
- `shapes`
- `transform`

**Structs:** AACoverage,Affine,AttrFlags,Baked,Bgra8,CachedGlyph,CachedText,Checker,ColorChecker,ColorManifold,ColorMap,ColorReflect,ColorScreenToDir,ColorSky,ColorSurface,Curve,DataSource,Discrete,EmbeddedSource,Field,Font,Frame,Geometry,GlyphCache,Image,Lift,Line,LoadedFont,Map,MmapSource,OptLine,OptQuad,PlaneGeometry,Rasterize,Reflect,RenderOptions,Rgba8,Scale,ScreenToDir,Sky,SphereAt,Stripe,Sum,Surface,TensorShape,Text,ThreadPool,Translate,UnitSphere,W,X,Y,Z

**Traits:** FontSource,GlyphExt,Manifold,ManifoldExt,Pixel

**Enums:** Color,Glyph,NamedColor

**Functions:** aa_coverage,circle,execute,execute_stripe,half_plane_x,half_plane_y,render_parallel,render_parallel_pooled,render_work_stealing,square,threshold

---

### pixelflow_ml

**Structs:** EluFeature,HarmonicAttention,HarmonicAttentionIsGlobalIllumination,LinearAttention,RandomFourierFeature,ShFeatureMap

**Traits:** FeatureMap

---

### pixelflow_runtime

**Modules:**
- `api`
- `api/private`
- `api/public`
- `channel`
- `config`
- `display`
- `display/driver`
- `display/drivers`
- `display/messages`
- `display/ops`
- `display/platform`
- `engine_troupe`
- `error`
- `frame`
- `input`
- `platform`
- `platform/macos`
- `platform/macos/cocoa`
- `platform/macos/cocoa/event_type`
- `platform/macos/events`
- `platform/macos/platform`
- `platform/macos/sys`
- `platform/macos/window`
- `platform/waker`
- `render_pool`
- `testing`
- `testing/mock_engine`
- `traits`
- `vsync_actor`

**Structs:** ActorHandle,ActorScheduler,AppState,CGPoint,CGRect,CGSize,CocoaWaker,Directory,DriverActor,EngineChannels,EngineConfig,EngineHandle,EngineHandler,ExposedHandles,FramePacket,MacWindow,MetalOps,MockEngine,Modifiers,MTLOrigin,MTLRegion,MTLSize,NoOpWaker,NSApplication,NSEvent,NSPasteboard,NSPoint,NSRect,NSSize,NSView,NSWindow,PerformanceConfig,PlatformActor,RenderedResponse,RenderOptions,Troupe,VsyncActor,VsyncConfig,WindowConfig,WindowDescriptor,WindowId

**Traits:** Actor,Application,DisplayDriver,EventLoopWaker,Platform,PlatformOps,WakeHandler

**Enums:** AppAction,AppData,AppManagement,CursorIcon,DisplayControl,DisplayData,DisplayEvent,DisplayMgmt,DriverCommand,EngineCommand,EngineControl,EngineData,EngineEvent,EngineEventControl,EngineEventData,EngineEventManagement,KeySymbol,Message,MouseButton,ReceivedMessage,RuntimeError,SendError,VsyncCommand,VsyncManagement

**Functions:** class,create_engine_actor,create_engine_channels,create_frame_channel,create_recycle_channel,make_nsstring,map_event,nsstring_to_string,objc_getClass,objc_msgSend,render_parallel,run,sel,sel_registerName,send,send_1,send_2,send_3,send_4

---

### xtask

**Functions:** bundle_run,find_workspace_root,main

---

