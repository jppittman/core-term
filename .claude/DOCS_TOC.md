# Project Documentation Index

Auto-generated TOC of `cargo doc` output. Use this to find types, traits, and modules.

---

## Crates

### actor_scheduler

**Structs:** ActorHandle,ActorScheduler

**Traits:** Actor,ActorTypes,TroupeActor,WakeHandler

**Enums:** Message,ParkHint,SendError

**Functions:** create_actor

---

### actor_scheduler_macros

---

### core_term

**Modules:**
- `ansi`
- `ansi/commands`
- `color`
- `config`
- `glyph`
- `io`
- `io/epoll`
- `io/event`
- `io/event_monitor_actor`
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
- `term/cursor_visibility`
- `terminal_app`
- `term/layout`
- `term/modes`
- `term/screen`
- `term/snapshot`
- `term/unicode`

**Structs:** AnsiProcessor,AppearanceConfig,AttrFlags,Attributes,BehaviorConfig,Cell,CellChannel,ColorScheme,Config,ConstCoverage,ContentCell,Cursor,CursorConfig,CursorController,CursorRenderState,epoll_event,EpollEvent,EpollFlags,EventMonitor,EventMonitorActor,FontConfig,Keybinding,KeybindingsConfig,Layout,LocalCoords,Modifiers,MouseConfig,NixPty,Point,PtyConfig,RenderRequest,ScreenContext,Selection,SelectionRange,ShellConfig,SnapshotLine,TerminalApp,TerminalAppParams,TerminalEmulator,TerminalSnapshot,TerminalSurface

**Traits:** AnsiParser,CellFactory,EventSource,PtyChannel,TerminalInterface

**Enums:** AltScreenClear,AnsiCommand,AppEvent,Attribute,C0Control,CharacterSet,Color,ControlEvent,CsiCommand,CursorShape,CursorVisibility,EmulatorAction,EmulatorInput,EpollCtlOp,EscCommand,Glyph,KeySymbol,LineState,Mode,ModeAction,NamedColor,PtyCommand,ScrollHistory,SelectionMode,StandardModeConstant,TabClearMode,UserInputAction

**Functions:** build_grid,get_char_display_width,map_key_event_to_action,map_to_dec_line_drawing,spawn_terminal_app

---

### pixelflow_core

**Modules:**
- `backend`
- `backend/fastmath`
- `backend/scalar`
- `backend/x86`
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
- `jet/jet2`
- `jet/jet2h`
- `jet/jet3`
- `jet/path_jet`
- `manifold`
- `ops`
- `ops/binary`
- `ops/compare`
- `ops/logic`
- `ops/unary`
- `ops/vector`
- `variables`

**Structs:** Abs,Add,AddMasked,And,At,AtArray,BNot,BoxedManifold,ClosureMap,Cos,Div,F32x4,FastMathGuard,Fix,Floor,Ge,Gt,Le,Lt,Map,Mask4,MaskScalar,Max,Min,Mul,MulAdd,MulRecip,MulRsqrt,Or,Pack,PathJet,Project,Rsqrt,Scalar,ScalarF32,ScalarU32,Select,Sh2Field,ShCoeffs,ShProject,ShReconstruct,Sin,SoftGt,SoftLt,SoftSelect,SphericalHarmonic,Sqrt,Sse2,Sub,Texture,Thunk,U32x4,W,X,Y,Z,ZonalHarmonic

**Traits:** Backend,Computational,Differentiable,Dimension,FieldCondition,Manifold,ManifoldExt,MaskOps,Projectable,SimdOps,SimdU32Ops,Vector

**Enums:** Axis

**Functions:** cosine_lobe_sh2,materialize,materialize_discrete,materialize_discrete_fields,scale,sh2_basis_at,sh2_multiply,sh2_multiply_field,sh2_multiply_static_field

---

### pixelflow_graphics

**Modules:**
- `animation`
- `baked`
- `fonts`
- `fonts/cache`
- `fonts/combinators`
- `fonts/loader`
- `fonts/text`
- `fonts/ttf`
- `image`
- `mesh`
- `patch`
- `render`
- `render/aa`
- `render/color`
- `render/frame`
- `render/pixel`
- `render/rasterizer`
- `render/rasterizer/parallel`
- `scene3d`
- `shapes`
- `spatial_bsp`
- `subdiv`
- `subdivision`
- `transform`

**Structs:** AACoverage,AttrFlags,Baked,BezierPatch,Bgra8,CachedGlyph,CachedText,Checker,ColorChecker,ColorCube,ColorReflect,ColorScreenToDir,ColorSky,ColorSurface,Curve,DataSource,EigenCoeffs,EigenStructure,EmbeddedSource,Font,Frame,Geometry,GeometryColor,GeometryMask,GlyphCache,HeightFieldGeometry,Image,InteriorNode,Lift,Line,LoadedFont,Map,MmapSource,OptLine,OptQuad,Oscillate,PlaneGeometry,Point3,Positioned,Quad,QuadMesh,Rasterize,Reflect,RenderOptions,Rgba8,Scale,SceneObject,ScreenToDir,Sky,SpatialBSP,Stripe,SubdivisionGeometry,SubdivisionPatch,SubdivisionSurface,Sum,Surface,SurfaceStats,TensorShape,Translate,Union,UnitSphere,W,X,Y,Z

**Traits:** FontSource,GlyphExt,Manifold,ManifoldExt,Pixel,Scene

**Enums:** Axis,Color,Glyph,NamedColor,NodeRef

**Functions:** aa_coverage,affine,annulus,bicubic,bspline_patch,circle,color_manifold,eigen_patch,ellipse,execute,execute_stripe,get_eigen,Grayscale,half_plane_x,half_plane_y,loop_blinn_quad,make_line,make_quad,rectangle,render_parallel,render_parallel_pooled,render_work_stealing,screen_remap,square,text,threshold,time_shift,validate_eigen_domain

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
- `display/drivers/x11`
- `display/messages`
- `display/ops`
- `display/platform`
- `engine_troupe`
- `error`
- `frame`
- `input`
- `platform`
- `platform/linux`
- `platform/waker`
- `render_pool`
- `testing`
- `testing/mock_engine`
- `traits`
- `vsync_actor`

**Structs:** ActorHandle,ActorScheduler,AppState,Directory,DriverActor,EngineChannels,EngineConfig,EngineHandle,EngineHandler,ExposedHandles,FramePacket,LinuxOps,MockEngine,Modifiers,NoOpWaker,PerformanceConfig,PlatformActor,RenderedResponse,RenderOptions,Troupe,VsyncActor,VsyncConfig,WindowConfig,WindowDescriptor,WindowId,X11DisplayDriver,X11Waker

**Traits:** Actor,Application,DisplayDriver,EventLoopWaker,Platform,PlatformOps,WakeHandler

**Enums:** AppAction,AppData,AppManagement,CursorIcon,DisplayControl,DisplayData,DisplayEvent,DisplayMgmt,DriverCommand,EngineCommand,EngineControl,EngineData,EngineEvent,EngineEventControl,EngineEventData,EngineEventManagement,KeySymbol,Message,MouseButton,ReceivedMessage,RuntimeError,SendError,VsyncCommand,VsyncManagement

**Functions:** create_engine_actor,create_engine_channels,create_frame_channel,create_recycle_channel,render_parallel,run

---

### xtask

**Structs:** EigenData

**Functions:** bake_eigen,bundle_run,find_workspace_root,format_f32_array,main,read_f64_le

---

