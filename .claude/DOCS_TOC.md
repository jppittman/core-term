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

### cfg_if

---

### colorchoice

**Enums:** ColorChoice

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

### libc

**Structs:** __darwin_arm_exception_state64,__darwin_arm_neon_state64,__darwin_arm_thread_state64,__darwin_mcontext64,addrinfo,aiocb,arphdr,attribute_set_t,attrlist,attrreference_t,bpf_hdr,cmsghdr,ctl_info,dirent,Dl_info,dqblk,fd_set,flock,fpunchhole_t,fsid_t,fspecread_t,fstore_t,ftrimactivefile_t,glob_t,group,host_cpu_load_info,hostent,icmp6_ifstat,if_data,if_data64,if_msghdr,if_msghdr2,if_nameindex,ifa_msghdr,ifaddrs,ifconf,ifdevmtu,ifkpi,ifma_msghdr,ifma_msghdr2,ifmibdata,ifreq,ifs_iso_8802_3,image_offset,in_addr,in_pktinfo,in6_addr,in6_addrlifetime,in6_ifreq,in6_ifstat,in6_pktinfo,iovec,ip_mreq,ip_mreq_source,ip_mreqn,ipc_perm,ipv6_mreq,itimerval,kevent,kevent64_s,lconv,linger,load_command,log2phys,mach_header,mach_header_64,mach_task_basic_info,mach_timebase_info,malloc_statistics_t,malloc_zone_t,max_align_t,msghdr,mstats,ntptimeval,option,os_unfair_lock_s,passwd,pollfd,proc_bsdinfo,proc_fdinfo,proc_taskallinfo,proc_taskinfo,proc_threadinfo,proc_vnodepathinfo,processor_basic_info,processor_cpu_load_info,processor_set_basic_info,processor_set_load_info,protoent,pthread_attr_t,pthread_cond_t,pthread_condattr_t,pthread_mutex_t,pthread_mutexattr_t,pthread_once_t,pthread_rwlock_t,pthread_rwlockattr_t,radvisory,regex_t,regmatch_t,rlimit,rt_metrics,rt_msghdr,rt_msghdr2,rusage,rusage_info_v0,rusage_info_v1,rusage_info_v2,rusage_info_v3,rusage_info_v4,sa_endpoints_t,sched_param,segment_command,segment_command_64,sembuf,semid_ds,servent,sf_hdtr,shmid_ds,sigaction,sigevent,siginfo_t,sigval,sockaddr,sockaddr_ctl,sockaddr_dl,sockaddr_in,sockaddr_in6,sockaddr_inarp,sockaddr_ndrv,sockaddr_storage,sockaddr_un,sockaddr_vm,stack_t,stat,statfs,statvfs,task_thread_times_info,tcp_connection_info,termios,thread_affinity_policy,thread_background_policy,thread_basic_info,thread_extended_info,thread_extended_policy,thread_identifier_info,thread_latency_qos_policy,thread_precedence_policy,thread_standard_policy,thread_throughput_qos_policy,thread_time_constraint_policy,time_value_t,timespec,timeval,timeval32,timex,tm,tms,ucontext_t,utimbuf,utmpx,utsname,vinfo_stat,vm_range_t,vm_statistics,vm_statistics64,vnode_info,vnode_info_path,vol_attributes_attr_t,vol_capabilities_attr_t,winsize,xsw_usage,xucred

**Enums:** c_void,DIR,FILE,fpos_t,qos_class_t,sysdir_search_path_directory_t,sysdir_search_path_domain_mask_t,timezone

**Functions:** __error,_dyld_get_image_header,_dyld_get_image_name,_dyld_get_image_vmaddr_slide,_dyld_image_count,_exit,_NSGetArgc,_NSGetArgv,_NSGetEnviron,_NSGetExecutablePath,_NSGetProgname,_WSTATUS,abort,abs,accept,access,acct,adjtime,aio_cancel,aio_error,aio_fsync,aio_read,aio_return,aio_suspend,aio_write,alarm,aligned_alloc,arc4random,arc4random_buf,arc4random_uniform,asctime,asctime_r,atexit,atof,atoi,atol,atoll,backtrace,backtrace_async,backtrace_from_fp,backtrace_image_offsets,backtrace_symbols,backtrace_symbols_fd,basename,bind,brk,bsearch,calloc,CCRandomGenerateBytes,cfgetispeed,cfgetospeed,cfmakeraw,cfsetispeed,cfsetospeed,cfsetspeed,chdir,chflags,chmod,chown,chroot,clearerr,clock_getres,clock_gettime,clock_settime,clonefile,clonefileat,close,closedir,closelog,CMSG_DATA,CMSG_FIRSTHDR,CMSG_LEN,CMSG_NXTHDR,CMSG_SPACE,confstr,connect,connectx,copyfile,copyfile_state_alloc,copyfile_state_free,copyfile_state_get,copyfile_state_set,creat,ctime,ctime_r,devname,difftime,dirfd,dirname,disconnectx,dladdr,dlclose,dlerror,dlopen,dlsym,drand48,dup,dup2,duplocale,endgrent,endpwent,endservent,endutxent,erand48,exchangedata,execl,execle,execlp,execv,execve,execvp,exit,faccessat,fchdir,fchflags,fchmod,fchmodat,fchown,fchownat,fclonefileat,fclose,fcntl,fcopyfile,FD_CLR,FD_ISSET,FD_SET,FD_ZERO,fdopen,fdopendir,feof,ferror,fflush,fgetattrlist,fgetc,fgetpos,fgets,fgetxattr,fileno,flistxattr,flock,fmemopen,fmount,fnmatch,fopen,fork,forkpty,fpathconf,fprintf,fputc,fputs,fread,freadlink,free,freeaddrinfo,freeifaddrs,freelocale,fremovexattr,freopen,fscanf,fseek,fseeko,fsetattrlist,fsetpos,fsetxattr,fstat,fstatat,fstatfs,fstatvfs,fsync,ftell,ftello,ftok,ftruncate,futimens,futimes,fwrite,gai_strerror,getaddrinfo,getattrlist,getattrlistat,getattrlistbulk,getchar,getchar_unlocked,getcwd,getdate,getdomainname,getdtablesize,getegid,getentropy,getenv,geteuid,getfsstat,getgid,getgrent,getgrgid,getgrgid_r,getgrnam,getgrnam_r,getgrouplist,getgroups,gethostid,gethostname,gethostuuid,getifaddrs,getitimer,getline,getloadavg,getlogin,getmntinfo,getnameinfo,getopt,getopt_long,getpeereid,getpeername,getpgid,getpgrp,getpid,getppid,getpriority,getprogname,getprotobyname,getprotobynumber,getpwent,getpwnam,getpwnam_r,getpwuid,getpwuid_r,getrlimit,getrusage,getservbyname,getservbyport,getservent,getsid,getsockname,getsockopt,gettimeofday,getuid,getutxent,getutxid,getutxline,getxattr,glob,globfree,gmtime,gmtime_r,grantpt,host_processor_info,host_statistics,host_statistics64,hstrerror,htonl,htons,iconv,iconv_close,iconv_open,if_freenameindex,if_indextoname,if_nameindex,if_nametoindex,initgroups,ioctl,isalnum,isalpha,isatty,isblank,iscntrl,isdigit,isgraph,islower,isprint,ispunct,isspace,isupper,isxdigit,jrand48,kevent,kevent64,kill,killpg,kqueue,labs,lchown,lcong48,link,linkat,lio_listio,listen,listxattr,localeconv,localeconv_l,localtime,localtime_r,lockf,login_tty,lrand48,lseek,lstat,lutimes,mach_absolute_time,mach_error_string,mach_host_self,mach_task_self,mach_thread_self,mach_timebase_info,mach_vm_map,madvise,major,makedev,malloc,malloc_default_zone,malloc_good_size,malloc_printf,malloc_size,malloc_zone_calloc,malloc_zone_check,malloc_zone_free,malloc_zone_from_ptr,malloc_zone_log,malloc_zone_malloc,malloc_zone_print,malloc_zone_print_ptr_info,malloc_zone_realloc,malloc_zone_statistics,malloc_zone_valloc,memccpy,memchr,memcmp,memcpy,memmem,memmove,memset,memset_pattern16,memset_pattern4,memset_pattern8,memset_s,mincore,minor,mkdir,mkdirat,mkdtemp,mkfifo,mkfifoat,mknod,mknodat,mkstemp,mkstemps,mktime,mlock,mlockall,mmap,mount,mprotect,mrand48,mstats,msync,munlock,munlockall,munmap,nanosleep,newlocale,nice,nl_langinfo,nrand48,ntohl,ntohs,ntp_adjtime,ntp_gettime,open,open_memstream,open_wmemstream,openat,opendir,openlog,openpty,os_log_create,os_log_type_enabled,os_signpost_enabled,os_signpost_id_generate,os_signpost_id_make_with_pointer,os_sync_wait_on_address,os_sync_wait_on_address_with_deadline,os_sync_wait_on_address_with_timeout,os_sync_wake_by_address_all,os_sync_wake_by_address_any,os_unfair_lock_assert_not_owner,os_unfair_lock_assert_owner,os_unfair_lock_lock,os_unfair_lock_trylock,os_unfair_lock_unlock,pathconf,pause,pclose,perror,pipe,poll,popen,posix_madvise,posix_memalign,posix_openpt,posix_spawn,posix_spawn_file_actions_addclose,posix_spawn_file_actions_adddup2,posix_spawn_file_actions_addopen,posix_spawn_file_actions_destroy,posix_spawn_file_actions_init,posix_spawnattr_destroy,posix_spawnattr_get_qos_class_np,posix_spawnattr_getarchpref_np,posix_spawnattr_getbinpref_np,posix_spawnattr_getflags,posix_spawnattr_getpgroup,posix_spawnattr_getsigdefault,posix_spawnattr_getsigmask,posix_spawnattr_init,posix_spawnattr_set_qos_class_np,posix_spawnattr_setarchpref_np,posix_spawnattr_setbinpref_np,posix_spawnattr_setflags,posix_spawnattr_setpgroup,posix_spawnattr_setsigdefault,posix_spawnattr_setsigmask,posix_spawnp,pread,preadv,printf,proc_kmsgbuf,proc_libversion,proc_listallpids,proc_listchildpids,proc_listpgrppids,proc_listpids,proc_name,proc_pid_rusage,proc_pidfdinfo,proc_pidfileportinfo,proc_pidinfo,proc_pidpath,proc_regionfilename,proc_set_csm,proc_set_no_smt,proc_setthread_csm,proc_setthread_no_smt,pselect,pthread_atfork,pthread_attr_destroy,pthread_attr_get_qos_class_np,pthread_attr_getdetachstate,pthread_attr_getinheritsched,pthread_attr_getschedparam,pthread_attr_getschedpolicy,pthread_attr_getscope,pthread_attr_getstackaddr,pthread_attr_getstacksize,pthread_attr_init,pthread_attr_set_qos_class_np,pthread_attr_setdetachstate,pthread_attr_setinheritsched,pthread_attr_setschedparam,pthread_attr_setschedpolicy,pthread_attr_setscope,pthread_attr_setstackaddr,pthread_attr_setstacksize,pthread_cancel,pthread_cond_broadcast,pthread_cond_destroy,pthread_cond_init,pthread_cond_signal,pthread_cond_timedwait,pthread_cond_wait,pthread_condattr_destroy,pthread_condattr_getpshared,pthread_condattr_init,pthread_condattr_setpshared,pthread_cpu_number_np,pthread_create,pthread_create_from_mach_thread,pthread_detach,pthread_equal,pthread_exit,pthread_from_mach_thread_np,pthread_get_qos_class_np,pthread_get_stackaddr_np,pthread_get_stacksize_np,pthread_getname_np,pthread_getschedparam,pthread_getspecific,pthread_introspection_getspecific_np,pthread_introspection_hook_install,pthread_introspection_setspecific_np,pthread_jit_write_freeze_callbacks_np,pthread_jit_write_protect_np,pthread_jit_write_protect_supported_np,pthread_jit_write_with_callback_np,pthread_join,pthread_key_create,pthread_key_delete,pthread_kill,pthread_mach_thread_np,pthread_main_np,pthread_mutex_destroy,pthread_mutex_init,pthread_mutex_lock,pthread_mutex_trylock,pthread_mutex_unlock,pthread_mutexattr_destroy,pthread_mutexattr_getpshared,pthread_mutexattr_init,pthread_mutexattr_setpshared,pthread_mutexattr_settype,pthread_once,pthread_rwlock_destroy,pthread_rwlock_init,pthread_rwlock_rdlock,pthread_rwlock_tryrdlock,pthread_rwlock_trywrlock,pthread_rwlock_unlock,pthread_rwlock_wrlock,pthread_rwlockattr_destroy,pthread_rwlockattr_getpshared,pthread_rwlockattr_init,pthread_rwlockattr_setpshared,pthread_self,pthread_set_qos_class_self_np,pthread_setname_np,pthread_setschedparam,pthread_setspecific,pthread_sigmask,pthread_stack_frame_decode_np,pthread_threadid_np,ptrace,ptsname,putchar,putchar_unlocked,putenv,puts,pututxline,pwrite,pwritev,QCMD,qsort,querylocale,quotactl,raise,rand,read,readdir,readdir_r,readlink,readlinkat,readv,realloc,realpath,recv,recvfrom,recvmsg,regcomp,regerror,regexec,regfree,remove,removexattr,rename,renameat,renameatx_np,renamex_np,res_init,rewind,rewinddir,rmdir,sbrk,scanf,sched_get_priority_max,sched_get_priority_min,sched_yield,seed48,seekdir,select,sem_close,sem_open,sem_post,sem_trywait,sem_unlink,sem_wait,semctl,semget,semop,send,sendfile,sendmsg,sendto,setattrlist,setattrlistat,setbuf,setdomainname,setegid,setenv,seteuid,setgid,setgrent,setgroups,sethostid,sethostname,setitimer,setlocale,setlogin,setlogmask,setpgid,setpriority,setprogname,setpwent,setregid,setreuid,setrlimit,setservent,setsid,setsockopt,settimeofday,setuid,setutxent,setvbuf,setxattr,shm_open,shm_unlink,shmat,shmctl,shmdt,shmget,shutdown,sigaction,sigaddset,sigaltstack,sigdelset,sigemptyset,sigfillset,sigismember,signal,sigpending,sigprocmask,sigsuspend,sigwait,sleep,snprintf,socket,socketpair,sprintf,srand,srand48,sscanf,stat,statfs,statvfs,stpcpy,stpncpy,strcasecmp,strcasestr,strcat,strchr,strcmp,strcoll,strcpy,strcspn,strdup,strerror,strerror_r,strftime,strftime_l,strlen,strncasecmp,strncat,strncmp,strncpy,strndup,strnlen,strpbrk,strptime,strrchr,strsignal,strspn,strstr,strtod,strtof,strtok,strtok_r,strtol,strtoll,strtonum,strtoul,strtoull,strxfrm,symlink,symlinkat,sync,syscall,sysconf,sysctl,sysctlbyname,sysctlnametomib,sysdir_get_next_search_path_enumeration,sysdir_start_search_path_enumeration,syslog,system,task_create,task_for_pid,task_info,task_set_info,task_terminate,task_threads,tcdrain,tcflow,tcflush,tcgetattr,tcgetpgrp,tcgetsid,tcsendbreak,tcsetattr,tcsetpgrp,telldir,thread_info,thread_policy_get,thread_policy_set,time,timegm,times,tmpfile,tmpnam,tolower,toupper,truncate,ttyname,ttyname_r,umask,uname,ungetc,unlink,unlinkat,unlockpt,unmount,unsetenv,uselocale,usleep,utime,utimensat,utimes,utmpxname,vm_allocate,vm_deallocate,VM_MAKE_TAG,wait,wait4,waitid,waitpid,WCOREDUMP,wcslen,wcstombs,WEXITSTATUS,WIFCONTINUED,WIFEXITED,WIFSIGNALED,WIFSTOPPED,wmemchr,write,writev,WSTOPSIG,WTERMSIG

---

### libm

**Structs:** Libm

**Functions:** acos,acosf,acosh,acoshf,asin,asinf,asinh,asinhf,atan,atan2,atan2f,atanf,atanh,atanhf,cbrt,cbrtf,ceil,ceilf,copysign,copysignf,cos,cosf,cosh,coshf,erf,erfc,erfcf,erff,exp,exp10,exp10f,exp2,exp2f,expf,expm1,expm1f,fabs,fabsf,fdim,fdimf,floor,floorf,fma,fmaf,fmax,fmaxf,fmaximum,fmaximum_num,fmaximum_numf,fmaximumf,fmin,fminf,fminimum,fminimum_num,fminimum_numf,fminimumf,fmod,fmodf,frexp,frexpf,hypot,hypotf,ilogb,ilogbf,j0,j0f,j1,j1f,jn,jnf,ldexp,ldexpf,lgamma,lgamma_r,lgammaf,lgammaf_r,log,log10,log10f,log1p,log1pf,log2,log2f,logf,modf,modff,nextafter,nextafterf,pow,powf,remainder,remainderf,remquo,remquof,rint,rintf,round,roundeven,roundevenf,roundf,scalbn,scalbnf,sin,sincos,sincosf,sinf,sinh,sinhf,sqrt,sqrtf,tan,tanf,tanh,tanhf,tgamma,tgammaf,trunc,truncf,y0,y0f,y1,y1f,yn,ynf

---

### memchr

**Modules:**
- `arch`
- `arch/aarch64`
- `arch/all`
- `memmem`

**Structs:** Finder,FinderBuilder,FinderRev,FindIter,FindRevIter,Memchr,Memchr2,Memchr3

**Enums:** Prefilter,PrefilterConfig

**Functions:** find,find_iter,is_equal,is_equal_raw,is_prefix,is_suffix,memchr,memchr_iter,memchr2,memchr2_iter,memchr3,memchr3_iter,memrchr,memrchr_iter,memrchr2,memrchr2_iter,memrchr3,memrchr3_iter,rfind,rfind_iter

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

### src 2

---

### trait 2.impl

---

### type 2.impl

---

### utf8parse

**Structs:** Parser

**Traits:** Receiver

---

### xtask

**Functions:** bundle_run,find_workspace_root,main

---

