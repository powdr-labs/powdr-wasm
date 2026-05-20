mod builtin_functions;
mod compile;
mod proving;
#[cfg(test)]
mod tests;

use clap::{Args, Parser, Subcommand};
use crush::loader::rwm::RWMStages;
use crush::loader::{
    CommonStages, FunctionAsm, FunctionProcessingStage, Module, PartiallyParsedProgram, Statistics,
};
use eyre::Result;
use itertools::Itertools;
use metrics_tracing_context::{MetricsLayer, TracingContextLayer};
use metrics_util::{debugging::DebuggingRecorder, layers::Layer};
use openvm_circuit::arch::VmState;
use openvm_instructions::exe::VmExe;
use openvm_sdk::StdIn;
use openvm_stark_sdk::bench::serialize_metric_snapshot;
use powdr_openvm::{extraction_utils::OriginalVmConfig, program::OriginalCompiledProgram};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::mpsc::channel;
use std::sync::{Mutex, RwLock};
use tracing_forest::ForestLayer;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry, layer::SubscriberExt};

use tracing::Level;
type F = openvm_stark_sdk::p3_baby_bear::BabyBear;

use crate::builtin_functions::BuiltinFunction;
use crush_translation::{Directive, LinkedProgram, OpenVMSettings};

use crush_circuit::CrushConfig;
use powdr_autoprecompiles::PowdrConfig;

#[derive(Parser)]
struct CliArgs {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Args)]
struct PowdrArgs {
    /// Number of APCs to generate
    #[arg(long, default_value_t = 0)]
    apc_count: u64,
    /// Directory to persist all APC candidates + a metrics summary
    #[arg(long)]
    apc_candidates_dir: Option<PathBuf>,
    /// Maximum number of instructions in an APC
    #[arg(long)]
    apc_max_instructions: Option<u32>,
    /// Ignore APCs executed less times than the cutoff
    #[arg(long)]
    apc_exec_count_cutoff: Option<u32>,
}

impl PowdrArgs {
    fn build_powdr_config(&self) -> PowdrConfig {
        let mut config = powdr_openvm::default_powdr_openvm_config(self.apc_count, 0);
        if let Some(ref apc_candidates_dir) = self.apc_candidates_dir {
            config = config.with_apc_candidates_dir(apc_candidates_dir);
        }
        config = config.with_superblocks(1, self.apc_max_instructions, self.apc_exec_count_cutoff);
        config
    }

    fn build_riscv_powdr_config(&self) -> PowdrConfig {
        let mut config = powdr_openvm_riscv::default_powdr_openvm_config(self.apc_count, 0);
        if let Some(ref apc_candidates_dir) = self.apc_candidates_dir {
            config = config.with_apc_candidates_dir(apc_candidates_dir);
        }
        config = config.with_superblocks(1, self.apc_max_instructions, self.apc_exec_count_cutoff);
        config
    }
}

#[derive(Subcommand)]
enum Commands {
    /// Just prints the program's OpenVM instructions
    Print {
        /// Path to the WASM program
        program: String,
        /// Support unaligned memory accesses (needed for e.g. Go-compiled WASM)
        #[arg(long, default_value_t = false)]
        unaligned_memory: bool,
    },
    /// Runs a function from the program with arguments
    Run {
        /// Path to the WASM program
        program: String,
        /// Function name
        function: String,
        /// Guest inputs in the order the guest reads them.
        /// Each value is either a u32 literal or file:<path> for binary file contents.
        #[arg(long)]
        input: Vec<String>,
        /// Path to output metrics JSON file
        #[arg(long)]
        metrics: Option<PathBuf>,
        /// Support unaligned memory accesses (needed for e.g. Go-compiled WASM)
        #[arg(long, default_value_t = false)]
        unaligned_memory: bool,
    },
    /// Compile a WASM program: WASM loading, PGO, APC generation, and keygen.
    /// Outputs a compiled artifact directory that can be used by `prove` or `prove-riscv`.
    Compile {
        /// Path to the WASM program
        program: String,
        /// Function name (entry point)
        function: String,
        /// Guest inputs for PGO profiling (needed when apc_count > 0).
        /// Each value is either a u32 literal or file:<path> for binary file contents.
        #[arg(long)]
        input: Vec<String>,
        #[command(flatten)]
        powdr: PowdrArgs,
        /// Directory to write the compiled artifact to
        #[arg(long)]
        output_dir: PathBuf,
        /// Support unaligned memory accesses (needed for e.g. Go-compiled WASM)
        #[arg(long, default_value_t = false)]
        unaligned_memory: bool,
    },
    /// Compile a RISC-V program: Rust compilation, PGO, APC generation, and keygen.
    /// Outputs a compiled artifact directory that can be used by `prove-riscv`.
    CompileRiscv {
        /// Path to the Rust crate
        program: String,
        /// Guest inputs for PGO profiling (needed when apc_count > 0).
        /// Each value is either a u32 literal or file:<path> for binary file contents.
        #[arg(long)]
        input: Vec<String>,
        #[command(flatten)]
        powdr: PowdrArgs,
        /// Directory to write the compiled artifact to
        #[arg(long)]
        output_dir: PathBuf,
    },
    /// Proves execution of a function from the WASM program with the given arguments
    /// (generates and verifies a full cryptographic proof).
    /// Can either take a WASM program directly (convenience mode) or a pre-compiled
    /// artifact directory from `compile` (benchmark mode, excludes compile overhead from metrics).
    Prove {
        /// Path to the WASM program (convenience mode)
        program: Option<String>,
        /// Function name (required in convenience mode)
        function: Option<String>,
        /// Guest inputs in the order the guest reads them.
        /// Each value is either a u32 literal or file:<path> for binary file contents.
        #[arg(long)]
        input: Vec<String>,
        #[command(flatten)]
        powdr: PowdrArgs,
        /// Also run aggregation (inner recursion) after the app proof
        #[arg(long, default_value_t = false)]
        recursion: bool,
        /// Path to output metrics JSON file
        #[arg(long)]
        metrics: Option<PathBuf>,
        /// Directory with cached proving keys (from `keygen` command, convenience mode only)
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        /// Directory with pre-compiled artifact (from `compile` command)
        #[arg(long)]
        compiled_dir: Option<PathBuf>,
        /// Support unaligned memory accesses (needed for e.g. Go-compiled WASM)
        #[arg(long, default_value_t = false)]
        unaligned_memory: bool,
        /// Max trace height per chip in a segment (must be a power of two,
        /// since traces are padded to the next power of two anyway).
        /// Lower values cause earlier segmentation, reducing peak GPU memory usage.
        /// Default: 2^22 (4194304).
        #[arg(long)]
        max_segment_len: Option<u32>,
    },
    /// Generate and cache proving keys to a directory (for use with `prove --cache-dir`)
    Keygen {
        /// Directory to write cached proving keys to
        cache_dir: PathBuf,
    },
    /// Mock-proves execution of a function from the WASM program with the given arguments
    /// (constraint verification only, no cryptographic proof)
    MockProve {
        /// Path to the WASM program
        program: String,
        /// Function name
        function: String,
        /// Guest inputs in the order the guest reads them.
        /// Each value is either a u32 literal or file:<path> for binary file contents.
        #[arg(long)]
        input: Vec<String>,
        /// Support unaligned memory accesses (needed for e.g. Go-compiled WASM)
        #[arg(long, default_value_t = false)]
        unaligned_memory: bool,
    },
    /// Proves execution of a function from the RISC-V program with the given arguments.
    /// Even though not the main goal of this crate, this is useful for benchmarking against
    /// powdr-wasm.
    /// Can either take a Rust crate directly (convenience mode) or a pre-compiled
    /// artifact directory from `compile-riscv` (benchmark mode).
    ProveRiscv {
        /// Path to the Rust crate (convenience mode)
        program: Option<String>,
        /// Guest inputs in the order the guest reads them.
        /// Each value is either a u32 literal or file:<path> for binary file contents.
        #[arg(long)]
        input: Vec<String>,
        #[command(flatten)]
        powdr: PowdrArgs,
        /// Path to output metrics JSON file
        #[arg(long)]
        metrics: Option<PathBuf>,
        /// Directory with pre-compiled artifact (from `compile-riscv` command)
        #[arg(long)]
        compiled_dir: Option<PathBuf>,
        /// Max trace height per chip in a segment (must be a power of two,
        /// since traces are padded to the next power of two anyway).
        /// Lower values cause earlier segmentation, reducing peak GPU memory usage.
        /// Default: 2^22 (4194304).
        #[arg(long)]
        max_segment_len: Option<u32>,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    setup_tracing_with_log_level(Level::INFO);

    // Parse command line arguments
    let cli_args = CliArgs::parse();
    match cli_args.command {
        Commands::Print {
            program,
            unaligned_memory,
        } => {
            let wasm_bytes = std::fs::read(&program).expect("Failed to read WASM file");
            let (_module, functions) = load_wasm(&wasm_bytes, unaligned_memory);

            for func in &functions {
                println!("Function {}:", func.func_idx);
                for directive in &func.directives {
                    println!("  {directive:?}");
                }
            }
        }
        Commands::Run {
            program,
            function,
            input,
            metrics,
            unaligned_memory,
        } => {
            let wasm_bytes = std::fs::read(&program).expect("Failed to read WASM file");
            let (module, functions) = load_wasm(&wasm_bytes, unaligned_memory);

            // Create and execute program
            let mut linked_program = LinkedProgram::new(module, functions);
            let stdin = make_stdin(&input);

            let run = || -> Result<()> {
                let output = linked_program.execute(CrushConfig::default(), &function, stdin)?;
                println!("output: {output:?}");
                Ok(())
            };

            if let Some(metrics_path) = metrics {
                run_with_metric_collection_to_file(
                    std::fs::File::create(metrics_path).expect("Failed to create metrics file"),
                    run,
                )?;
            } else {
                run()?;
            }
        }
        Commands::Compile {
            program,
            function,
            input,
            powdr,
            output_dir,
            unaligned_memory,
        } => {
            let wasm_bytes = std::fs::read(&program).expect("Failed to read WASM file");
            let original_program =
                load_wasm_original_program(&wasm_bytes, &function, unaligned_memory);
            let stdin = make_stdin(&input);
            let powdr_config = powdr.build_powdr_config();
            compile::compile_crush_to_disk(original_program, stdin, powdr_config, &output_dir)
                .map_err(|e| eyre::eyre!("{e}"))?;
            println!("Compiled to {}", output_dir.display());
        }
        Commands::CompileRiscv {
            program,
            input,
            powdr,
            output_dir,
        } => {
            let stdin = make_stdin(&input);
            let powdr_config = powdr.build_riscv_powdr_config();
            compile::compile_riscv_to_disk(&program, stdin, powdr_config, &output_dir)
                .map_err(|e| eyre::eyre!("{e}"))?;
            println!("Compiled RISC-V to {}", output_dir.display());
        }
        Commands::Prove {
            program,
            function,
            input,
            powdr,
            recursion,
            metrics,
            cache_dir,
            compiled_dir,
            unaligned_memory,
            max_segment_len,
        } => {
            let stdin = make_stdin(&input);

            let prove = || -> Result<()> {
                if let Some(compiled_dir) = compiled_dir {
                    proving::prove_from_compiled(&compiled_dir, stdin, recursion, max_segment_len)
                        .map_err(|e| eyre::eyre!("{e}"))?;
                } else {
                    let program =
                        program.expect("program is required when --compiled-dir is not provided");
                    let function =
                        function.expect("function is required when --compiled-dir is not provided");
                    let wasm_bytes = std::fs::read(&program).expect("Failed to read WASM file");
                    let original_program =
                        load_wasm_original_program(&wasm_bytes, &function, unaligned_memory);
                    let powdr_config = powdr.build_powdr_config();
                    proving::prove(
                        original_program,
                        stdin,
                        recursion,
                        powdr_config,
                        cache_dir.as_deref(),
                        max_segment_len,
                    )
                    .map_err(|e| eyre::eyre!("{e}"))?;
                }
                println!("Proof verified successfully.");
                Ok(())
            };

            if let Some(metrics_path) = metrics {
                run_with_metric_collection_to_file(
                    std::fs::File::create(metrics_path).expect("Failed to create metrics file"),
                    prove,
                )?;
            } else {
                prove()?;
            }
        }
        Commands::Keygen { cache_dir } => {
            proving::keygen_to_disk(&cache_dir)?;
            println!("Keys written to {}", cache_dir.display());
        }
        Commands::MockProve {
            program,
            function,
            input,
            unaligned_memory,
        } => {
            let exe = load_wasm_exe(&program, &function, unaligned_memory);
            let stdin = make_stdin(&input);
            let vm_config = CrushConfig::default();

            let initial_state = VmState::initial(
                &vm_config.system,
                &exe.init_memory,
                exe.pc_start,
                stdin.clone(),
            );

            #[cfg(feature = "cuda")]
            {
                proving::mock_prove_gpu(&exe, initial_state).map_err(|e| eyre::eyre!("{e}"))?;
                println!("GPU mock proof verified successfully.");
            }
            #[cfg(not(feature = "cuda"))]
            {
                proving::mock_prove(&exe, initial_state).map_err(|e| eyre::eyre!("{e}"))?;
                println!("Mock proof verified successfully.");
            }
        }
        Commands::ProveRiscv {
            program,
            input,
            powdr,
            metrics,
            compiled_dir,
            max_segment_len,
        } => {
            let prove = || -> Result<()> {
                if let Some(compiled_dir) = compiled_dir {
                    let stdin = make_stdin(&input);
                    proving::prove_riscv_from_compiled(
                        &compiled_dir,
                        stdin,
                        true,
                        max_segment_len,
                    )
                    .map_err(|e| eyre::eyre!("{e}"))?;
                } else {
                    let program =
                        program.expect("program is required when --compiled-dir is not provided");
                    // Resolve to absolute so compile_openvm finds the right crate
                    let program_abs =
                        std::fs::canonicalize(&program).expect("Failed to resolve program path");
                    let program_str = program_abs.to_str().unwrap();

                    let original = powdr_openvm_riscv::compile_openvm(
                        program_str,
                        powdr_openvm_riscv::GuestOptions::default(),
                    )
                    .map_err(|e| eyre::eyre!("{e}"))?;

                    let config = powdr.build_riscv_powdr_config();
                    let pgo_config = if powdr.apc_count > 0 {
                        let stdin = make_stdin(&input);
                        let execution_profile =
                            powdr_openvm::execution_profile_from_guest(&original, stdin);
                        powdr_openvm_riscv::PgoConfig::Cell(execution_profile, None)
                    } else {
                        powdr_openvm_riscv::PgoConfig::None
                    };
                    let compiled = powdr_openvm_riscv::compile_exe(
                        original,
                        config,
                        pgo_config,
                        powdr_autoprecompiles::empirical_constraints::EmpiricalConstraints::default(
                        ),
                    )
                    .map_err(|e| eyre::eyre!("{e}"))?;

                    let stdin = make_stdin(&input);
                    powdr_openvm_riscv::prove(
                        &compiled,
                        false,
                        true,
                        stdin,
                        max_segment_len.map(|v| v as usize),
                    )
                    .map_err(|e| eyre::eyre!("{e}"))?;
                }
                println!("RISC-V proof verified successfully.");
                Ok(())
            };

            if let Some(metrics_path) = metrics {
                run_with_metric_collection_to_file(
                    std::fs::File::create(metrics_path).expect("Failed to create metrics file"),
                    prove,
                )?;
            } else {
                prove()?;
            }
        }
    }

    Ok(())
}

fn load_wasm_exe(program: &str, function: &str, unaligned_memory: bool) -> VmExe<F> {
    let wasm_bytes = std::fs::read(program).expect("Failed to read WASM file");
    let (module, functions) = load_wasm(&wasm_bytes, unaligned_memory);
    let linked_program = LinkedProgram::new(module, functions);
    linked_program.program_with_entry_point(function)
}

fn load_wasm_original_program<'a>(
    wasm_bytes: &'a [u8],
    function: &str,
    unaligned_memory: bool,
) -> OriginalCompiledProgram<'a, autoprecompiles::CrushISA> {
    let (module, functions) = load_wasm(wasm_bytes, unaligned_memory);
    let linked_program = LinkedProgram::new(module, functions);
    let exe = Arc::new(linked_program.program_with_entry_point(function));
    let vm_config = OriginalVmConfig::new(CrushConfig::default());

    OriginalCompiledProgram {
        exe,
        vm_config,
        linked_program,
    }
}

/// Build StdIn from `--input` values. Each value is either:
/// - a u32 literal (e.g. "42")
/// - `file:<path>` to include raw file bytes
fn make_stdin(inputs: &[String]) -> StdIn {
    let mut stdin = StdIn::default();
    for input in inputs {
        if let Some(path) = input.strip_prefix("file:") {
            stdin.write_bytes(
                &std::fs::read(path).unwrap_or_else(|e| panic!("Failed to read {path}: {e}")),
            );
        } else {
            let val: u32 = input
                .parse()
                .unwrap_or_else(|e| panic!("Invalid u32 input '{input}': {e}"));
            stdin.write(&val);
        }
    }
    stdin
}

fn load_wasm(
    wasm_bytes: &[u8],
    unaligned_memory: bool,
) -> (Module<'_>, Vec<FunctionAsm<Directive<F>>>) {
    let mut settings = OpenVMSettings::<F>::new();
    if unaligned_memory {
        settings = settings.with_unaligned_memory();
    }
    load_wasm_with_settings(wasm_bytes, settings)
}

fn load_wasm_with_settings(
    wasm_bytes: &[u8],
    settings: OpenVMSettings<F>,
) -> (Module<'_>, Vec<FunctionAsm<Directive<F>>>) {
    let PartiallyParsedProgram { s: _, m, functions } =
        crush::loader::load_wasm(settings, wasm_bytes).unwrap();

    let num_functions = functions.len() as u32;
    let tracker = RwLock::new(Some(builtin_functions::Tracker::new(num_functions)));
    let global_stats = Mutex::new(Statistics::default());
    let module = RwLock::new(m);
    let label_gen = AtomicU32::new(0);
    let (jobs_s, jobs_r) = channel();
    let jobs_r = Mutex::new(jobs_r);

    let functions = std::thread::scope(|scope| {
        type FunctionInProcessing<'a> = RWMStages<'a, OpenVMSettings<F>>;
        enum Job<'a> {
            PatchFunc(u32, FunctionInProcessing<'a>),
            FinishFunc(u32, FunctionInProcessing<'a>),
            LoadBuiltin(BuiltinFunction),
        }

        let (patched_s, patched_r) = channel();
        let (final_s, final_r) = channel();

        let num_threads = num_cpus::get().max(1);
        for _ in 0..num_threads {
            let tracker = &tracker;
            let global_stats = &global_stats;
            let label_gen = &label_gen;
            let module = &module;
            let jobs_r = &jobs_r;
            let patched_s = patched_s.clone();
            let final_s = final_s.clone();
            scope.spawn(move || {
                let mut stats = Statistics::default();
                while let Ok(job) = {
                    // This extra scope is needed to release the lock before processing the function.
                    jobs_r.lock().unwrap().recv()
                } {
                    match job {
                        Job::PatchFunc(func_idx, mut func) => {
                            // Advance to BlocklessDag stage
                            let module = module.read().unwrap();
                            let dag = loop {
                                if let RWMStages::CommonStages(CommonStages::BlocklessDag(dag)) =
                                    &mut func
                                {
                                    break dag;
                                }
                                func = func
                                    .advance_stage(
                                        &settings,
                                        &module,
                                        func_idx,
                                        label_gen,
                                        Some(&mut stats),
                                    )
                                    .unwrap();
                            };

                            // In the BlocklessDag stage, we need to find instructions we don't
                            // implement and replace them with function calls to built-in functions.
                            tracker
                                .read()
                                .unwrap()
                                .as_ref()
                                .unwrap()
                                .replace_with_builtins(dag);

                            patched_s.send((func_idx, func)).unwrap();
                        }
                        Job::FinishFunc(func_idx, func) => {
                            let module = module.read().unwrap();
                            let func = func
                                .advance_all_stages(
                                    &settings,
                                    &module,
                                    func_idx,
                                    label_gen,
                                    Some(&mut stats),
                                )
                                .unwrap();
                            final_s.send((func_idx, func)).unwrap();
                        }
                        Job::LoadBuiltin(builtin) => {
                            let func_idx = builtin.function_index;
                            let func = builtin.load(&module.read().unwrap(), label_gen);
                            final_s.send((func_idx, func)).unwrap();
                        }
                    }
                }
                *global_stats.lock().unwrap() += stats;
            });
        }
        // Drop the channel endpoints this thread will no longer use.
        drop(final_s);
        drop(patched_s);

        // Send the functions for processing.
        for (idx, func) in functions.into_iter().enumerate() {
            let rwm_func: FunctionInProcessing<'_> = func.into();
            jobs_s.send(Job::PatchFunc(idx as u32, rwm_func)).unwrap();
        }

        // Received the patched functions, update the module, and resend for processing.
        let patched_functions = patched_r
            .into_iter()
            .take(num_functions as usize)
            .collect_vec();
        let tracker = tracker.write().unwrap().take().unwrap();
        let used_builtins = tracker.into_used_builtins().collect_vec();
        {
            let mut module = module.write().unwrap();
            for bf in &used_builtins {
                let given_idx = module.append_function(bf.definition.params, bf.definition.results);
                assert_eq!(given_idx, bf.function_index);
            }
        }

        // Now that module was updated, send the functions back to resume processing.
        for (idx, func) in patched_functions {
            jobs_s.send(Job::FinishFunc(idx, func)).unwrap();
        }

        // And also send the built-in functions to be loaded
        for bf in used_builtins {
            jobs_s.send(Job::LoadBuiltin(bf)).unwrap();
        }

        // Signal threads there are no more jobs.
        drop(jobs_s);

        // Receive the fully processed functions, which included the loaded builtins.
        final_r
            .into_iter()
            .sorted_unstable_by_key(|f| f.0)
            .map(|f| f.1)
            .collect_vec()
    });

    println!(
        "crush loading statistics: {}",
        global_stats.into_inner().unwrap()
    );

    (module.into_inner().unwrap(), functions)
}

pub fn setup_tracing_with_log_level(level: Level) {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(format!("{level},p3_=warn")));
    let _ = Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .with(MetricsLayer::new())
        .with(openvm_stark_sdk::metrics_tracing::TimingMetricsLayer::new())
        .try_init();
}

/// export stark-backend metrics to the given file
fn run_with_metric_collection_to_file<R>(file: std::fs::File, f: impl FnOnce() -> R) -> R {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let recorder = TracingContextLayer::all().layer(recorder);
    metrics::set_global_recorder(recorder).unwrap();
    let res = f();

    serde_json::to_writer_pretty(&file, &serialize_metric_snapshot(snapshotter.snapshot()))
        .unwrap();
    res
}
