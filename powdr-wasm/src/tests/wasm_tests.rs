use crate::proving::mock_prove;
use crate::*;
use openvm_circuit::arch::VmState;
use openvm_sdk::StdIn;
use serde::Deserialize;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tracing::Level;

use super::helpers;

#[derive(Debug, Clone)]
enum ExpectedResult {
    Values(Vec<u32>),
    NanCanonical { is_f64: bool },
    NanArithmetic { is_f64: bool },
}

type TestCase = (String, Vec<u32>, ExpectedResult);
type TestModule = (String, u32, Vec<TestCase>);

#[derive(Debug, Deserialize)]
struct TestFile {
    commands: Vec<CommandEntry>,
}

#[derive(Debug, Deserialize)]
struct CommandEntry {
    #[serde(rename = "type")]
    cmd_type: String,
    filename: Option<String>,
    line: Option<u32>,
    action: Option<Action>,
    expected: Option<Vec<Expected>>,
}

#[derive(Debug, Deserialize)]
struct Action {
    #[serde(rename = "type")]
    action_type: String,
    field: Option<String>,
    args: Option<Vec<Value>>,
    #[allow(dead_code)]
    module: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Expected {
    #[serde(rename = "type")]
    expected_type: String,
    #[allow(dead_code)]
    lane: Option<String>,
    value: Option<String>,
}

fn extract_wast_test_info(
    wast_file: &str,
) -> Result<(PathBuf, Vec<TestModule>), Box<dyn std::error::Error>> {
    // Convert .wast to .json using wast2json
    let target_dir =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("../wast_target");
    fs::create_dir_all(&target_dir).unwrap();
    let wast_path = Path::new(wast_file).canonicalize()?;
    let json_path =
        target_dir.join(Path::new(wast_path.file_stem().unwrap()).with_extension("json"));

    let output = Command::new("wast2json")
        .arg(wast_path)
        .arg("--debug-names")
        .current_dir(&target_dir)
        .output()
        .unwrap();

    if !output.status.success() {
        return Err(format!(
            "wast2json failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    // Parse the JSON file
    let json_content = fs::read_to_string(&json_path)?;
    let test_file: TestFile = serde_json::from_str(&json_content)?;

    let mut test_cases = Vec::new();
    let mut current_module = None;
    let mut current_line = 0;
    let mut assert_cases = Vec::new();

    for cmd in test_file.commands {
        match cmd.cmd_type.as_str() {
            "module" => {
                if let Some(module) = current_module.take()
                    && !assert_cases.is_empty()
                {
                    test_cases.push((module, current_line, assert_cases.clone()));
                    assert_cases.clear();
                }
                current_module = cmd.filename;
                current_line = cmd.line.unwrap_or(0);
            }
            "action" | "assert_return" => {
                if let (Some(action), Some(expected)) = (cmd.action, cmd.expected)
                    && action.action_type == "invoke"
                    && let (Some(field), Some(args)) = (action.field, action.args)
                {
                    let args_u32: Vec<u32> = args
                        .iter()
                        .filter_map(|v| {
                            if let Value::Object(obj) = v {
                                if let Some(Value::String(val_str)) = obj.get("value") {
                                    // In OpenVM we read the inputs as u32s, so here we
                                    // need to parse the input as 32-bit limbs.
                                    if let Some(Value::String(ty_str)) = obj.get("type") {
                                        parse_as_vec_u32(ty_str, val_str)
                                    } else {
                                        Some(vec![val_str.parse::<u32>().unwrap()])
                                    }
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        })
                        .flatten()
                        .collect();

                    // Check if any expected value is a NaN assertion
                    let expected_result = if expected.len() == 1 {
                        let e = &expected[0];
                        match e.value.as_deref() {
                            Some("nan:canonical") => ExpectedResult::NanCanonical {
                                is_f64: e.expected_type == "f64",
                            },
                            Some("nan:arithmetic") => ExpectedResult::NanArithmetic {
                                is_f64: e.expected_type == "f64",
                            },
                            _ => {
                                let vals: Vec<u32> = expected
                                    .iter()
                                    .filter_map(|e| {
                                        e.value
                                            .as_ref()
                                            .and_then(|v| parse_as_vec_u32(&e.expected_type, v))
                                    })
                                    .flatten()
                                    .collect();
                                ExpectedResult::Values(vals)
                            }
                        }
                    } else {
                        let vals: Vec<u32> = expected
                            .iter()
                            .filter_map(|e| {
                                e.value
                                    .as_ref()
                                    .and_then(|v| parse_as_vec_u32(&e.expected_type, v))
                            })
                            .flatten()
                            .collect();
                        ExpectedResult::Values(vals)
                    };

                    assert_cases.push((field, args_u32, expected_result));
                }
            }
            _ => {}
        }
    }

    if let Some(module) = current_module
        && !assert_cases.is_empty()
    {
        test_cases.push((module, current_line, assert_cases));
    }

    Ok((target_dir, test_cases))
}

fn parse_as_vec_u32(ty: &str, value: &str) -> Option<Vec<u32>> {
    match ty {
        "i32" => {
            let v = value.parse::<u32>().unwrap();
            Some(vec![v])
        }
        "i64" => {
            let v = value.parse::<u64>().unwrap();
            Some(vec![v as u32, (v >> 32) as u32])
        }
        "f32" => {
            // wast2json represents f32 as decimal u32 bit pattern
            let v = value.parse::<u32>().unwrap();
            Some(vec![v])
        }
        "f64" => {
            // wast2json represents f64 as decimal u64 bit pattern
            let v = value.parse::<u64>().unwrap();
            Some(vec![v as u32, (v >> 32) as u32])
        }
        _ => None,
    }
}

#[allow(dead_code)]
fn parse_val(s: &str) -> Result<u32, Box<dyn std::error::Error>> {
    if s.starts_with("i32.const ") {
        let val_str = s.trim_start_matches("i32.const ").trim();
        if val_str.starts_with("0x") {
            u32::from_str_radix(val_str.trim_start_matches("0x"), 16).map_err(|e| e.into())
        } else if val_str.starts_with("-0x") {
            u32::from_str_radix(val_str.trim_start_matches("-0x"), 16)
                .map(|v| (!v).wrapping_add(1))
                .map_err(|e| e.into())
        } else if val_str.starts_with("-") {
            val_str
                .parse::<i32>()
                .map(|v| v as u32)
                .map_err(|e| e.into())
        } else {
            val_str.parse::<u32>().map_err(|e| e.into())
        }
    } else {
        Err("Unsupported value format".into())
    }
}

fn load_wasm_module(wasm_bytes: &[u8]) -> LinkedProgram<'_, F> {
    let (module, functions) = load_wasm(wasm_bytes, false);
    LinkedProgram::new(module, functions)
}

fn load_wasm_module_with_settings(
    wasm_bytes: &[u8],
    settings: OpenVMSettings<F>,
) -> LinkedProgram<'_, F> {
    let (module, functions) = load_wasm_with_settings(wasm_bytes, settings);
    LinkedProgram::new(module, functions)
}

fn run_and_prove_single_wasm_test(
    module_path: &str,
    function: &str,
    args: &[u32],
    expected: &[u32],
    byte_inputs: &[&[u8]],
) -> Result<(), Box<dyn std::error::Error>> {
    let wasm_bytes = std::fs::read(module_path).expect("Failed to read WASM file");
    let mut module = load_wasm_module(&wasm_bytes);
    run_wasm_test_function(&mut module, function, args, expected, true, byte_inputs)
}

/// Run a WASM program through execution with output verification.
/// When `prove` is true, also runs metered execution, preflight, and mock
/// proof (all stages). Supports multi-segment programs.
fn run_wasm_test_function(
    module: &mut LinkedProgram<F>,
    function: &str,
    args: &[u32],
    expected: &[u32],
    prove: bool,
    byte_inputs: &[&[u8]],
) -> Result<(), Box<dyn std::error::Error>> {
    let output =
        run_wasm_test_function_raw(module, function, args, expected.len(), prove, byte_inputs)?;
    if !expected.is_empty() {
        assert_eq!(
            output, expected,
            "Test failed for {function}({args:?}): expected {expected:?}, got {output:?}"
        );
    }
    Ok(())
}

/// Run a WASM program and return the output as Vec<u32>.
/// When `prove` is true, also runs metered execution, preflight, and mock proof.
fn run_wasm_test_function_raw(
    module: &mut LinkedProgram<F>,
    function: &str,
    args: &[u32],
    output_words: usize,
    prove: bool,
    byte_inputs: &[&[u8]],
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    run_wasm_test_function_raw_with_config(
        module,
        function,
        args,
        output_words,
        prove,
        byte_inputs,
        CrushConfig::default(),
    )
}

fn run_wasm_test_function_raw_with_config(
    module: &mut LinkedProgram<F>,
    function: &str,
    args: &[u32],
    output_words: usize,
    prove: bool,
    byte_inputs: &[&[u8]],
    vm_config: CrushConfig,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    setup_tracing_with_log_level(Level::WARN);
    println!("Running WASM test with {function}({args:?}): output_words={output_words}");

    // Capture the exe before module.execute() mutates memory_image.
    let exe = module.program_with_entry_point(function);

    let make_stdin = || {
        let mut stdin = StdIn::default();
        for &arg in args {
            stdin.write(&arg);
        }
        for bytes in byte_inputs {
            stdin.write_bytes(bytes);
        }
        stdin
    };

    let initial_state = VmState::initial(
        &vm_config.system,
        &exe.init_memory,
        exe.pc_start,
        make_stdin(),
    );

    // Execution (also updates module.memory_image for wast test reuse)
    println!("  Execution");
    let output_bytes = module.execute(vm_config.clone(), function, make_stdin())?;

    let output: Vec<u32> = if output_words > 0 {
        output_bytes[..output_words * 4]
            .chunks(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    } else {
        vec![]
    };

    if !prove {
        return Ok(output);
    }

    // Metered execution
    println!("  Metered execution");
    let (segments, _) =
        helpers::test_metered_execution(vm_config.clone(), &exe, initial_state.clone())?;
    let total_insns: u64 = segments.iter().map(|s| s.num_insns).sum();
    println!(
        "    {} segment(s), {} total instructions",
        segments.len(),
        total_insns
    );

    // Preflight
    println!("  Preflight");
    helpers::test_preflight(vm_config.clone(), &exe, initial_state.clone())?;

    // Mock proof (CPU)
    println!("  Mock proof (CPU)");
    mock_prove(vm_config.clone(), &exe, initial_state.clone())?;

    // Mock proof (GPU)
    #[cfg(feature = "cuda")]
    {
        println!("  Mock proof (GPU)");
        crate::proving::mock_prove_gpu(vm_config, &exe, initial_state)?;
    }

    Ok(output)
}

#[test]
fn test_i32() {
    run_wasm_test("../wasm_tests/i32.wast").unwrap()
}

#[test]
fn test_i64() {
    run_wasm_test("../wasm_tests/i64.wast").unwrap()
}

#[test]
fn test_f32() {
    run_wasm_test("../wasm_tests/f32.wast").unwrap()
}

#[test]
fn test_f64() {
    run_wasm_test("../wasm_tests/f64.wast").unwrap()
}

#[test]
fn test_f32_official() {
    run_wasm_test("../wasm_tests/f32_official.wast").unwrap()
}

#[test]
fn test_f64_official() {
    run_wasm_test("../wasm_tests/f64_official.wast").unwrap()
}

#[test]
fn test_f32_cmp() {
    run_wasm_test("../wasm_tests/f32_cmp.wast").unwrap()
}

#[test]
fn test_f64_cmp() {
    run_wasm_test("../wasm_tests/f64_cmp.wast").unwrap()
}

#[test]
fn test_f32_bitwise() {
    run_wasm_test("../wasm_tests/f32_bitwise.wast").unwrap()
}

#[test]
fn test_f64_bitwise() {
    run_wasm_test("../wasm_tests/f64_bitwise.wast").unwrap()
}

#[test]
fn test_conversions() {
    run_wasm_test("../wasm_tests/conversions.wast").unwrap()
}

#[test]
fn test_address() {
    run_wasm_test("../wasm_tests/address.wast").unwrap()
}

#[test]
fn test_memory_grow() {
    run_wasm_test("../wasm_tests/memory_grow.wast").unwrap()
}

#[test]
fn test_call_indirect() {
    run_wasm_test("../wasm_tests/call_indirect.wast").unwrap()
}

#[test]
fn test_func() {
    run_wasm_test("../wasm_tests/func.wast").unwrap()
}

#[test]
fn test_call() {
    run_wasm_test("../wasm_tests/call.wast").unwrap()
}

#[test]
fn test_br_if() {
    run_wasm_test("../wasm_tests/br_if.wast").unwrap()
}

#[test]
fn test_return() {
    run_wasm_test("../wasm_tests/return.wast").unwrap()
}

#[test]
fn test_loop() {
    run_wasm_test("../wasm_tests/loop.wast").unwrap()
}

#[test]
fn test_memory_fill() {
    run_wasm_test("../wasm_tests/memory_fill.wast").unwrap()
}

#[test]
fn test_unaligned_memory_with_assume_unaligned() {
    run_wasm_test_with_settings(
        "../wasm_tests/unaligned.wast",
        OpenVMSettings::new().with_unaligned_memory(),
    )
    .unwrap()
}

#[test]
fn test_address_aligned_memory_with_assume_unaligned() {
    // Run the standard address.wast tests with unaligned memory support enabled.
    // This is a regression test: the byte-by-byte path must produce the same
    // results as the alignment-trusting path for naturally aligned data.
    run_wasm_test_with_settings(
        "../wasm_tests/address.wast",
        OpenVMSettings::new().with_unaligned_memory(),
    )
    .unwrap()
}

fn run_wasm_test(tf: &str) -> Result<(), Box<dyn std::error::Error>> {
    run_wasm_test_with_settings(tf, OpenVMSettings::new())
}

fn run_wasm_test_with_settings(
    tf: &str,
    settings: OpenVMSettings<F>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (target_dir, test_cases) = extract_wast_test_info(tf)?;

    // Run all test cases
    for (module_path, _line, cases) in &test_cases {
        let full_module_path = target_dir.join(module_path);

        // Load the module to be executed multiple times.
        println!("Loading test module: {module_path}");
        let wasm_bytes = std::fs::read(full_module_path).expect("Failed to read WASM file");
        let mut module = load_wasm_module_with_settings(&wasm_bytes, settings);

        for (function, args, expected) in cases {
            match expected {
                ExpectedResult::Values(vals) => {
                    run_wasm_test_function(&mut module, function, args, vals, false, &[])?;
                }
                ExpectedResult::NanCanonical { is_f64 } => {
                    let output_words = if *is_f64 { 2 } else { 1 };
                    let output = run_wasm_test_function_raw(
                        &mut module,
                        function,
                        args,
                        output_words,
                        false,
                        &[],
                    )?;
                    check_nan_canonical(&output, *is_f64, function, args);
                }
                ExpectedResult::NanArithmetic { is_f64 } => {
                    let output_words = if *is_f64 { 2 } else { 1 };
                    let output = run_wasm_test_function_raw(
                        &mut module,
                        function,
                        args,
                        output_words,
                        false,
                        &[],
                    )?;
                    check_nan_arithmetic(&output, *is_f64, function, args);
                }
            }
        }
    }

    Ok(())
}

fn check_nan_canonical(output: &[u32], is_f64: bool, function: &str, args: &[u32]) {
    if is_f64 {
        assert!(output.len() >= 2, "Expected 2 words for f64 NaN check");
        let bits = (output[0] as u64) | ((output[1] as u64) << 32);
        // Canonical NaN: sign bit can be either, exponent all 1s, mantissa = 1<<51
        assert!(
            (bits & 0x7FFFFFFFFFFFFFFF) == 0x7FF8000000000000,
            "Test failed for {function}({args:?}): expected canonical f64 NaN, got bits {bits:#018x}"
        );
    } else {
        assert!(!output.is_empty(), "Expected 1 word for f32 NaN check");
        let bits = output[0];
        // Canonical NaN: sign bit can be either, exponent all 1s, mantissa = 1<<22
        assert!(
            (bits & 0x7FFFFFFF) == 0x7FC00000,
            "Test failed for {function}({args:?}): expected canonical f32 NaN, got bits {bits:#010x}"
        );
    }
}

fn check_nan_arithmetic(output: &[u32], is_f64: bool, function: &str, args: &[u32]) {
    if is_f64 {
        assert!(output.len() >= 2, "Expected 2 words for f64 NaN check");
        let bits = (output[0] as u64) | ((output[1] as u64) << 32);
        // Arithmetic NaN: any quiet NaN (exponent all 1s, quiet bit set)
        assert!(
            (bits & 0x7FF8000000000000) == 0x7FF8000000000000,
            "Test failed for {function}({args:?}): expected arithmetic f64 NaN, got bits {bits:#018x}"
        );
    } else {
        assert!(!output.is_empty(), "Expected 1 word for f32 NaN check");
        let bits = output[0];
        // Arithmetic NaN: any quiet NaN (exponent all 1s, quiet bit set)
        assert!(
            (bits & 0x7FC00000) == 0x7FC00000,
            "Test failed for {function}({args:?}): expected arithmetic f32 NaN, got bits {bits:#010x}"
        );
    }
}

#[test]
fn test_fib() {
    run_and_prove_single_wasm_test("../sample-programs/fib_loop.wasm", "fib", &[10], &[55], &[])
        .unwrap()
}

#[test]
fn test_n_first_sums() {
    run_and_prove_single_wasm_test(
        "../sample-programs/n_first_sum.wasm",
        "n_first_sum",
        &[42, 0],
        &[903, 0],
        &[],
    )
    .unwrap()
}

#[test]
fn test_call_indirect_wasm() {
    run_and_prove_single_wasm_test(
        "../sample-programs/call_indirect.wasm",
        "test",
        &[],
        &[1],
        &[],
    )
    .unwrap();
    run_and_prove_single_wasm_test(
        "../sample-programs/call_indirect.wasm",
        "call_op",
        &[0, 10, 20],
        &[30],
        &[],
    )
    .unwrap();
    run_and_prove_single_wasm_test(
        "../sample-programs/call_indirect.wasm",
        "call_op",
        &[1, 10, 3],
        &[7],
        &[],
    )
    .unwrap();
}

#[test]
fn test_keccak() {
    run_and_prove_single_wasm_test("../sample-programs/keccak.wasm", "main", &[0, 0], &[], &[])
        .unwrap()
}

#[test]
fn test_keeper_js() {
    // This is program is a stripped down version of geth, compiled for Go's js target.
    // Source: https://github.com/ethereum/go-ethereum/tree/master/cmd/keeper
    // Compile command:
    //   GOOS=js GOARCH=wasm go -gcflags=all=-d=softfloat build -tags "example" -o keeper.wasm
    run_and_prove_single_wasm_test(
        "../sample-programs/keeper_js.wasm",
        "run",
        &[0, 0],
        &[],
        &[],
    )
    .unwrap();
}

#[test]
fn test_keeper_wasi() {
    // Same keeper program, compiled for WASI (wasip1) target.
    // Compile command:
    //   GOOS=wasip1 GOARCH=wasm go build -gcflags=all=-d=softfloat -tags "crush" -o keeper_wasi.wasm
    let payload = std::fs::read("../sample-programs/keeper/hoodi_payload.bin")
        .expect("failed to read hoodi_payload.bin");
    let wasm_bytes =
        std::fs::read("../sample-programs/keeper_wasi.wasm").expect("failed to read WASM file");
    let mut module =
        load_wasm_module_with_settings(&wasm_bytes, OpenVMSettings::new().with_unaligned_memory());

    // Capture exe before execute() mutates memory_image.
    let exe = module.program_with_entry_point("_start");
    let vm_config = CrushConfig::default();

    run_wasm_test_function(&mut module, "_start", &[], &[], false, &[&payload]).unwrap();

    // Run metered execution to report instruction count.
    let mut stdin = StdIn::default();
    stdin.write_bytes(&payload);
    let initial_state = VmState::initial(&vm_config.system, &exe.init_memory, exe.pc_start, stdin);
    let (segments, _) =
        helpers::test_metered_execution(CrushConfig::default(), &exe, initial_state).unwrap();
    let total_insns: u64 = segments.iter().map(|s| s.num_insns).sum();
    println!(
        "  keeper_wasi: {} segment(s), {} total instructions",
        segments.len(),
        total_insns
    );
}

#[test]
fn test_keeper_decode_only() {
    // Keeper with only input reading + RLP deserialization (no ExecuteStateless).
    // Used to measure the cycle cost of deserialization alone.
    let payload = std::fs::read("../sample-programs/keeper/hoodi_payload.bin")
        .expect("failed to read hoodi_payload.bin");
    let wasm_bytes = std::fs::read("../sample-programs/keeper_decode_only.wasm")
        .expect("failed to read WASM file");
    let mut module =
        load_wasm_module_with_settings(&wasm_bytes, OpenVMSettings::new().with_unaligned_memory());

    // Capture exe before execute() mutates memory_image.
    let exe = module.program_with_entry_point("_start");
    let vm_config = CrushConfig::default();

    run_wasm_test_function(&mut module, "_start", &[], &[], true, &[&payload]).unwrap();

    // Run metered execution to report instruction count.
    let mut stdin = StdIn::default();
    stdin.write_bytes(&payload);
    let initial_state = VmState::initial(&vm_config.system, &exe.init_memory, exe.pc_start, stdin);
    let (segments, _) =
        helpers::test_metered_execution(CrushConfig::default(), &exe, initial_state).unwrap();
    let total_insns: u64 = segments.iter().map(|s| s.num_insns).sum();
    println!(
        "  keeper_decode_only: {} segment(s), {} total instructions",
        segments.len(),
        total_insns
    );
}

fn keccak_rust_crush(iterations: u32, expected_first_byte: u32) {
    run_crush_guest(
        "keccak_with_inputs",
        "main",
        &[0, 0],
        &[iterations, expected_first_byte],
        &[],
        &[],
    )
}

fn keccak_precompile_crush(iterations: u32, expected_first_byte: u32) {
    let path = format!(
        "{}/../sample-programs/keccak_precompile",
        env!("CARGO_MANIFEST_DIR")
    );
    build_wasm(&PathBuf::from(&path));
    let wasm_path = format!("{path}/target/wasm32-unknown-unknown/release/keccak_precompile.wasm");
    let wasm_bytes = std::fs::read(&wasm_path).expect("Failed to read WASM file");
    let mut module = load_wasm_module(&wasm_bytes);
    run_wasm_test_function_raw_with_config(
        &mut module,
        "main",
        &[0, 0, iterations, expected_first_byte],
        0,
        true,
        &[],
        CrushConfig::default().with_keccak(),
    )
    .unwrap();
}

#[test]
fn test_keccak_rust_crush_1() {
    // keccak([0; 32]) = [0x29, ...], 0x29 = 41
    keccak_rust_crush(1, 41);
}

#[test]
fn test_keccak_rust_crush_2() {
    // keccak^2([0; 32]) = [0x51, ...], 0x51 = 81
    keccak_rust_crush(2, 81);
}

#[test]
fn test_keccak_rust_crush_3() {
    // keccak^3([0; 32]) = [0x35, ...], 0x35 = 53
    keccak_rust_crush(3, 53);
}

#[test]
fn test_keccak_precompile_crush_1() {
    keccak_precompile_crush(1, 41);
}

#[test]
fn test_keccak_precompile_crush_2() {
    keccak_precompile_crush(2, 81);
}

#[test]
#[should_panic]
fn test_keccak_rust_crush_1_wrong() {
    keccak_rust_crush(1, 42);
}

#[test]
#[should_panic]
fn test_keccak_rust_crush_2_wrong() {
    keccak_rust_crush(2, 82);
}

#[test]
#[should_panic]
fn test_keccak_rust_crush_3_wrong() {
    keccak_rust_crush(3, 54);
}

fn u256_matmul_crush(reps: u32) {
    run_crush_guest("u256_matmul", "main", &[0, 0], &[reps], &[], &[])
}

#[test]
fn test_u256_crush_1() {
    u256_matmul_crush(1);
}

#[test]
fn test_u256_crush_2() {
    u256_matmul_crush(2);
}

#[test]
fn test_keccak_rust_read_vec() {
    run_crush_guest(
        "read_vec",
        "main",
        &[0, 0],
        &[0xffaabbcc, 0xeedd0066],
        &[],
        &[],
    )
}

#[test]
fn test_read_serde() {
    // Serialize SampleData { label: "hello", value: 1_000_000 } with postcard
    // on the host, pass raw bytes via write_bytes, and let the guest deserialize.
    // postcard uses varint encoding for u64 and length-prefixed strings,
    // so the wire format differs from the raw field layout.
    #[derive(serde::Serialize)]
    struct SampleData {
        label: String,
        value: u64,
    }

    let data = SampleData {
        label: "hello".to_string(),
        value: 1_000_000,
    };
    let bytes = postcard::to_allocvec(&data).unwrap();

    run_crush_guest("read_serde", "main", &[0, 0], &[], &[], &[&bytes])
}

fn run_eth_block(block_input: &str, prove: bool) {
    let wasm_path = format!(
        "{}/../sample-programs/eth-block/openvm-client-eth.wasm",
        env!("CARGO_MANIFEST_DIR")
    );
    let input_path = format!(
        "{}/../sample-programs/eth-block/{block_input}",
        env!("CARGO_MANIFEST_DIR")
    );
    let input_bytes = std::fs::read(&input_path).expect("Failed to read block input");
    let wasm_bytes = std::fs::read(&wasm_path).expect("Failed to read WASM file");
    let mut module = load_wasm_module(&wasm_bytes);
    run_wasm_test_function(&mut module, "main", &[0, 0], &[], prove, &[&input_bytes]).unwrap()
}

#[test]
fn test_eth_block_1() {
    run_eth_block("1.bin", true);
}

#[test]
fn test_eth_block_24171377() {
    run_eth_block("24171377.bin", false);
}

#[test]
fn test_eth_block_24171384() {
    run_eth_block("24171384.bin", false);
}

fn run_crush_guest(
    case: &str,
    main_function: &str,
    func_inputs: &[u32],
    data_inputs: &[u32],
    outputs: &[u32],
    byte_inputs: &[&[u8]],
) {
    let path = format!("{}/../sample-programs/{case}", env!("CARGO_MANIFEST_DIR"));
    build_wasm(&PathBuf::from(&path));
    let wasm_path = format!("{path}/target/wasm32-unknown-unknown/release/{case}.wasm",);
    let args = func_inputs
        .iter()
        .chain(data_inputs)
        .copied()
        .collect::<Vec<_>>();
    run_and_prove_single_wasm_test(&wasm_path, main_function, &args, outputs, byte_inputs).unwrap()
}

fn build_wasm(path: &PathBuf) {
    assert!(path.exists(), "Target directory does not exist: {path:?}",);

    let output = Command::new("cargo")
        .arg("build")
        .arg("--release")
        .arg("--target")
        .arg("wasm32-unknown-unknown")
        .current_dir(path)
        .output()
        .expect("Failed to run cargo build");

    if !output.status.success() {
        eprintln!("stderr:\n{}", String::from_utf8_lossy(&output.stderr));
        eprintln!("stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    }

    assert!(output.status.success(), "cargo build failed for {path:?}",);
}

// ==================== Go Keccak Guest Tests ====================

fn go_keccak_test(wasm_path: &str, iterations: u32, expected_byte: u32, keccak: bool) {
    let wasm_bytes = std::fs::read(wasm_path).expect("failed to read Go keccak WASM file");
    let mut module =
        load_wasm_module_with_settings(&wasm_bytes, OpenVMSettings::new().with_unaligned_memory());

    let vm_config = if keccak {
        CrushConfig::default().with_keccak()
    } else {
        CrushConfig::default()
    };

    run_wasm_test_function_raw_with_config(
        &mut module,
        "_start",
        &[iterations, expected_byte],
        0,
        true,
        &[],
        vm_config,
    )
    .unwrap();
}

#[test]
fn test_go_keccak_software_1() {
    go_keccak_test("../sample-programs/go_keccak_software.wasm", 1, 41, false);
}

#[test]
fn test_go_keccak_precompile_1() {
    go_keccak_test("../sample-programs/go_keccak_precompile.wasm", 1, 41, true);
}
