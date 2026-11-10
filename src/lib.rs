pub mod bias;
mod c_bridge;
pub mod conv;
pub mod data_format_conversion;
pub mod data_structures;
pub mod errors;
mod global_settings;
pub mod linear;
pub mod memory_management;
pub mod non_linear;
pub mod norm;
pub mod pooling;
#[cfg(feature = "python-bridge")]
mod python_bridge;
mod qant_driver_import;
pub mod utils;

use crate::data_structures::QantTensor1;
use errors::ToolkitError;

// Needed by Rust callers of `memory_management::QantMemoryHandle`: without the
// re-export the type is unnameable outside this crate and the device-memory
// API is reachable only through the C and Python bridges.
pub use qant_driver_import::QantDataRole;
pub use qant_driver_import::driver_get_available_npus as get_available_npus;
pub use qant_driver_import::driver_get_device_info as get_device_info;
pub use qant_driver_import::driver_get_perf_counter as get_perf_counter;
pub use qant_driver_import::driver_get_version as get_driver_info;
pub use qant_driver_import::driver_init_npu as init_npu;
pub use qant_driver_import::driver_release_npu as release_npu;
pub use qant_driver_import::driver_reset_perf_counter as reset_perf_counter;
pub use qant_driver_import::driver_setup_logging as setup_logging;

/// Multiplies elementwise us and vs, for any shape
pub fn mul_elementwise_bf16<'a, 'b>(
    npu_id: u32,
    us: &QantTensor1<'a>,
    vs: &QantTensor1<'a>,
) -> Result<QantTensor1<'b>, errors::ToolkitError> {
    if us.shape() != vs.shape() {
        return Err(toolkit_error!(
            "us and vs must have the same length, got {0} and {1}",
            us.shape()[0],
            vs.shape()[0]
        ));
    }

    let ws = qant_driver_import::driver_mul(npu_id, us.data(), vs.data())?;

    QantTensor1::new(ws, *us.shape())
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use crate::{
        data_format_conversion::to_bf16s, reset_perf_counter,
        utils::test_utils::assert_almost_equal_vec,
    };

    use super::*;

    #[test]
    fn init_release() {
        init_npu(0).unwrap();
        release_npu(0).unwrap();
    }

    #[test]
    fn multithreading() {
        let mut handles = vec![];

        for _ in 0..100 {
            let handle = std::thread::spawn(move || {
                // all threads want to work on npu 0
                let n_numbers = 1e4 as usize;
                let mut rng = StdRng::seed_from_u64(12345);
                let us = to_bf16s(
                    &(0..n_numbers)
                        .map(|_| rng.random_range(-1.0f32..1.0))
                        .collect::<Vec<_>>(),
                );
                let vs = to_bf16s(
                    &(0..n_numbers)
                        .map(|_| rng.random_range(-1.0f32..1.0))
                        .collect::<Vec<_>>(),
                );
                let tol = 0.01;

                let us_tensor = QantTensor1::new(&us, [n_numbers]).unwrap();
                let vs_tensor = QantTensor1::new(&vs, [n_numbers]).unwrap();
                let res = mul_elementwise_bf16(0, &us_tensor, &vs_tensor).unwrap();
                let expected: Vec<_> = us.iter().zip(&vs).map(|(&u, &v)| u * v).collect();
                assert_almost_equal_vec(res.data(), &expected, tol);
            });

            handles.push(handle);
        }

        // Wait for all threads to finish
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn driver_info() {
        let info = get_driver_info(0).unwrap();
        println!("{info}");
        assert!(info.contains("commit"));
    }

    #[test]
    fn device_info() {
        let dev_info = get_device_info(0).unwrap();
        println!("{dev_info:?}");
        assert!(dev_info.sensor.voltage_12v >= 10.)
    }

    #[test]
    fn available_npus() {
        let available_npus = get_available_npus().unwrap();
        println!("{available_npus:?}");
        assert!(available_npus.keys().len() > 0)
    }

    #[test]
    fn perf_counters() {
        reset_perf_counter(0).unwrap();
        get_perf_counter(0).unwrap();
    }
}
