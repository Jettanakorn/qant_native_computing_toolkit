//! The device-memory API as an external Rust crate sees it. An integration
//! test can only use public items, so this fails to compile if `QantDataRole`
//! stops being exported.
#![cfg(feature = "cpu-backend")]

use qant_native_computing_toolkit::{
    QantDataRole,
    data_format_conversion::to_bf16s,
    data_structures::QantTensor2,
    memory_management::{QantMemoryHandle, mul_matrix_vec_batched_mem},
};

#[test]
fn preloaded_matrix_vector_from_public_api() {
    // [[1, 2], [3, 4], [5, 6]] · [1, -1] = [-1, -1, -1]
    let m = to_bf16s(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let v = to_bf16s(&[1.0f32, -1.0]);
    let matrix = QantTensor2::new(&m[..], [3, 2]).unwrap();
    let vector = QantTensor2::new(&v[..], [1, 2]).unwrap();

    let hm = QantMemoryHandle::from_matrix(0, &matrix, QantDataRole::Weights).unwrap();
    let hv = QantMemoryHandle::from_matrix(0, &vector, QantDataRole::Features).unwrap();
    let hr = QantMemoryHandle::allocate(0, 1, 3, QantDataRole::Features).unwrap();
    mul_matrix_vec_batched_mem(&hm, &hv, &hr).unwrap();

    let out = hr.download().unwrap();
    assert_eq!(out.shape(), &[1, 3]);
    assert_eq!(out.data(), &to_bf16s(&[-1.0f32, -1.0, -1.0])[..]);
}
