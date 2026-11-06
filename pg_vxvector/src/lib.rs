use pgrx::pg_sys::ffi;
use pgrx::prelude::*;
use pgrx::JsonB;
use std::ffi::{CStr, CString};

pgrx::pg_module_magic!();

#[pgrx::pg_guard]
unsafe extern "C-unwind" {
    fn vxvector_pg_init();
    fn vxvector_pg_run(
        table: pg_sys::Oid,
        column: *const std::ffi::c_char,
        lists: *const i32,
        list_count: i32,
        sampling: i32,
        batch: i32,
        iterations: i32,
        restarts: i32,
        metric: *const std::ffi::c_char,
        skip_index: bool,
        spherical: bool,
        residual: bool,
        seed: i32,
        budget: i64,
        centroids: *const std::ffi::c_char,
        index: *const std::ffi::c_char,
        timeout_ms: i32,
    ) -> *const std::ffi::c_char;
}

#[pg_guard]
pub unsafe extern "C-unwind" fn _PG_init() {
    vxvector_pg_init();
}

fn cstring(value: &str) -> CString {
    CString::new(value)
        .unwrap_or_else(|_| pgrx::error!("Text arguments cannot contain a zero byte"))
}

#[pg_extern]
fn create_vector_index(
    source_table: pg_sys::Oid,
    column_name: &str,
    lists: Vec<Option<i32>>,
    sampling_factor: i32,
    batch_size: i32,
    kmeans_iterations: i32,
    kmeans_nredo: i32,
    distance_operator: &str,
    skip_index_build: bool,
    spherical_centroids: bool,
    residual_quantization: bool,
    seed: i32,
    memory_budget_bytes: i64,
    centroid_table: Option<&str>,
    index_name: Option<&str>,
    worker_timeout_ms: i32,
) -> JsonB {
    let lists: Vec<i32> = lists
        .into_iter()
        .map(|v| v.unwrap_or_else(|| pgrx::error!("lists cannot contain NULL")))
        .collect();
    let column = cstring(column_name);
    let metric = cstring(distance_operator);
    let centroids = centroid_table.map(cstring);
    let index = index_name.map(cstring);
    let result = unsafe {
        vxvector_pg_run(
            source_table,
            column.as_ptr(),
            lists.as_ptr(),
            lists.len() as i32,
            sampling_factor,
            batch_size,
            kmeans_iterations,
            kmeans_nredo,
            metric.as_ptr(),
            skip_index_build,
            spherical_centroids,
            residual_quantization,
            seed,
            memory_budget_bytes,
            centroids.as_ref().map_or(std::ptr::null(), |v| v.as_ptr()),
            index.as_ref().map_or(std::ptr::null(), |v| v.as_ptr()),
            worker_timeout_ms,
        )
    };
    let text = unsafe { CStr::from_ptr(result) }
        .to_str()
        .expect("UTF-8 worker result");
    JsonB(serde_json::from_str(text).expect("valid result JSON"))
}
