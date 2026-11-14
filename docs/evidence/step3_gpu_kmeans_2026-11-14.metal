// Written by vxc for the Apple GPU.
#include <metal_stdlib>
using namespace metal;

// vx.kernel vxvector_gpu_assign_v1 m1 m2r m2r i32 i32 i32
kernel void vxvector_gpu_assign_v1(
    device int *p0 [[buffer(0)]],
    device const float *p1 [[buffer(1)]],
    device const float *p2 [[buffer(2)]],
    constant uint *vx_w [[buffer(3)]],
    uint3 vx_tgid [[threadgroup_position_in_grid]],
    uint3 vx_tid [[thread_position_in_threadgroup]],
    uint3 vx_ntg [[threadgroups_per_grid]],
    uint3 vx_tptg [[threads_per_threadgroup]],
    uint vx_lane [[thread_index_in_simdgroup]],
    uint vx_simd [[threads_per_simdgroup]]) {
  const int p0_b = as_type<int>(vx_w[0]);
  const int p0_o = p0_b + 0;
  const int p0_n0 = as_type<int>(vx_w[2]);
  const int p0_s0 = 1;
  const int p1_b = as_type<int>(vx_w[4]);
  const int p1_o = p1_b + 0;
  const int p1_n0 = as_type<int>(vx_w[6]);
  const int p1_s0 = as_type<int>(vx_w[8]);
  const int p1_n1 = as_type<int>(vx_w[7]);
  const int p1_s1 = 1;
  const int p2_b = as_type<int>(vx_w[10]);
  const int p2_o = p2_b + 0;
  const int p2_n0 = as_type<int>(vx_w[12]);
  const int p2_s0 = as_type<int>(vx_w[14]);
  const int p2_n1 = as_type<int>(vx_w[13]);
  const int p2_s1 = 1;
  const int p3 = as_type<int>(vx_w[16]);
  const int p4 = as_type<int>(vx_w[17]);
  const int p5 = as_type<int>(vx_w[18]);
  int v0 = (int)1u;
  int v1 = (int)0u;
  int v2 = (int)255u;
  int v3 = (int)256u;
  float v4 = 0.0f;
  int v5 = (int)1u;
  bool v6 = true;
  int v7 = (int)vx_tgid.x;
  int v8 = (int)v7;
  int v9 = (int)vx_ntg.x;
  int v10 = (int)v9;
  int v11 = (int)((uint)p3 + (uint)v2);
  int v12 = v11 / v3;
  {
    for (int v13 = v8; v13 < v12; v13 = (int)((uint)v13 + (uint)v10)) {
      int v14 = (int)vx_tid.x;
      int v15 = (int)v14;
      {
        for (int v16 = v15; v16 < v3; v16 = (int)((uint)v16 + (uint)v3)) {
          int v17 = (int)((uint)v13 * (uint)v3);
          int v18 = (int)((uint)v17 + (uint)v16);
          bool v19 = v18 < p3;
          if (v19) {
            int v20;
            float v21;
            {
              int c22 = v1;
              float c23 = v4;
              for (int v24 = v1; v24 < p4; v24 = (int)((uint)v24 + (uint)v0)) {
                int v25 = c22;
                float v26 = c23;
                float v27;
                {
                  float c28 = v4;
                  for (int v29 = v1; v29 < p5; v29 = (int)((uint)v29 + (uint)v0)) {
                    float v30 = c28;
                    int v31 = (int)v18;
                    int v32 = p1_n1;
                    int v33 = (int)((uint)v31 * (uint)v32);
                    const int o34 = p1_b + v33;
                    int v35 = (int)v29;
                    float v36 = p1[o34 + v35 * 1];
                    int v37 = (int)v24;
                    int v38 = p2_n1;
                    int v39 = (int)((uint)v37 * (uint)v38);
                    const int o40 = p2_b + v39;
                    float v41 = p2[o40 + v35 * 1];
                    float v42 = v36 - v41;
                    float v43 = v42 * v42;
                    float v44 = v30 + v43;
                    c28 = v44;
                  }
                  v27 = c28;
                }
                bool v45 = v24 == v1;
                bool v46;
                if (v45) {
                  v46 = v6;
                } else {
                  bool v47 = v27 < v26;
                  v46 = v47;
                }
                int v48 = v46 ? v24 : v25;
                float v49 = v46 ? v27 : v26;
                c22 = v48;
                c23 = v49;
              }
              v20 = c22;
              v21 = c23;
            }
            int v50 = (int)v18;
            p0[p0_o + v50 * p0_s0] = v20;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
  }
  return;
}

// vx.kernel vxvector_gpu_seed_v1 m2 m2r i32 i32 i32
kernel void vxvector_gpu_seed_v1(
    device float *p0 [[buffer(0)]],
    device const float *p1 [[buffer(1)]],
    constant uint *vx_w [[buffer(2)]],
    uint3 vx_tgid [[threadgroup_position_in_grid]],
    uint3 vx_tid [[thread_position_in_threadgroup]],
    uint3 vx_ntg [[threadgroups_per_grid]],
    uint3 vx_tptg [[threads_per_threadgroup]],
    uint vx_lane [[thread_index_in_simdgroup]],
    uint vx_simd [[threads_per_simdgroup]]) {
  const int p0_b = as_type<int>(vx_w[0]);
  const int p0_o = p0_b + 0;
  const int p0_n0 = as_type<int>(vx_w[2]);
  const int p0_s0 = as_type<int>(vx_w[4]);
  const int p0_n1 = as_type<int>(vx_w[3]);
  const int p0_s1 = 1;
  const int p1_b = as_type<int>(vx_w[6]);
  const int p1_o = p1_b + 0;
  const int p1_n0 = as_type<int>(vx_w[8]);
  const int p1_s0 = as_type<int>(vx_w[10]);
  const int p1_n1 = as_type<int>(vx_w[9]);
  const int p1_s1 = 1;
  const int p2 = as_type<int>(vx_w[12]);
  const int p3 = as_type<int>(vx_w[13]);
  const int p4 = as_type<int>(vx_w[14]);
  int v0 = (int)256u;
  int v1 = (int)1u;
  int v2 = (int)vx_tgid.x;
  int v3 = (int)v2;
  int v4 = (int)vx_ntg.x;
  int v5 = (int)v4;
  {
    for (int v6 = v3; v6 < p3; v6 = (int)((uint)v6 + (uint)v5)) {
      int v7 = (int)vx_tid.x;
      int v8 = (int)v7;
      {
        for (int v9 = v8; v9 < p4; v9 = (int)((uint)v9 + (uint)v0)) {
          long v10 = (long)v6;
          long v11 = (long)p2;
          long v12 = (long)((ulong)v10 * (ulong)v11);
          long v13 = (long)p3;
          long v14 = v12 / v13;
          int v15 = (int)v14;
          int v16 = (int)v6;
          int v17 = p0_n1;
          int v18 = (int)((uint)v16 * (uint)v17);
          const int o19 = p0_b + v18;
          int v20 = (int)v9;
          int v21 = (int)v15;
          int v22 = p1_n1;
          int v23 = (int)((uint)v21 * (uint)v22);
          const int o24 = p1_b + v23;
          float v25 = p1[o24 + v20 * 1];
          p0[o19 + v20 * 1] = v25;
        }
      }
      threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
  }
  return;
}

// vx.kernel vxvector_gpu_update_v1 m2 m1 m2 m1r m2r i32 i32 i32 i32
kernel void vxvector_gpu_update_v1(
    device float *p0 [[buffer(0)]],
    device int *p1 [[buffer(1)]],
    device int *p2 [[buffer(2)]],
    device const int *p3 [[buffer(3)]],
    device const float *p4 [[buffer(4)]],
    constant uint *vx_w [[buffer(5)]],
    uint3 vx_tgid [[threadgroup_position_in_grid]],
    uint3 vx_tid [[thread_position_in_threadgroup]],
    uint3 vx_ntg [[threadgroups_per_grid]],
    uint3 vx_tptg [[threads_per_threadgroup]],
    uint vx_lane [[thread_index_in_simdgroup]],
    uint vx_simd [[threads_per_simdgroup]]) {
  const int p0_b = as_type<int>(vx_w[0]);
  const int p0_o = p0_b + 0;
  const int p0_n0 = as_type<int>(vx_w[2]);
  const int p0_s0 = as_type<int>(vx_w[4]);
  const int p0_n1 = as_type<int>(vx_w[3]);
  const int p0_s1 = 1;
  const int p1_b = as_type<int>(vx_w[6]);
  const int p1_o = p1_b + 0;
  const int p1_n0 = as_type<int>(vx_w[8]);
  const int p1_s0 = 1;
  const int p2_b = as_type<int>(vx_w[10]);
  const int p2_o = p2_b + 0;
  const int p2_n0 = as_type<int>(vx_w[12]);
  const int p2_s0 = as_type<int>(vx_w[14]);
  const int p2_n1 = as_type<int>(vx_w[13]);
  const int p2_s1 = 1;
  const int p3_b = as_type<int>(vx_w[16]);
  const int p3_o = p3_b + 0;
  const int p3_n0 = as_type<int>(vx_w[18]);
  const int p3_s0 = 1;
  const int p4_b = as_type<int>(vx_w[20]);
  const int p4_o = p4_b + 0;
  const int p4_n0 = as_type<int>(vx_w[22]);
  const int p4_s0 = as_type<int>(vx_w[24]);
  const int p4_n1 = as_type<int>(vx_w[23]);
  const int p4_s1 = 1;
  const int p5 = as_type<int>(vx_w[26]);
  const int p6 = as_type<int>(vx_w[27]);
  const int p7 = as_type<int>(vx_w[28]);
  const int p8 = as_type<int>(vx_w[29]);
  int v0 = (int)1u;
  int v1 = (int)0u;
  int v2 = (int)256u;
  float v3 = 0.0f;
  int v4 = (int)1u;
  int v5 = (int)vx_tgid.x;
  int v6 = (int)v5;
  int v7 = (int)vx_ntg.x;
  int v8 = (int)v7;
  {
    for (int v9 = v6; v9 < p6; v9 = (int)((uint)v9 + (uint)v8)) {
      int v10 = (int)vx_tid.x;
      int v11 = (int)v10;
      {
        for (int v12 = v11; v12 < p7; v12 = (int)((uint)v12 + (uint)v2)) {
          float v13;
          int v14;
          int v15;
          {
            float c16 = v3;
            int c17 = v1;
            int c18 = v0;
            for (int v19 = v1; v19 < p5; v19 = (int)((uint)v19 + (uint)v0)) {
              float v20 = c16;
              int v21 = c17;
              int v22 = c18;
              int v23 = (int)v19;
              int v24 = p3[p3_o + v23 * p3_s0];
              bool v25 = v24 == v9;
              float v26;
              int v27;
              int v28;
              if (v25) {
                int v29 = p4_n1;
                int v30 = (int)((uint)v23 * (uint)v29);
                const int o31 = p4_b + v30;
                int v32 = (int)v12;
                float v33 = p4[o31 + v32 * 1];
                float v34 = v20 + v33;
                int v35 = (int)((uint)v21 + (uint)v0);
                float v36 = v34 - v34;
                bool v37 = v36 == v3;
                int v38 = v37 ? v22 : v1;
                v26 = v34;
                v27 = v35;
                v28 = v38;
              } else {
                v26 = v20;
                v27 = v21;
                v28 = v22;
              }
              c16 = v26;
              c17 = v27;
              c18 = v28;
            }
            v13 = c16;
            v14 = c17;
            v15 = c18;
          }
          bool v39 = v14 == v1;
          if (v39) {
            int v40 = (int)v9;
            int v41 = p0_n1;
            int v42 = (int)((uint)v40 * (uint)v41);
            const int o43 = p0_b + v42;
            int v44 = (int)v12;
            int v45 = (int)((uint)v9 + (uint)p8);
            int v46 = v45 % p5;
            int v47 = (int)v46;
            int v48 = p4_n1;
            int v49 = (int)((uint)v47 * (uint)v48);
            const int o50 = p4_b + v49;
            float v51 = p4[o50 + v44 * 1];
            p0[o43 + v44 * 1] = v51;
          } else {
            int v52 = (int)v9;
            int v53 = p0_n1;
            int v54 = (int)((uint)v52 * (uint)v53);
            const int o55 = p0_b + v54;
            int v56 = (int)v12;
            float v57 = (float)v14;
            float v58 = v13 / v57;
            p0[o55 + v56 * 1] = v58;
          }
          bool v59 = v12 == v1;
          if (v59) {
            int v60 = (int)v9;
            p1[p1_o + v60 * p1_s0] = v14;
          }
          bool v61 = v15 == v1;
          if (v61) {
            int v62 = (int)v9;
            int v63 = p2_n1;
            int v64 = (int)((uint)v62 * (uint)v63);
            const int o65 = p2_b + v64;
            int v66 = (int)v12;
            p2[o65 + v66 * 1] = v0;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
  }
  return;
}
