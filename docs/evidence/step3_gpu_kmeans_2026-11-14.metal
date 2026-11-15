// Written by vxc for the Apple GPU.
#include <metal_stdlib>
using namespace metal;

// vx.kernel vxvector_gpu_accumulate_v1 m2 m1 m2 m1r m2r i32 i32 i32 i32
kernel void vxvector_gpu_accumulate_v1(
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
  float v2 = 0.0f;
  int v3 = (int)256u;
  int v4 = (int)1u;
  int v5 = (int)vx_tgid.x;
  int v6 = (int)v5;
  int v7 = (int)vx_ntg.x;
  int v8 = (int)v7;
  {
    for (int v9 = v6; v9 < p7; v9 = (int)((uint)v9 + (uint)v8)) {
      int v10 = (int)vx_tid.x;
      int v11 = (int)v10;
      {
        for (int v12 = v11; v12 < p8; v12 = (int)((uint)v12 + (uint)v3)) {
          int v13 = (int)v9;
          int v14 = p0_n1;
          int v15 = (int)((uint)v13 * (uint)v14);
          const int o16 = p0_b + v15;
          int v17 = (int)v12;
          float v18 = p0[o16 + v17 * 1];
          int v19 = (int)((uint)p5 + (uint)p6);
          float v20;
          int v21;
          int v22;
          {
            float c23 = v18;
            int c24 = v1;
            int c25 = v0;
            for (int v26 = p5; v26 < v19; v26 = (int)((uint)v26 + (uint)v0)) {
              float v27 = c23;
              int v28 = c24;
              int v29 = c25;
              int v30 = (int)v26;
              int v31 = p3[p3_o + v30 * p3_s0];
              bool v32 = v31 == v9;
              float v33;
              int v34;
              int v35;
              if (v32) {
                int v36 = p4_n1;
                int v37 = (int)((uint)v30 * (uint)v36);
                const int o38 = p4_b + v37;
                float v39 = p4[o38 + v17 * 1];
                float v40 = v27 + v39;
                int v41 = (int)((uint)v28 + (uint)v0);
                float v42 = v40 - v40;
                bool v43 = v42 == v2;
                int v44 = v43 ? v29 : v1;
                v33 = v40;
                v34 = v41;
                v35 = v44;
              } else {
                v33 = v27;
                v34 = v28;
                v35 = v29;
              }
              c23 = v33;
              c24 = v34;
              c25 = v35;
            }
            v20 = c23;
            v21 = c24;
            v22 = c25;
          }
          p0[o16 + v17 * 1] = v20;
          bool v45 = v12 == v1;
          if (v45) {
            int v46 = p1[p1_o + v13 * p1_s0];
            int v47 = (int)((uint)v46 + (uint)v21);
            p1[p1_o + v13 * p1_s0] = v47;
          }
          bool v48 = v22 == v1;
          if (v48) {
            int v49 = p2_n1;
            int v50 = (int)((uint)v13 * (uint)v49);
            const int o51 = p2_b + v50;
            p2[o51 + v17 * 1] = v0;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
  }
  return;
}

// vx.kernel vxvector_gpu_assign_v2 m1 m1 m2r m2r i32 i32 i32 i32 i32
kernel void vxvector_gpu_assign_v2(
    device int *p0 [[buffer(0)]],
    device float *p1 [[buffer(1)]],
    device const float *p2 [[buffer(2)]],
    device const float *p3 [[buffer(3)]],
    constant uint *vx_w [[buffer(4)]],
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
  const int p1_s0 = 1;
  const int p2_b = as_type<int>(vx_w[8]);
  const int p2_o = p2_b + 0;
  const int p2_n0 = as_type<int>(vx_w[10]);
  const int p2_s0 = as_type<int>(vx_w[12]);
  const int p2_n1 = as_type<int>(vx_w[11]);
  const int p2_s1 = 1;
  const int p3_b = as_type<int>(vx_w[14]);
  const int p3_o = p3_b + 0;
  const int p3_n0 = as_type<int>(vx_w[16]);
  const int p3_s0 = as_type<int>(vx_w[18]);
  const int p3_n1 = as_type<int>(vx_w[17]);
  const int p3_s1 = 1;
  const int p4 = as_type<int>(vx_w[20]);
  const int p5 = as_type<int>(vx_w[21]);
  const int p6 = as_type<int>(vx_w[22]);
  const int p7 = as_type<int>(vx_w[23]);
  const int p8 = as_type<int>(vx_w[24]);
  int v0 = (int)1u;
  int v1 = (int)0u;
  int v2 = (int)255u;
  int v3 = (int)256u;
  float v4 = 0.0f;
  int v5 = (int)8u;
  int v6 = (int)1u;
  int v7 = (int)2u;
  int v8 = (int)3u;
  int v9 = (int)4u;
  int v10 = (int)5u;
  int v11 = (int)6u;
  int v12 = (int)7u;
  bool v13 = true;
  int v14 = (int)vx_tgid.x;
  int v15 = (int)v14;
  int v16 = (int)vx_ntg.x;
  int v17 = (int)v16;
  int v18 = (int)((uint)p5 + (uint)v2);
  int v19 = v18 / v3;
  {
    for (int v20 = v15; v20 < v19; v20 = (int)((uint)v20 + (uint)v17)) {
      int v21 = (int)vx_tid.x;
      int v22 = (int)v21;
      {
        for (int v23 = v22; v23 < v3; v23 = (int)((uint)v23 + (uint)v3)) {
          int v24 = (int)((uint)v20 * (uint)v3);
          int v25 = (int)((uint)p4 + (uint)v24);
          int v26 = (int)((uint)v25 + (uint)v23);
          int v27 = (int)((uint)p4 + (uint)p5);
          bool v28 = v26 < v27;
          if (v28) {
            bool v29 = p6 > v1;
            int v30;
            float v31;
            if (v29) {
              int v32 = (int)v26;
              int v33 = p0[p0_o + v32 * p0_s0];
              float v34 = p1[p1_o + v32 * p1_s0];
              v30 = v33;
              v31 = v34;
            } else {
              v30 = v1;
              v31 = v4;
            }
            int v35 = p7 / v5;
            int v36;
            float v37;
            {
              int c38 = v30;
              float c39 = v31;
              for (int v40 = v1; v40 < v35; v40 = (int)((uint)v40 + (uint)v0)) {
                int v41 = c38;
                float v42 = c39;
                int v43 = (int)((uint)v40 * (uint)v5);
                int v44 = (int)((uint)p6 + (uint)v43);
                float v45;
                float v46;
                float v47;
                float v48;
                float v49;
                float v50;
                float v51;
                float v52;
                {
                  float c53 = v4;
                  float c54 = v4;
                  float c55 = v4;
                  float c56 = v4;
                  float c57 = v4;
                  float c58 = v4;
                  float c59 = v4;
                  float c60 = v4;
                  for (int v61 = v1; v61 < p8; v61 = (int)((uint)v61 + (uint)v0)) {
                    float v62 = c53;
                    float v63 = c54;
                    float v64 = c55;
                    float v65 = c56;
                    float v66 = c57;
                    float v67 = c58;
                    float v68 = c59;
                    float v69 = c60;
                    int v70 = (int)v26;
                    int v71 = p2_n1;
                    int v72 = (int)((uint)v70 * (uint)v71);
                    const int o73 = p2_b + v72;
                    int v74 = (int)v61;
                    float v75 = p2[o73 + v74 * 1];
                    int v76 = (int)v44;
                    int v77 = p3_n1;
                    int v78 = (int)((uint)v76 * (uint)v77);
                    const int o79 = p3_b + v78;
                    float v80 = p3[o79 + v74 * 1];
                    float v81 = v75 - v80;
                    int v82 = (int)((uint)v44 + (uint)v0);
                    int v83 = (int)v82;
                    int v84 = (int)((uint)v83 * (uint)v77);
                    const int o85 = p3_b + v84;
                    float v86 = p3[o85 + v74 * 1];
                    float v87 = v75 - v86;
                    int v88 = (int)((uint)v44 + (uint)v7);
                    int v89 = (int)v88;
                    int v90 = (int)((uint)v89 * (uint)v77);
                    const int o91 = p3_b + v90;
                    float v92 = p3[o91 + v74 * 1];
                    float v93 = v75 - v92;
                    int v94 = (int)((uint)v44 + (uint)v8);
                    int v95 = (int)v94;
                    int v96 = (int)((uint)v95 * (uint)v77);
                    const int o97 = p3_b + v96;
                    float v98 = p3[o97 + v74 * 1];
                    float v99 = v75 - v98;
                    int v100 = (int)((uint)v44 + (uint)v9);
                    int v101 = (int)v100;
                    int v102 = (int)((uint)v101 * (uint)v77);
                    const int o103 = p3_b + v102;
                    float v104 = p3[o103 + v74 * 1];
                    float v105 = v75 - v104;
                    int v106 = (int)((uint)v44 + (uint)v10);
                    int v107 = (int)v106;
                    int v108 = (int)((uint)v107 * (uint)v77);
                    const int o109 = p3_b + v108;
                    float v110 = p3[o109 + v74 * 1];
                    float v111 = v75 - v110;
                    int v112 = (int)((uint)v44 + (uint)v11);
                    int v113 = (int)v112;
                    int v114 = (int)((uint)v113 * (uint)v77);
                    const int o115 = p3_b + v114;
                    float v116 = p3[o115 + v74 * 1];
                    float v117 = v75 - v116;
                    int v118 = (int)((uint)v44 + (uint)v12);
                    int v119 = (int)v118;
                    int v120 = (int)((uint)v119 * (uint)v77);
                    const int o121 = p3_b + v120;
                    float v122 = p3[o121 + v74 * 1];
                    float v123 = v75 - v122;
                    float v124 = v81 * v81;
                    float v125 = v62 + v124;
                    float v126 = v87 * v87;
                    float v127 = v63 + v126;
                    float v128 = v93 * v93;
                    float v129 = v64 + v128;
                    float v130 = v99 * v99;
                    float v131 = v65 + v130;
                    float v132 = v105 * v105;
                    float v133 = v66 + v132;
                    float v134 = v111 * v111;
                    float v135 = v67 + v134;
                    float v136 = v117 * v117;
                    float v137 = v68 + v136;
                    float v138 = v123 * v123;
                    float v139 = v69 + v138;
                    c53 = v125;
                    c54 = v127;
                    c55 = v129;
                    c56 = v131;
                    c57 = v133;
                    c58 = v135;
                    c59 = v137;
                    c60 = v139;
                  }
                  v45 = c53;
                  v46 = c54;
                  v47 = c55;
                  v48 = c56;
                  v49 = c57;
                  v50 = c58;
                  v51 = c59;
                  v52 = c60;
                }
                bool v140 = v44 == v1;
                bool v141;
                if (v140) {
                  v141 = v13;
                } else {
                  bool v142 = v45 < v42;
                  v141 = v142;
                }
                int v143 = v141 ? v44 : v41;
                float v144 = v141 ? v45 : v42;
                bool v145 = v46 < v144;
                float v146 = v145 ? v46 : v144;
                int v147;
                if (v145) {
                  int v148 = (int)((uint)v44 + (uint)v0);
                  v147 = v148;
                } else {
                  v147 = v143;
                }
                bool v149 = v47 < v146;
                float v150 = v149 ? v47 : v146;
                int v151;
                if (v149) {
                  int v152 = (int)((uint)v44 + (uint)v7);
                  v151 = v152;
                } else {
                  v151 = v147;
                }
                bool v153 = v48 < v150;
                float v154 = v153 ? v48 : v150;
                int v155;
                if (v153) {
                  int v156 = (int)((uint)v44 + (uint)v8);
                  v155 = v156;
                } else {
                  v155 = v151;
                }
                bool v157 = v49 < v154;
                float v158 = v157 ? v49 : v154;
                int v159;
                if (v157) {
                  int v160 = (int)((uint)v44 + (uint)v9);
                  v159 = v160;
                } else {
                  v159 = v155;
                }
                bool v161 = v50 < v158;
                float v162 = v161 ? v50 : v158;
                int v163;
                if (v161) {
                  int v164 = (int)((uint)v44 + (uint)v10);
                  v163 = v164;
                } else {
                  v163 = v159;
                }
                bool v165 = v51 < v162;
                float v166 = v165 ? v51 : v162;
                int v167;
                if (v165) {
                  int v168 = (int)((uint)v44 + (uint)v11);
                  v167 = v168;
                } else {
                  v167 = v163;
                }
                bool v169 = v52 < v166;
                float v170 = v169 ? v52 : v166;
                int v171;
                if (v169) {
                  int v172 = (int)((uint)v44 + (uint)v12);
                  v171 = v172;
                } else {
                  v171 = v167;
                }
                c38 = v171;
                c39 = v170;
              }
              v36 = c38;
              v37 = c39;
            }
            int v173 = (int)((uint)v35 * (uint)v5);
            int v174;
            float v175;
            {
              int c176 = v36;
              float c177 = v37;
              for (int v178 = v173; v178 < p7; v178 = (int)((uint)v178 + (uint)v0)) {
                int v179 = c176;
                float v180 = c177;
                int v181 = (int)((uint)p6 + (uint)v178);
                float v182;
                {
                  float c183 = v4;
                  for (int v184 = v1; v184 < p8; v184 = (int)((uint)v184 + (uint)v0)) {
                    float v185 = c183;
                    int v186 = (int)v26;
                    int v187 = p2_n1;
                    int v188 = (int)((uint)v186 * (uint)v187);
                    const int o189 = p2_b + v188;
                    int v190 = (int)v184;
                    float v191 = p2[o189 + v190 * 1];
                    int v192 = (int)v181;
                    int v193 = p3_n1;
                    int v194 = (int)((uint)v192 * (uint)v193);
                    const int o195 = p3_b + v194;
                    float v196 = p3[o195 + v190 * 1];
                    float v197 = v191 - v196;
                    float v198 = v197 * v197;
                    float v199 = v185 + v198;
                    c183 = v199;
                  }
                  v182 = c183;
                }
                bool v200 = v181 == v1;
                bool v201;
                if (v200) {
                  v201 = v13;
                } else {
                  bool v202 = v182 < v180;
                  v201 = v202;
                }
                int v203 = v201 ? v181 : v179;
                float v204 = v201 ? v182 : v180;
                c176 = v203;
                c177 = v204;
              }
              v174 = c176;
              v175 = c177;
            }
            int v205 = (int)v26;
            p0[p0_o + v205 * p0_s0] = v174;
            p1[p1_o + v205 * p1_s0] = v175;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
  }
  return;
}

// vx.kernel vxvector_gpu_finish_update_v1 m2 m2r m1r m2r i32 i32 i32 i32
kernel void vxvector_gpu_finish_update_v1(
    device float *p0 [[buffer(0)]],
    device const float *p1 [[buffer(1)]],
    device const int *p2 [[buffer(2)]],
    device const float *p3 [[buffer(3)]],
    constant uint *vx_w [[buffer(4)]],
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
  const int p2_b = as_type<int>(vx_w[12]);
  const int p2_o = p2_b + 0;
  const int p2_n0 = as_type<int>(vx_w[14]);
  const int p2_s0 = 1;
  const int p3_b = as_type<int>(vx_w[16]);
  const int p3_o = p3_b + 0;
  const int p3_n0 = as_type<int>(vx_w[18]);
  const int p3_s0 = as_type<int>(vx_w[20]);
  const int p3_n1 = as_type<int>(vx_w[19]);
  const int p3_s1 = 1;
  const int p4 = as_type<int>(vx_w[22]);
  const int p5 = as_type<int>(vx_w[23]);
  const int p6 = as_type<int>(vx_w[24]);
  const int p7 = as_type<int>(vx_w[25]);
  int v0 = (int)0u;
  int v1 = (int)256u;
  int v2 = (int)1u;
  int v3 = (int)vx_tgid.x;
  int v4 = (int)v3;
  int v5 = (int)vx_ntg.x;
  int v6 = (int)v5;
  {
    for (int v7 = v4; v7 < p5; v7 = (int)((uint)v7 + (uint)v6)) {
      int v8 = (int)vx_tid.x;
      int v9 = (int)v8;
      {
        for (int v10 = v9; v10 < p6; v10 = (int)((uint)v10 + (uint)v1)) {
          int v11 = (int)v7;
          int v12 = p2[p2_o + v11 * p2_s0];
          bool v13 = v12 == v0;
          if (v13) {
            int v14 = p0_n1;
            int v15 = (int)((uint)v11 * (uint)v14);
            const int o16 = p0_b + v15;
            int v17 = (int)v10;
            int v18 = (int)((uint)v7 + (uint)p7);
            int v19 = v18 % p4;
            int v20 = (int)v19;
            int v21 = p3_n1;
            int v22 = (int)((uint)v20 * (uint)v21);
            const int o23 = p3_b + v22;
            float v24 = p3[o23 + v17 * 1];
            p0[o16 + v17 * 1] = v24;
          } else {
            int v25 = p0_n1;
            int v26 = (int)((uint)v11 * (uint)v25);
            const int o27 = p0_b + v26;
            int v28 = (int)v10;
            int v29 = p1_n1;
            int v30 = (int)((uint)v11 * (uint)v29);
            const int o31 = p1_b + v30;
            float v32 = p1[o31 + v28 * 1];
            float v33 = (float)v12;
            float v34 = v32 / v33;
            p0[o27 + v28 * 1] = v34;
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

// vx.kernel vxvector_gpu_zero_v1 m2 m1 i32 i32
kernel void vxvector_gpu_zero_v1(
    device float *p0 [[buffer(0)]],
    device int *p1 [[buffer(1)]],
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
  const int p1_s0 = 1;
  const int p2 = as_type<int>(vx_w[10]);
  const int p3 = as_type<int>(vx_w[11]);
  int v0 = (int)0u;
  int v1 = (int)256u;
  int v2 = (int)1u;
  float v3 = 0.0f;
  int v4 = (int)vx_tgid.x;
  int v5 = (int)v4;
  int v6 = (int)vx_ntg.x;
  int v7 = (int)v6;
  {
    for (int v8 = v5; v8 < p2; v8 = (int)((uint)v8 + (uint)v7)) {
      int v9 = (int)vx_tid.x;
      int v10 = (int)v9;
      {
        for (int v11 = v10; v11 < p3; v11 = (int)((uint)v11 + (uint)v1)) {
          int v12 = (int)v8;
          int v13 = p0_n1;
          int v14 = (int)((uint)v12 * (uint)v13);
          const int o15 = p0_b + v14;
          int v16 = (int)v11;
          p0[o15 + v16 * 1] = v3;
          bool v17 = v11 == v0;
          if (v17) {
            p1[p1_o + v12 * p1_s0] = v0;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
  }
  return;
}
