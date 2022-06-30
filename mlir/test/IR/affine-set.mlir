// RUN: mlir-opt -allow-unregistered-dialect %s | FileCheck %s

// Check if affine constraints with affine exprs on RHS can be parsed.

// CHECK-DAG: #set{{[0-9]+}} = affine_set<(d0) : (d0 - 1 == 0)>
#set0 = affine_set<(i) : (i == 1)>

// CHECK-DAG: "testset0"() {set = #set{{[0-9]+}}} : () -> ()
"testset0"() {set = #set0} : () -> ()

// ---

// CHECK-DAG: #set{{[0-9]+}} = affine_set<(d0)[s0, s1] : (d0 >= 0, -d0 + s0 >= 0, s0 - 5 == 0, -d0 + s1 + 1 >= 0)>
#set1 = affine_set<(i)[N, M] : (i >= 0, N >= i, N == 5, M + 1 >= i)>

// CHECK-DAG: "testset1"() {set = #set{{[0-9]+}}} : () -> ()
"testset1"() {set = #set1} : () -> ()

// ---

// CHECK-DAG: #set{{[0-9]+}} = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + s0 >= 0, d1 >= 0, d0 - d1 >= 0)>
#set2 = affine_set<(i, j)[N] : (i >= 0, N >= i, j >= 0, i >= j)>

// CHECK-DAG: "testset2"() {set = #set{{[0-9]+}}} : () -> ()
"testset2"() {set = #set2} : () -> ()

// ---

// CHECK-DAG: #set{{[0-9]+}} = affine_set<(d0, d1)[s0, s1] : (-(d0 + d1 + s0 + s1) == 0, d0 + d1 - (s0 + s1) == 0)>
#set3 = affine_set<(i0, i1)[N, M] : (0 == i0 + i1 + N + M, i0 + i1 == N + M)>

// CHECK-DAG: "testset3"() {set = #set{{[0-9]+}}} : () -> ()
"testset3"() {set = #set3} : () -> ()

// ---

// CHECK-DAG: #set{{[0-9]+}} = affine_set<(d0, d1)[s0, s1] : (-(d0 + d1 + s0 + s1) >= 0, d0 + d1 - (s0 + s1) >= 0)>
#set4 = affine_set<(i0, i1)[N, M] : (0 >= i0 + i1 + N + M, i0 + i1 >= N + M)>

// CHECK-DAG: "testset4"() {set = #set{{[0-9]+}}} : () -> ()
"testset4"() {set = #set4} : () -> ()

// ---

// CHECK-DAG: #set{{[0-9]+}} = affine_set<(d0, d1, d2, d3) : ((d0 + d1) mod 2 - (d2 + d3) floordiv 2 == 0, d0 mod 2 + d1 mod 2 - (d2 + d3 + d2) >= 0)>
#set5 = affine_set<(d0, d1, r0, r1) : ((d0 + d1) mod 2 == (r0 + r1) floordiv 2, ((d0) mod 2) + ((d1) mod 2) >= (r0 + r1) + r0)>

// CHECK-DAG: "testset5"() {set = #set{{[0-9]+}}} : () -> ()
"testset5"() {set = #set5} : () -> ()

// ---

// Check if affine constraints with `<=` can be parsed.

// CHECK-DAG: #set{{[0-9]+}} = affine_set<(d0)[s0] : (d0 >= 0, -d0 + s0 >= 0)>
#set6 = affine_set<(i)[N] : (i >= 0, i <= N)>

// CHECK-DAG: "testset6"() {set = #set{{[0-9]+}}} : () -> ()
"testset6"() {set = #set6} : () -> ()

// ---

// CHECK-DAG: #set{{[0-9]+}} = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + s0 >= 0, d1 - d0 >= 0, -d1 + s0 >= 0)>
#set7 = affine_set<(i, j)[N] : (i >= 0, i <= N, j >= i, j <= N)>

// CHECK-DAG: "testset7"() {set = #set{{[0-9]+}}} : () -> ()
"testset7"() {set = #set7} : () -> ()

// ---

// CHECK-DAG: #set{{[0-9]+}} = affine_set<(d0, d1, d2, d3)[s0, s1] : (d0 >= 0, -d0 + s0 floordiv 64 - 1 >= 0, d2 >= 0, -d2 + s1 floordiv 64 - 1 >= 0, d1 >= 0, -d1 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#set8 = affine_set<(i0, i1, j0, j1)[N, M] : (i0 >= 0, i0 <= (N floordiv 64) - 1, j0 >= 0, j0 <= (M floordiv 64) - 1, i1 >= 0, i1 <= 63, j1 >= 0, j1 <= 63)>

// CHECK-DAG: "testset8"() {set = #set{{[0-9]+}}} : () -> ()
"testset8"() {set = #set8} : () -> ()

// ---
