//===- HLIndexOps.td - HL Indexing operation definitions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR high level indexing operations.
//
//===----------------------------------------------------------------------===//

#ifndef HLINDEX_OPS
#define HLINDEX_OPS

include "mlir/Dialect/HLIndex/IR/HLIndexAttributes.td"
include "mlir/Dialect/HLIndex/IR/HLIndexBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for HLIndex dialect ops.
class HLIndex_Op<string mnemonic, list<Trait> traits = []> :
    Op<HLIndex_Dialect, mnemonic, traits>;

def HLIndexApplyOp : HLIndex_Op<"apply", [Pure]> {
  let summary = "index map apply operation";
  let description = [{
    The `hl_index.apply` operation applies an index map to a list of SSA values,
    yielding a single SSA value. The number of arguements to `hl_index.apply`
    must be equal to the number of inputs to the index mapping; The input
    operands and result must all have ‘index’ type.

    Example:

    ```mlir
    #map10 = index_map<(v0, v1) -> (v0 floordiv 8 + v1 floordiv 128)>
    ...
    %1 = hl_index.apply #map10 (%s, %t)

    // Inline example.
    %2 = hl_index.apply index_map<(d0) -> (-d0)> (%42)
    ```
  }];
  let arguments = (ins IndexMapAttr:$map, Variadic<Index>:$mapOperands);
  let results = (outs Index);

  let builders = [
    OpBuilder<(ins "AffineMap":$map, "ValueRange":$mapOperands),
    [{
      build($_builder, $_state, $_builder.getIndexType(), IndexMapAttr::get(map), mapOperands);
    }]>,
    OpBuilder<(ins "ArrayRef<AffineExpr> ":$exprList,"ValueRange":$mapOperands),
    [{
      build($_builder, $_state, $_builder.getIndexType(),
            IndexMapAttr::get(AffineMap::inferFromExprList(exprList).front()),
            mapOperands);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Returns the affine map to be applied by this operation.
    AffineMap getIndexMap() { return getMap().getAffineMap(); }
    /// Returns the affine expr result.
    AffineExpr getIndexExpr() { return getIndexMap().getResult(0); }
  }];

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

#endif // HLINDEX_OPS
