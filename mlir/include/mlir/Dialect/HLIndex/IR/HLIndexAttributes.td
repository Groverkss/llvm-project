//===- HLIndexAttributes.td - HL Indexing Attributes -------*- tablegen -*-===//
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

#ifndef HLINDEX_ATTRIBUTES
#define HLINDEX_ATTRIBUTES

include "mlir/Dialect/HLIndex/IR/HLIndexBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

// Base class for HLIndex dialect attributes.
class HLIndex_Attr<string name, string attrMnemonic, list<Trait> traits = []> :
    AttrDef<HLIndex_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def IndexMapAttr : HLIndex_Attr<"IndexMap", "index_map"> {
  let summary = [{A Multi-Indexing Map}];
  let description = [{
    Syntax:

    ```
    index-map-attribute ::= `index_map` `<` affine-map `>`
    ```

    Examples:

    ```mlir
    index_map<(d0) -> (d0)>
    index_map<(d0, d1, d2) -> (d0, d1)>
    ```
  }];

  let parameters = (ins "AffineMap":$value);
  let builders = [
    AttrBuilderWithInferredContext<(ins "AffineMap":$value), [{
      return $_get(value.getContext(), value);
    }]>
  ];

  let extraClassDeclaration = [{
    using ValueType = AffineMap;
    AffineMap getAffineMap() const { return getValue(); }
  }];

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

#endif // HLINDEX_ATTRIBUTES
