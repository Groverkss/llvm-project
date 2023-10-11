//===- HLIndexDialect.td - HL Indexing dialect definition --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HLINDEX_DIALECT
#define HLINDEX_DIALECT

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// HLIndexDialect
//===----------------------------------------------------------------------===//

def HLIndex_Dialect : Dialect {
  let name = "hl_index";

  let summary = "The High Level Indexing dialect";
  let description = [{
  }];

  let cppNamespace = "::mlir::hl_index";

  let extraClassDeclaration = [{
    /// Register all dialect attributes.
    void registerAttributes();
    /// Register all dialect operations.
    void registerOperations();
  }];

  let hasConstantMaterializer = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // INDEX_DIALECT
