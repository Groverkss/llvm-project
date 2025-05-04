//===- HLIndexOps.h - HL Indexing operation definitions -------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_HLINDEX_IR_HLINDEXOPS_H
#define MLIR_DIALECT_HLINDEX_IR_HLINDEXOPS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"

#include "mlir/Dialect/HLIndex/IR/HLIndexOpsDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/HLIndex/IR/HLIndexAttributes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/HLIndex/IR/HLIndexOps.h.inc"

#endif // MLIR_DIALECT_HLINDEX_IR_HLINDEXOPS_H
