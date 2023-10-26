//===- HLIndexDialect.cpp - HL Index dialect and types---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the HLIndex dialect types and dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/HLIndex/IR/HLIndexDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/HLIndex/IR/HLIndexOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::hl_index;

#define DEBUG_TYPE "hlindex-dialect"

//===----------------------------------------------------------------------===//
// HLIndexDialect
//===----------------------------------------------------------------------===//

void HLIndexDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/HLIndex/IR/HLIndexAttributes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/HLIndex/IR/HLIndexOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// IndexMapAttr
//===----------------------------------------------------------------------===//

Attribute IndexMapAttr::parse(AsmParser &odsParser, Type odsType) {
  AffineMap map;
  if (odsParser.parseLess() || odsParser.parseAffineMap(map) ||
      odsParser.parseGreater())
    return Attribute();
  return IndexMapAttr::get(map);
}

void IndexMapAttr::print(AsmPrinter &odsPrinter) const {
  odsPrinter << "<";
  odsPrinter << getValue();
  odsPrinter << ">";
}

LogicalResult
IndexMapAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                     AffineMap value) {
  // Affine map should only have symbols.
  if (value.getNumDims() != 0)
    return emitError() << "index map should not have any dimensions";
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/HLIndex/IR/HLIndexAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/HLIndex/IR/HLIndexOps.cpp.inc"
