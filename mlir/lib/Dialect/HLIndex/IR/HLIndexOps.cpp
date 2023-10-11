//===- HLIndexOps.cpp - HLIndex Operations --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/HLIndex/IR/HLIndexOps.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::hl_index;

#define DEBUG_TYPE "hlindex-ops"

//===----------------------------------------------------------------------===//
// HLIndexApplyOp
//===----------------------------------------------------------------------===//

/// Parses dimension and symbol list and returns true if parsing failed.
static ParseResult parseDimAndSymbolList(OpAsmParser &parser,
                                         SmallVectorImpl<Value> &operands,
                                         unsigned &numDims) {
  SmallVector<OpAsmParser::UnresolvedOperand, 8> opInfos;
  if (parser.parseOperandList(opInfos, OpAsmParser::Delimiter::Paren))
    return failure();
  // Store number of dimensions for validation by caller.
  numDims = opInfos.size();

  // Parse the optional symbol operands.
  auto indexTy = parser.getBuilder().getIndexType();
  return failure(parser.parseOperandList(
                     opInfos, OpAsmParser::Delimiter::OptionalSquare) ||
                 parser.resolveOperands(opInfos, indexTy, operands));
}

ParseResult HLIndexApplyOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  auto indexTy = builder.getIndexType();

  IndexMapAttr mapAttr;
  unsigned numDims;
  if (parser.parseAttribute(mapAttr, "map", result.attributes) ||
      parseDimAndSymbolList(parser, result.operands, numDims) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  auto map = mapAttr.getValue();

  if (map.getNumDims() != numDims ||
      numDims + map.getNumSymbols() != result.operands.size()) {
    return parser.emitError(parser.getNameLoc(),
                            "dimension or symbol index mismatch");
  }

  result.types.append(map.getNumResults(), indexTy);
  return success();
}

/// Prints dimension and symbol list.
static void printDimAndSymbolList(Operation::operand_iterator begin,
                                  Operation::operand_iterator end,
                                  unsigned numDims, OpAsmPrinter &printer) {
  OperandRange operands(begin, end);
  printer << '(' << operands.take_front(numDims) << ')';
  if (operands.size() > numDims)
    printer << '[' << operands.drop_front(numDims) << ']';
}

void HLIndexApplyOp::print(OpAsmPrinter &p) {
  p << " " << getMapAttr();
  printDimAndSymbolList(operand_begin(), operand_end(),
                        getIndexMap().getNumDims(), p);
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"map"});
}

LogicalResult HLIndexApplyOp::verify() {
  // Check input and output dimensions match.
  AffineMap affineMap = getIndexMap();

  // Verify that operand count matches affine map dimension and symbol count.
  if (getNumOperands() != affineMap.getNumDims() + affineMap.getNumSymbols())
    return emitOpError(
        "operand count and affine map dimension and symbol count must match");

  // Verify that the map only produces one result.
  if (affineMap.getNumResults() != 1)
    return emitOpError("mapping must produce one value");

  return success();
}

/// Materialize a single constant operation from a given attribute value with
/// the desired resultant type.
Operation *HLIndexDialect::materializeConstant(OpBuilder &builder,
                                               Attribute value, Type type,
                                               Location loc) {
  return arith::ConstantOp::materialize(builder, value, type, loc);
}

OpFoldResult HLIndexApplyOp::fold(FoldAdaptor adaptor) {
  auto map = getIndexMap();

  // Fold dims and symbols to existing values.
  auto expr = map.getResult(0);
  if (auto dim = expr.dyn_cast<AffineDimExpr>())
    return getOperand(dim.getPosition());
  if (auto sym = expr.dyn_cast<AffineSymbolExpr>())
    return getOperand(map.getNumDims() + sym.getPosition());

  // Otherwise, default to folding the map.
  SmallVector<Attribute, 1> result;
  if (failed(map.constantFold(adaptor.getMapOperands(), result)))
    return {};
  return result[0];
}
