//===- PresburgerBasicSet.h - MLIR PresburgerBasicSet Class -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A class to represent an integer polyhedra.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/PresburgerBasicSet.h"

namespace mlir {

std::unique_ptr<PresburgerBasicSet> PresburgerBasicSet::clone() const {
  return std::make_unique<PresburgerBasicSet>(*this);
}

void PresburgerBasicSet::reset(unsigned numReservedInequalities,
                               unsigned numReservedEqualities,
                               unsigned newNumReservedCols, unsigned newNumDims,
                               unsigned newNumSymbols, unsigned newNumLocals) {
  assert(newNumReservedCols >= newNumDims + newNumSymbols + newNumLocals + 1 &&
         "minimum 1 column");
  *this = PresburgerBasicSet(numReservedInequalities, numReservedEqualities,
                             newNumReservedCols, newNumDims, newNumSymbols,
                             newNumLocals);
}

void PresburgerBasicSet::reset(unsigned newNumDims, unsigned newNumSymbols,
                               unsigned newNumLocals) {
  reset(0, 0, newNumDims + newNumSymbols + newNumLocals + 1, newNumDims,
        newNumSymbols, newNumLocals);
}

void PresburgerBasicSet::append(const PresburgerBasicSet &other) {
  assert(other.getNumCols() == getNumCols());
  assert(other.getNumDimIds() == getNumDimIds());
  assert(other.getNumSymbolIds() == getNumSymbolIds());

  inequalities.reserveRows(inequalities.getNumRows() +
                           other.getNumInequalities());
  equalities.reserveRows(equalities.getNumRows() + other.getNumEqualities());

  for (unsigned r = 0, e = other.getNumInequalities(); r < e; r++) {
    addInequality(other.getInequality(r));
  }
  for (unsigned r = 0, e = other.getNumEqualities(); r < e; r++) {
    addEquality(other.getEquality(r));
  }
}

unsigned PresburgerBasicSet::insertDimId(unsigned pos, unsigned num) {
  return insertId(IdKind::Dimension, pos, num);
}

unsigned PresburgerBasicSet::insertSymbolId(unsigned pos, unsigned num) {
  return insertId(IdKind::Symbol, pos, num);
}

unsigned PresburgerBasicSet::insertLocalId(unsigned pos, unsigned num) {
  return insertId(IdKind::Local, pos, num);
}

unsigned PresburgerBasicSet::insertId(IdKind kind, unsigned pos, unsigned num) {
  assertAtMostNumIdKind(pos, kind);

  unsigned absolutePos = getIdKindOffset(kind) + pos;
  if (kind == IdKind::Dimension)
    numDims += num;
  else if (kind == IdKind::Symbol)
    numSymbols += num;
  numIds += num;

  inequalities.insertColumns(absolutePos, num);
  equalities.insertColumns(absolutePos, num);

  return absolutePos;
}

unsigned PresburgerBasicSet::appendDimId(unsigned num) {
  unsigned pos = getNumDimIds();
  insertId(IdKind::Dimension, pos, num);
  return pos;
}

unsigned PresburgerBasicSet::appendSymbolId(unsigned num) {
  unsigned pos = getNumSymbolIds();
  insertId(IdKind::Symbol, pos, num);
  return pos;
}

unsigned PresburgerBasicSet::appendLocalId(unsigned num) {
  unsigned pos = getNumLocalIds();
  insertId(IdKind::Local, pos, num);
  return pos;
}

void PresburgerBasicSet::addEquality(ArrayRef<int64_t> eq) {
  assert(eq.size() == getNumCols());
  unsigned row = equalities.appendExtraRow();
  for (unsigned i = 0, e = eq.size(); i < e; ++i)
    equalities(row, i) = eq[i];
}

void PresburgerBasicSet::addInequality(ArrayRef<int64_t> inEq) {
  assert(inEq.size() == getNumCols());
  unsigned row = inequalities.appendExtraRow();
  for (unsigned i = 0, e = inEq.size(); i < e; ++i)
    inequalities(row, i) = inEq[i];
}

void PresburgerBasicSet::removeId(IdKind kind, unsigned pos) {
  removeIdRange(kind, pos, pos + 1);
}

void PresburgerBasicSet::removeId(unsigned pos) {
  removeIdRange(pos, pos + 1);
}

void PresburgerBasicSet::removeIdRange(IdKind kind, unsigned idStart,
                                          unsigned idLimit) {
  assertAtMostNumIdKind(idLimit, kind);
  removeIdRange(getIdKindOffset(kind) + idStart,
                getIdKindOffset(kind) + idLimit);
}

void PresburgerBasicSet::removeIdRange(unsigned idStart, unsigned idLimit) {
  assert(idLimit < getNumCols() && "invalid id limit");

  if (idStart >= idLimit)
    return;

  // We are going to be removing one or more identifiers from the range.
  assert(idStart < numIds && "invalid idStart position");

  // TODO: Make 'removeIdRange' a lambda called from here.
  // Remove eliminated identifiers from the constraints..
  equalities.removeColumns(idStart, idLimit - idStart);
  inequalities.removeColumns(idStart, idLimit - idStart);

  // Update members numDims, numSymbols and numIds.
  unsigned numDimsEliminated = 0;
  unsigned numLocalsEliminated = 0;
  unsigned numColsEliminated = idLimit - idStart;
  if (idStart < numDims) {
    numDimsEliminated = std::min(numDims, idLimit) - idStart;
  }
  // Check how many local id's were removed. Note that our identifier order is
  // [dims, symbols, locals]. Local id start at position numDims + numSymbols.
  if (idLimit > numDims + numSymbols) {
    numLocalsEliminated = std::min(
        idLimit - std::max(idStart, numDims + numSymbols), getNumLocalIds());
  }
  unsigned numSymbolsEliminated =
      numColsEliminated - numDimsEliminated - numLocalsEliminated;

  numDims -= numDimsEliminated;
  numSymbols -= numSymbolsEliminated;
  numIds = numIds - numColsEliminated;
}

void PresburgerBasicSet::removeEquality(unsigned pos) {
  equalities.removeRow(pos);
}

void PresburgerBasicSet::removeInequality(unsigned pos) {
  inequalities.removeRow(pos);
}

void PresburgerBasicSet::removeEqualityRange(unsigned begin, unsigned end) {
  if (begin >= end)
    return;
  equalities.removeRows(begin, end - begin);
}

void PresburgerBasicSet::removeInequalityRange(unsigned begin,
                                                  unsigned end) {
  if (begin >= end)
    return;
  inequalities.removeRows(begin, end - begin);
}

void PresburgerBasicSet::swapId(unsigned posA, unsigned posB) {
  assert(posA < getNumIds() && "invalid position A");
  assert(posB < getNumIds() && "invalid position B");

  if (posA == posB)
    return;

  for (unsigned r = 0, e = getNumInequalities(); r < e; r++)
    std::swap(atIneq(r, posA), atIneq(r, posB));
  for (unsigned r = 0, e = getNumEqualities(); r < e; r++)
    std::swap(atEq(r, posA), atEq(r, posB));
}

unsigned PresburgerBasicSet::getIdKindOffset(IdKind kind) const {
  if (kind == IdKind::Dimension)
    return 0;
  if (kind == IdKind::Symbol)
    return getNumDimIds();
  if (kind == IdKind::Local)
    return getNumDimAndSymbolIds();
  llvm_unreachable("IdKind expected to be Dimension, Symbol or Local!");
}

void PresburgerBasicSet::assertAtMostNumIdKind(unsigned val,
                                                  IdKind kind) const {
  if (kind == IdKind::Dimension)
    assert(val <= getNumDimIds());
  else if (kind == IdKind::Symbol)
    assert(val <= getNumSymbolIds());
  else if (kind == IdKind::Local)
    assert(val <= getNumLocalIds());
  else
    llvm_unreachable("IdKind expected to be Dimension, Symbol or Local!");
}

void PresburgerBasicSet::clearConstraints() {
  equalities.resizeVertically(0);
  inequalities.resizeVertically(0);
}

} // namespace mlir
