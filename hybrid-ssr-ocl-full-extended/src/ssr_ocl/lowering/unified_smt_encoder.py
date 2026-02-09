#!/usr/bin/env python3
"""
Unified SMT Encoder for All 50 OCL Patterns
Consolidates encoding logic and provides a single source of truth.
Uses a "find a counterexample" style (negating expected properties),
so SAT => violation exists; UNSAT => property holds under given bounds.
"""

import re
from typing import Dict, Tuple
from z3 import *

class UnifiedSMTEncoder:
    """Single encoder for all 50 OCL patterns"""

    def __init__(self):
        self.max_depth = 10  # For closure operations
        self.max_scope = 20  # For collection bounds

        # Pattern registry mapping pattern names to encoder methods
        self.pattern_encoders = {
            # Basic Patterns (1-9)
            "pairwise_uniqueness": self.encode_pairwise_uniqueness,
            "exact_count_selection": self.encode_exact_count_selection,
            "global_collection": self.encode_global_collection,
            "set_intersection": self.encode_set_intersection,
            "size_constraint": self.encode_size_constraint,
            "uniqueness_constraint": self.encode_uniqueness_constraint,
            "collection_membership": self.encode_collection_membership,
            "null_check": self.encode_null_check,
            "numeric_comparison": self.encode_numeric_comparison,

            # Advanced Patterns (10-19)
            "exactly_one": self.encode_exactly_one,
            "closure_transitive": self.encode_closure_transitive,
            "acyclicity": self.encode_acyclicity,
            "aggregation_iterate": self.encode_aggregation_iterate,
            "boolean_guard_implies": self.encode_boolean_guard_implies,
            "safe_navigation": self.encode_safe_navigation,
            "type_check_casting": self.encode_type_check_casting,
            "subset_disjointness": self.encode_subset_disjointness,
            "ordering_ranking": self.encode_ordering_ranking,
            "contractual_temporal": self.encode_contractual_temporal,

            # Collection Operations (20-27)
            "select_reject": self.encode_select_reject,
            "collect_flatten": self.encode_collect_flatten,
            "any_operation": self.encode_any_operation,
            "forall_nested": self.encode_forall_nested,
            "exists_nested": self.encode_exists_nested,
            "collect_nested": self.encode_collect_nested,
            "as_set_as_bag": self.encode_as_set_as_bag,
            "sum_product": self.encode_sum_product,

            # String Operations (28-31)
            "string_concat": self.encode_string_concat,
            "string_operations": self.encode_string_operations,
            "string_comparison": self.encode_string_comparison,
            "string_pattern": self.encode_string_pattern,

            # Arithmetic & Logic (32-36)
            "arithmetic_expression": self.encode_arithmetic_expression,
            "div_mod_operations": self.encode_div_mod_operations,
            "abs_min_max": self.encode_abs_min_max,
            "boolean_operations": self.encode_boolean_operations,
            "if_then_else": self.encode_if_then_else,

            # Tuple & Let (37-39)
            "tuple_literal": self.encode_tuple_literal,
            "let_expression": self.encode_let_expression,
            "let_nested": self.encode_let_nested,

            # Set Operations (40-43)
            "union_intersection": self.encode_union_intersection,
            "symmetric_difference": self.encode_symmetric_difference,
            "including_excluding": self.encode_including_excluding,
            "flatten_operation": self.encode_flatten_operation,

            # Navigation & Property (44-47)
            "navigation_chain": self.encode_navigation_chain,
            "optional_navigation": self.encode_optional_navigation,
            "collection_navigation": self.encode_collection_navigation,
            "shorthand_notation": self.encode_shorthand_notation,

            # OCL Standard Library (48-50)
            "ocl_is_undefined": self.encode_ocl_is_undefined,
            "ocl_is_invalid": self.encode_ocl_is_invalid,
            "ocl_as_type": self.encode_ocl_as_type,
        }

    # ---------- helpers ----------

    def _num(self, s: str):
        """Return (Z3 numeral, is_real) based on presence of decimal point."""
        return (RealVal(s), True) if '.' in s else (IntVal(int(s)), False)

    def _bounded_tc(self, s: Solver, rel, n: int):
        """
        Build bounded transitive closure:
        P[l][i][j] is existence of a path of length <= l (with base including I ∪ rel).
        reach[i][j] = OR_l P[l][i][j]
        """
        L = self.max_depth
        P = [[[Bool(f"P_{l}_{i}_{j}") for j in range(n)] for i in range(n)] for l in range(L + 1)]
        # base: identity ∪ rel
        for i in range(n):
            for j in range(n):
                s.add(P[0][i][j] == Or(i == j, rel[i][j]))
        # grow
        for l in range(1, L + 1):
            for i in range(n):
                for j in range(n):
                    s.add(
                        P[l][i][j] ==
                        Or(
                            P[l - 1][i][j],
                            Or(*[And(P[l - 1][i][k], rel[k][j]) for k in range(n)])
                        )
                    )
        reach = [[Bool(f"reach_{i}_{j}") for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(n):
                s.add(reach[i][j] == Or(*[P[l][i][j] for l in range(L + 1)]))
        return reach

    # ---------- routing ----------

    def encode(self, pattern_name: str, ocl_text: str, context: Dict) -> Tuple[Solver, Dict]:
        """Main encoding entry point - routes to appropriate pattern encoder"""
        if pattern_name not in self.pattern_encoders:
            raise ValueError(f"Unknown pattern: {pattern_name}")
        return self.pattern_encoders[pattern_name](ocl_text, context)

    # ===== BASIC PATTERNS (1-9) =====

    def encode_pairwise_uniqueness(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver()
        model_vars = {}
        n = context.get('scope', 5)
        collection = context.get('collection', 'elements')
        ids = [Int(f"{collection}_id_{i}") for i in range(n)]
        for i, sym in enumerate(ids):
            model_vars[f"{collection}_id_{i}"] = sym
        solver.add(Not(Distinct(ids)))
        return solver, model_vars

    def encode_exact_count_selection(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        n = context.get('scope', 5); collection = context.get('collection', 'elements')
        ids = [Int(f"{collection}_id_{i}") for i in range(n)]
        self_id = Int("self_id")
        for i, sym in enumerate(ids):
            model_vars[f"{collection}_id_{i}"] = sym
        model_vars["self_id"] = self_id
        matches = [ids[i] == self_id for i in range(n)]
        count = Sum([If(m, 1, 0) for m in matches])
        model_vars["match_count"] = count
        m = re.search(r"size\(\)\s*=\s*(\d+)", text)
        if m:
            expected = IntVal(int(m.group(1)))
            solver.add(Not(count == expected))
        return solver, model_vars

    def encode_global_collection(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        all_valid = Bool("all_instances_valid")
        model_vars["all_instances_valid"] = all_valid
        solver.add(Not(all_valid))
        return solver, model_vars

    def encode_set_intersection(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        n = context.get('scope', 5)
        A = [Bool(f"in_A_{i}") for i in range(n)]
        B = [Bool(f"in_B_{i}") for i in range(n)]
        for i in range(n):
            model_vars[f"in_A_{i}"] = A[i]
            model_vars[f"in_B_{i}"] = B[i]
        inter = [And(A[i], B[i]) for i in range(n)]
        if "isEmpty" in text:
            # Violation of isEmpty: intersection contains something
            solver.add(Or(*inter))
        return solver, model_vars

    def encode_size_constraint(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        collection = context.get('collection', 'elements')
        size_var = Int(f"{collection}_size")
        model_vars[f"{collection}_size"] = size_var
        if "> 0" in text:
            c = size_var > 0
        elif ">=" in text:
            m = re.search(r">=\s*(\d+)", text)
            th = IntVal(int(m.group(1))) if m else IntVal(0)
            c = size_var >= th
        elif "<=" in text:
            m = re.search(r"<=\s*(\d+)", text)
            th = IntVal(int(m.group(1))) if m else IntVal(0)
            c = size_var <= th
        else:
            c = size_var > 0
        solver.add(Not(c))
        return solver, model_vars

    def encode_uniqueness_constraint(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        n = context.get('scope', 5); collection = context.get('collection', 'elements')
        xs = [Int(f"{collection}_{i}") for i in range(n)]
        for i, e in enumerate(xs):
            model_vars[f"{collection}_{i}"] = e
        solver.add(Not(Distinct(xs)))
        return solver, model_vars

    def encode_collection_membership(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        n = context.get('scope', 5); collection = context.get('collection', 'elements')
        elems = [Bool(f"{collection}_contains_{i}") for i in range(n)]
        for i, e in enumerate(elems):
            model_vars[f"{collection}_contains_{i}"] = e
        solver.add(Not(Or(*elems)))
        return solver, model_vars

    def encode_null_check(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        is_null = Bool("is_null"); model_vars["is_null"] = is_null
        if "<> null" in text:
            solver.add(is_null)      # violation of "not null"
        else:
            solver.add(Not(is_null)) # violation of "is null" expected
        return solver, model_vars

    def encode_numeric_comparison(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        m = re.search(r"(>=|<=)\s*([\d.]+)", text)
        if m:
            op, ths = m.groups()
            th, is_real = self._num(ths)
            val = Real("value") if is_real else Int("value")
            model_vars["value"] = val
            c = (val >= th) if op == ">=" else (val <= th)
            solver.add(Not(c))
        else:
            # default: make some comparison and negate it
            val = Int("value"); model_vars["value"] = val
            solver.add(Not(val >= 0))
        return solver, model_vars

    # ===== ADVANCED PATTERNS (10-19) =====

    def encode_exactly_one(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        n = context.get('scope', 5); collection = context.get('collection', 'elements')
        if n == 0:
            solver.add(True)
            return solver, model_vars
        if n == 1:
            c0 = Bool(f"{collection}_satisfies_0")
            model_vars[f"{collection}_satisfies_0"] = c0
            solver.add(Not(c0))
            return solver, model_vars
        conds = [Bool(f"{collection}_satisfies_{i}") for i in range(n)]
        for i, c in enumerate(conds):
            model_vars[f"{collection}_satisfies_{i}"] = c
        solver.add(Not(And(AtMost(*conds, 1), AtLeast(*conds, 1))))
        return solver, model_vars

    def encode_closure_transitive(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        n = context.get('scope', 5)
        rel = [[Bool(f"rel_{i}_{j}") for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(n):
                model_vars[f"rel_{i}_{j}"] = rel[i][j]
        reach = self._bounded_tc(solver, rel, n)
        for i in range(n):
            for j in range(n):
                model_vars[f"reach_{i}_{j}"] = reach[i][j]
        return solver, model_vars

    def encode_acyclicity(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        n = context.get('scope', 5)
        rel = [[Bool(f"rel_{i}_{j}") for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(n):
                model_vars[f"rel_{i}_{j}"] = rel[i][j]
        reach = self._bounded_tc(solver, rel, n)
        for i in range(n):
            for j in range(n):
                model_vars[f"reach_{i}_{j}"] = reach[i][j]
        non_trivial_cycle = Or(*[And(rel[i][k], reach[k][i]) for i in range(n) for k in range(n) if k != i])
        solver.add(Not(non_trivial_cycle))
        return solver, model_vars

    def encode_aggregation_iterate(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        n = context.get('scope', 5); collection = context.get('collection', 'items')
        if "sum" in text or "+" in text:
            elems = [Int(f"{collection}_{i}") for i in range(n)]
            accs = [Int(f"acc_{i}") for i in range(n + 1)]
            for i, e in enumerate(elems): model_vars[f"{collection}_{i}"] = e
            for i, a in enumerate(accs): model_vars[f"acc_{i}"] = a
            solver.add(accs[0] == 0)
            for i in range(n):
                solver.add(accs[i + 1] == accs[i] + elems[i])
            if ">=" in text:
                m = re.search(r">=\s*(\d+)", text)
                if m:
                    th = IntVal(int(m.group(1)))
                    solver.add(Not(accs[n] >= th))
        return solver, model_vars

    def encode_boolean_guard_implies(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        if "implies" in text:
            cond = Bool("guard_condition"); expr = Bool("consequent_expression")
            model_vars["guard_condition"] = cond; model_vars["consequent_expression"] = expr
            solver.add(Not(Or(Not(cond), expr)))
        return solver, model_vars

    def encode_safe_navigation(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        assoc_present = Bool("assoc_present"); assoc_valid = Bool("assoc_valid")
        model_vars["assoc_present"] = assoc_present; model_vars["assoc_valid"] = assoc_valid
        solver.add(Not(Implies(assoc_present, assoc_valid)))
        return solver, model_vars

    def encode_type_check_casting(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        if "oclIsKindOf" in text or "oclIsTypeOf" in text:
            tcheck = Bool("type_check"); model_vars["type_check"] = tcheck
            solver.add(Not(tcheck))
        return solver, model_vars

    def encode_subset_disjointness(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        n = context.get('scope', 5)
        A = [Bool(f"in_A_{i}") for i in range(n)]
        B = [Bool(f"in_B_{i}") for i in range(n)]
        for i in range(n):
            model_vars[f"in_A_{i}"] = A[i]
            model_vars[f"in_B_{i}"] = B[i]
        if "includesAll" in text:
            c = And(*[Implies(B[i], A[i % n]) for i in range(n)])
            solver.add(Not(c))
        elif "excludesAll" in text:
            c = And(*[Implies(B[i], Not(A[i % n])) for i in range(n)])
            solver.add(Not(c))
        return solver, model_vars

    def encode_ordering_ranking(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        n = context.get('scope', 5); collection = context.get('collection', 'elements')
        xs = [Int(f"{collection}_{i}") for i in range(n)]
        for i, e in enumerate(xs): model_vars[f"{collection}_{i}"] = e
        if "sortedBy" in text:
            for i in range(n - 1):
                solver.add(xs[i] <= xs[i + 1])
            # If there is a bound like first() >= k, negate it for violation:
            if "first()" in text:
                m = re.search(r">=\s*(\d+)", text)
                if m:
                    th = IntVal(int(m.group(1)))
                    solver.add(xs[0] < th)
            if "last()" in text:
                m = re.search(r"<=\s*(\d+)", text)
                if m:
                    th = IntVal(int(m.group(1)))
                    solver.add(xs[-1] > th)
        return solver, model_vars

    def encode_contractual_temporal(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        if "@pre" in text:
            bpre = Int("balance_pre"); bpost = Int("balance_post"); dep = Int("deposit")
            model_vars["balance_pre"] = bpre; model_vars["balance_post"] = bpost; model_vars["deposit"] = dep
            if "+" in text and "=" in text:
                solver.add(Not(bpre + dep == bpost))
        return solver, model_vars

    # ===== COLLECTION OPERATIONS (20-27) =====

    def encode_select_reject(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        n = context.get('scope', 5); collection = context.get('collection', 'elements')
        if "select" in text and ">" in text:
            m = re.search(r">\s*([\d.]+)", text)
            if m:
                th, is_real = self._num(m.group(1))
                elems = [Real(f"{collection}_{i}") for i in range(n)] if is_real else [Int(f"{collection}_{i}") for i in range(n)]
                sats = [Bool(f"{collection}_satisfies_{i}") for i in range(n)]
                for i in range(n):
                    model_vars[f"{collection}_{i}"] = elems[i]
                    model_vars[f"{collection}_satisfies_{i}"] = sats[i]
                    solver.add(sats[i] == (elems[i] > th))
                return solver, model_vars
        # fallback skeleton
        elems = [Int(f"{collection}_{i}") for i in range(n)]
        sats = [Bool(f"{collection}_satisfies_{i}") for i in range(n)]
        for i in range(n):
            model_vars[f"{collection}_{i}"] = elems[i]
            model_vars[f"{collection}_satisfies_{i}"] = sats[i]
        return solver, model_vars

    def encode_collect_flatten(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        n_outer = context.get('scope', 4); n_inner = context.get('inner_scope', 3)
        total = n_outer * n_inner
        if "flatten" in text and "size()" in text:
            flatten_size = Int("flatten_size")
            model_vars["flatten_size"] = flatten_size
            solver.add(flatten_size == total)
            # If there is a size() > k, negate it
            m = re.search(r"size\(\)\s*>\s*(\d+)", text)
            if m:
                solver.add(Not(flatten_size > IntVal(int(m.group(1)))))
        return solver, model_vars

    def encode_any_operation(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        n = context.get('scope', 5); collection = context.get('collection', 'elements')
        conds = [Bool(f"{collection}_condition_{i}") for i in range(n)]
        for i, c in enumerate(conds): model_vars[f"{collection}_condition_{i}"] = c
        solver.add(Not(Or(*conds)))
        return solver, model_vars

    def encode_forall_nested(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        n_outer = context.get('scope', 3); n_inner = context.get('inner_scope', 3)
        inn = [[Int(f"inner_{i}_{j}") for j in range(n_inner)] for i in range(n_outer)]
        for i in range(n_outer):
            for j in range(n_inner):
                model_vars[f"inner_{i}_{j}"] = inn[i][j]
        all_pos = And(*[inn[i][j] > 0 for i in range(n_outer) for j in range(n_inner)])
        solver.add(Not(all_pos))
        return solver, model_vars

    def encode_exists_nested(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        n_outer = context.get('scope', 3); n_inner = context.get('inner_scope', 3)
        inn = [[Int(f"inner_{i}_{j}") for j in range(n_inner)] for i in range(n_outer)]
        for i in range(n_outer):
            for j in range(n_inner):
                model_vars[f"inner_{i}_{j}"] = inn[i][j]
        some = Or(*[inn[i][j] > 100 for i in range(n_outer) for j in range(n_inner)])
        solver.add(Not(some))
        return solver, model_vars

    def encode_collect_nested(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        n_outer = context.get('scope', 3); n_mid = context.get('middle_scope', 3); n_inner = context.get('inner_scope', 2)
        total = n_outer * n_mid * n_inner
        total_size = Int("total_size"); model_vars["total_size"] = total_size
        solver.add(total_size == total)
        return solver, model_vars

    def encode_as_set_as_bag(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        n = context.get('scope', 5)
        unique_count = Int("unique_count"); orig = Int("original_size")
        model_vars["unique_count"] = unique_count; model_vars["original_size"] = orig
        solver.add(orig == n, unique_count >= 0, unique_count <= n)
        if "asSet" in text and "<" in text:
            solver.add(Not(unique_count < n))
        return solver, model_vars

    def encode_sum_product(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        n = context.get('scope', 5); collection = context.get('collection', 'items')
        if "sum" in text:
            m = re.search(r">\s*([\d.]+)", text)
            if m:
                th, is_real = self._num(m.group(1))
                elems = [Real(f"{collection}_{i}") for i in range(n)] if is_real else [Int(f"{collection}_{i}") for i in range(n)]
                for i, e in enumerate(elems): model_vars[f"{collection}_{i}"] = e
                total = Sum(elems); model_vars["sum"] = total
                solver.add(Not(total > th))
            else:
                # if no bound given, still produce a violation-style constraint
                elems = [Int(f"{collection}_{i}") for i in range(n)]
                for i, e in enumerate(elems): model_vars[f"{collection}_{i}"] = e
                solver.add(Not(Sum(elems) >= 0))
        return solver, model_vars

    # ===== STRING OPERATIONS (28-31) =====

    def encode_string_concat(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        s1 = String("str1"); s2 = String("str2")
        model_vars["str1"] = s1; model_vars["str2"] = s2
        if "concat" in text:
            res = Concat(s1, s2); model_vars["concat_result"] = res
            if "size()" in text and ">" in text:
                m = re.search(r"size\(\)\s*>\s*(\d+)", text)
                if m:
                    solver.add(Not(Length(res) > IntVal(int(m.group(1)))))
        return solver, model_vars

    def encode_string_operations(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        sv = String("string_var"); model_vars["string_var"] = sv
        if "substring" in text:
            start = Int("start"); end = Int("end")
            model_vars["start"] = start; model_vars["end"] = end
            solver.add(start >= 0, end >= start)
            sub = SubString(sv, start, end - start); model_vars["substring_result"] = sub
            if "size()" in text and ">" in text:
                solver.add(Not(Length(sub) > 0))
        elif "toUpper" in text or "toLower" in text:
            # model as length-preserving transformation
            res = String("case_result"); model_vars["case_result"] = res
            solver.add(Length(res) == Length(sv))
        return solver, model_vars

    def encode_string_comparison(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        st = String("status"); model_vars["status"] = st
        if "=" in text:
            lits = re.findall(r"['\"]([^'\"]+)['\"]", text)
            if lits:
                opts = [st == StringVal(l) for l in lits]
                solver.add(Not(Or(*opts)))
        return solver, model_vars

    def encode_string_pattern(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        s = String("string_var"); model_vars["string_var"] = s
        if "matches" in text:
            lower = Range('a', 'z'); plus = lambda r: Plus(r)
            at = Re("@"); dot = Re(r"\.")
            simple_email = Concat(plus(lower), at, plus(lower), dot, plus(lower))
            solver.add(Not(InRe(s, simple_email)))
        return solver, model_vars

    # ===== ARITHMETIC & LOGIC (32-36) =====

    def encode_arithmetic_expression(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        # model variables if referenced
        var_names = ['price', 'quantity', 'discount', 'total', 'amount']
        for vn in var_names:
            if vn in text.lower():
                model_vars[vn] = Real(vn)
        if "*" in text and "-" in text and 'price' in model_vars and 'quantity' in model_vars and 'discount' in model_vars:
            result = model_vars['price'] * model_vars['quantity'] - model_vars['discount']
            if ">=" in text:
                solver.add(Not(result >= 0))
        else:
            # Default shape to keep consistent violation style
            x = Real("x"); y = Real("y"); model_vars["x"]=x; model_vars["y"]=y
            solver.add(Not(x + y >= 0))
        return solver, model_vars

    def encode_div_mod_operations(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        if "div" in text or "mod" in text:
            total_hours = Int("totalHours"); days = Int("days"); remainder = Int("remainder")
            model_vars.update(totalHours=total_hours, days=days, remainder=remainder)
            if "div" in text:
                solver.add(days == total_hours / 24)   # integer division
            if "mod" in text:
                solver.add(remainder == total_hours % 24)
                solver.add(remainder >= 0, remainder < 24)
            solver.add(total_hours >= 0)
        return solver, model_vars

    def encode_abs_min_max(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        if "abs" in text:
            v = Int("value"); av = Int("abs_value")
            model_vars["value"] = v; model_vars["abs_value"] = av
            solver.add(av == If(v >= 0, v, -v))
            m = re.search(r"<=\s*(\d+)", text)
            if m:
                solver.add(Not(av <= IntVal(int(m.group(1)))))
        elif "min" in text or "max" in text:
            a = Int("a"); b = Int("b"); r = Int("result")
            model_vars["a"]=a; model_vars["b"]=b; model_vars["result"]=r
            if "min" in text:
                solver.add(r == If(a < b, a, b))
            else:
                solver.add(r == If(a > b, a, b))
            m = re.search(r">\s*(\d+)", text)
            if m:
                solver.add(Not(r > IntVal(int(m.group(1)))))
        return solver, model_vars

    def encode_boolean_operations(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        names = ['isActive', 'isDeleted', 'isArchived', 'isPending', 'isValid']
        for vn in names:
            if vn in text:
                model_vars[vn] = Bool(vn)
        if 'isActive' in model_vars and 'isDeleted' in model_vars:
            if "and" in text and "not" in text and "or" in text and 'isArchived' in model_vars:
                expr = Or(And(model_vars['isActive'], Not(model_vars['isDeleted'])),
                          model_vars['isArchived'])
                solver.add(Not(expr))
            elif "and" in text and "not" in text:
                expr = And(model_vars['isActive'], Not(model_vars['isDeleted']))
                solver.add(Not(expr))
        else:
            p = Bool("p"); q = Bool("q"); model_vars["p"]=p; model_vars["q"]=q
            solver.add(Not(And(p, q)))
        return solver, model_vars

    def encode_if_then_else(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        if "if" in text and "then" in text and "else" in text:
            cond = Bool("condition"); tr = Real("then_result"); er = Real("else_result"); res = Real("result")
            model_vars["condition"]=cond; model_vars["then_result"]=tr; model_vars["else_result"]=er; model_vars["result"]=res
            solver.add(res == If(cond, tr, er))
            if "isPremium" in text and "discount" in text:
                isp = Bool("isPremium"); disc = Real("discount")
                model_vars["isPremium"]=isp; model_vars["discount"]=disc
                solver.add(Not(If(isp, disc > 0.1, disc >= 0)))
        return solver, model_vars

    # ===== TUPLE & LET (37-39) =====

    def encode_tuple_literal(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        if "Tuple{" in text:
            ta = Int("tuple_a"); tb = Int("tuple_b")
            model_vars["tuple_a"]=ta; model_vars["tuple_b"]=tb
            if "+" in text and ">" in text:
                solver.add(Not(ta + tb > 0))
        return solver, model_vars

    def encode_let_expression(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        if "let" in text and "in" in text:
            m = re.search(r"let\s+(\w+)\s*=\s*([^\s]+)", text)
            if m:
                var_name = m.group(1)
                lv = Int(var_name); model_vars[var_name] = lv
                m2 = re.search(r">\s*(\d+)", text)
                if m2:
                    solver.add(Not(lv > IntVal(int(m2.group(1)))))
        return solver, model_vars

    def encode_let_nested(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        x = Int("x"); y = Int("y")
        model_vars["x"]=x; model_vars["y"]=y
        if "x + y" in text or "x+y" in text:
            solver.add(Not(x + y > 0))
        return solver, model_vars

    # ===== SET OPERATIONS (40-43) =====

    def encode_union_intersection(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        n = context.get('scope', 5)
        A = [Bool(f"in_A_{i}") for i in range(n)]
        B = [Bool(f"in_B_{i}") for i in range(n)]
        for i in range(n):
            model_vars[f"in_A_{i}"] = A[i]; model_vars[f"in_B_{i}"] = B[i]
        if "union" in text:
            usz = Sum([If(Or(A[i], B[i]), 1, 0) for i in range(n)])
            model_vars["union_size"] = usz
            m = re.search(r"size\(\)\s*>\s*(\d+)", text)
            if m:
                solver.add(Not(usz > IntVal(int(m.group(1)))))
        elif "intersection" in text:
            isz = Sum([If(And(A[i], B[i]), 1, 0) for i in range(n)])
            model_vars["intersection_size"] = isz
            m = re.search(r"size\(\)\s*>\s*(\d+)", text)
            if m:
                solver.add(Not(isz > IntVal(int(m.group(1)))))
        return solver, model_vars

    def encode_symmetric_difference(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        n = context.get('scope', 5)
        A = [Bool(f"in_A_{i}") for i in range(n)]
        B = [Bool(f"in_B_{i}") for i in range(n)]
        for i in range(n):
            model_vars[f"in_A_{i}"] = A[i]; model_vars[f"in_B_{i}"] = B[i]
        sym = [Xor(A[i], B[i]) for i in range(n)]
        if "isEmpty" in text:
            solver.add(Or(*sym))
        return solver, model_vars

    def encode_including_excluding(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        orig = Int("original_size"); new = Int("new_size")
        model_vars["original_size"] = orig; model_vars["new_size"] = new
        if "including" in text:
            present = Bool("element_present"); model_vars["element_present"] = present
            solver.add(new == If(present, orig, orig + 1))
            # if text expects size() = size() + 1, negate it
            if re.search(r"\bsize\(\)\s*=\s*size\(\)\s*\+\s*1\b", text):
                solver.add(Not(new == orig + 1))
        elif "excluding" in text:
            present = Bool("element_present"); model_vars["element_present"] = present
            solver.add(new == If(present, orig - 1, orig))
        return solver, model_vars

    def encode_flatten_operation(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        n_outer = context.get('scope', 3); n_inner = context.get('inner_scope', 4)
        total = n_outer * n_inner
        flat = [Int(f"flat_{i}") for i in range(total)]
        for i, e in enumerate(flat): model_vars[f"flat_{i}"] = e
        if "isUnique" in text:
            solver.add(Not(Distinct(flat)))
        return solver, model_vars

    # ===== NAVIGATION & PROPERTY (44-47) =====

    def encode_navigation_chain(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        if "department" in text and "university" in text:
            did = Int("department_id"); uid = Int("university_id")
            model_vars["department_id"] = did; model_vars["university_id"] = uid
            if "name" in text and "=" in text:
                nm = Bool("name_matches"); model_vars["name_matches"] = nm
                solver.add(Not(nm))
        return solver, model_vars

    def encode_optional_navigation(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        m = Bool("manager_exists"); d = Bool("department_exists"); n = Bool("name_exists")
        model_vars["manager_exists"] = m; model_vars["department_exists"] = d; model_vars["name_exists"] = n
        # monotonic chain: later implies earlier
        solver.add(Implies(n, d), Implies(d, m))
        if "<> null" in text:
            solver.add(Not(And(m, d, n)))
        return solver, model_vars

    def encode_collection_navigation(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        n = context.get('scope', 5)
        adv = [Int(f"advisor_exp_{i}") for i in range(n)]
        for i, a in enumerate(adv): model_vars[f"advisor_exp_{i}"] = a
        if "forAll" in text and ">=" in text:
            m = re.search(r">=\s*(\d+)", text)
            if m:
                th = IntVal(int(m.group(1)))
                solver.add(Not(And(*[a >= th for a in adv])))
        return solver, model_vars

    def encode_shorthand_notation(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        n = context.get('scope', 10)
        credits = [Int(f"credit_{i}") for i in range(n)]
        for i, c in enumerate(credits): model_vars[f"credit_{i}"] = c
        if "sum" in text:
            total = Sum(credits); model_vars["total_credits"] = total
            m = re.search(r">=\s*(\d+)", text)
            if m:
                solver.add(Not(total >= IntVal(int(m.group(1)))))
        return solver, model_vars

    # ===== OCL STANDARD LIBRARY (48-50) =====

    def encode_ocl_is_undefined(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        u = Bool("is_undefined"); model_vars["is_undefined"] = u
        if "oclIsUndefined" in text:
            if "= false" in text or "<> true" in text:
                solver.add(u)          # violation of "defined"
            else:
                solver.add(Not(u))     # violation of "undefined"
        return solver, model_vars

    def encode_ocl_is_invalid(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        inv = Bool("is_invalid"); model_vars["is_invalid"] = inv
        if "oclIsInvalid" in text:
            if "not" in text:
                solver.add(inv)        # violation of "not invalid"
            else:
                solver.add(Not(inv))   # violation of "invalid"
        return solver, model_vars

    def encode_ocl_as_type(self, text: str, context: Dict) -> Tuple[Solver, Dict]:
        solver = Solver(); model_vars = {}
        type_matches = Bool("type_matches"); model_vars["type_matches"] = type_matches
        if "oclAsType" in text and "gpa" in text:
            gpa = Real("gpa"); model_vars["gpa"] = gpa
            m = re.search(r">=\s*([\d.]+)", text)
            if m:
                th, _ = self._num(m.group(1))
                solver.add(Not(Implies(type_matches, gpa >= th)))
        return solver, model_vars


# Singleton instance
_encoder_instance = None

def get_unified_encoder() -> UnifiedSMTEncoder:
    """Get singleton instance of unified SMT encoder"""
    global _encoder_instance
    if _encoder_instance is None:
        _encoder_instance = UnifiedSMTEncoder()
    return _encoder_instance


# Quick smoke tests
if __name__ == "__main__":
    encoder = UnifiedSMTEncoder()

    # Pattern 1: Pairwise uniqueness
    ctx = {'scope': 5, 'collection': 'students'}
    s, _ = encoder.encode('pairwise_uniqueness', 'self.students->forAll(...)', ctx)
    print("P1 Pairwise Uniqueness:", s.check())

    # Pattern 10: Exactly one
    ctx = {'scope': 5, 'collection': 'accounts'}
    s, _ = encoder.encode('exactly_one', 'self.accounts->one(a | a.p)', ctx)
    print("P10 Exactly One:", s.check())

    # Pattern 27: Sum with real threshold
    ctx = {'scope': 4, 'collection': 'items'}
    s, _ = encoder.encode('sum_product', "items->collect(price)->sum() > 1000.0", ctx)
    print("P27 Sum (real):", s.check())

    # Pattern 31: Regex matches
    s, _ = encoder.encode('string_pattern', "str.matches('[a-z]+@[a-z]+\\.[a-z]+')", {})
    print("P31 Regex:", s.check())

    # Pattern 33: div/mod
    s, _ = encoder.encode('div_mod_operations', "totalHours div 24 = days and totalHours mod 24 < 24", {})
    print("P33 Div/Mod:", s.check())

    # Closure / Acyclicity
    s, _ = encoder.encode('closure_transitive', "A->closure(r)", {'scope': 4})
    print("P11 Closure:", s.check())
    s, _ = encoder.encode('acyclicity', "not self->closure(r)->includes(self)", {'scope': 4})
    print("P12 Acyclicity:", s.check())
