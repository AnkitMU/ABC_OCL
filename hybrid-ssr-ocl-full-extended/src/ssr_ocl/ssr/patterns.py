COMMON_PATTERNS = {
  "nonnegative_int": "self.%attr% >= 0",
  "collection_forall_positive": "self.%assoc%->forAll(x | x.%attr% > 0)"
}
