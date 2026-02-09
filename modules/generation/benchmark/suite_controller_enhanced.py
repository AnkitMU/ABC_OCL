"""
Enhanced Benchmark Suite Controller with Advanced Features
Integrates all 6 advanced features with comprehensive logging.

Features Integrated:
1. Metadata Enrichment (operators, depth, difficulty)
2. Manifest Generation (ML-friendly JSONL format)
3. Structural Similarity (tree-edit deduplication)
4. Semantic Similarity (clustering)
5. Implication Checking (constraint relationships)
6. UNSAT Generation (negative examples via mutation)
"""
import json
import random
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from modules.core.models import Metamodel, OCLConstraint
from modules.semantic.metamodel.xmi_extractor import MetamodelExtractor
from modules.generation.benchmark.engine_v2 import BenchmarkEngineV2
from modules.generation.benchmark.bench_config import BenchmarkProfile, QuantitiesConfig
from modules.verification.framework_verifier import FrameworkConstraintVerifier
from modules.utils.logger import FrameworkLogger, get_logger

# Advanced features (function-based modules)
from modules.generation.benchmark import metadata_enricher
from modules.generation.benchmark import manifest_generator
from modules.generation.benchmark import constraint_similarity
from modules.generation.benchmark import implication_checker
from modules.generation.benchmark import unsat_generator

from .suite_config import BenchmarkSuite, ModelSpec, ProfileSpec, get_difficulty_profile


class EnhancedSuiteController:
    """
    Enhanced controller with all advanced features integrated.
    
    Enhancements:
    - Comprehensive logging at every step
    - Metadata enrichment with operators/depth/difficulty
    - UNSAT generation via mutation strategies
    - Structural (tree-edit) similarity for deduplication
    - Semantic similarity for clustering
    - Implication checking for relationships
    - Manifest.jsonl generation for ML pipelines
    """
    
    def __init__(
        self,
        suite: BenchmarkSuite,
        verbose: bool = True,
        debug: bool = False,
        enable_research_features: bool = True
    ):
        """
        Initialize enhanced suite controller.
        
        Args:
            suite: Benchmark suite specification
            verbose: Enable verbose logging
            debug: Enable debug logging
            enable_research_features: Enable all 6 advanced features
        """
        self.suite = suite
        self.enable_research_features = enable_research_features
        self.logger = FrameworkLogger(verbose=verbose, debug=debug, name="SuiteController")
        
        self.output_root = Path(suite.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Get git commit if not provided
        if suite.git_commit is None:
            suite.git_commit = self._get_git_commit()
        
        # Initialize advanced modules
        if self.enable_research_features:
            self.logger.step("Initializing advanced features...")
            self._init_research_modules()
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                commit = result.stdout.strip()
                self.logger.debug(f"Git commit: {commit[:8]}")
                return commit
        except Exception as e:
            self.logger.debug(f"Could not get git commit: {e}")
        return None
    
    def _init_research_modules(self):
        """Initialize all research feature modules."""
        self.logger.indent()
        
        try:
            self.logger.debug("Verifying metadata_enricher module...")
            assert hasattr(metadata_enricher, 'enrich_constraint_metadata')
            self.logger.success("metadata_enricher ready")
            
            self.logger.debug("Verifying manifest_generator module...")
            assert hasattr(manifest_generator, 'generate_manifest')
            self.logger.success("manifest_generator ready")
            
            self.logger.debug("Verifying constraint_similarity module...")
            assert hasattr(constraint_similarity, 'ast_similarity')
            assert hasattr(constraint_similarity, 'compute_embeddings_batch')
            self.logger.success("constraint_similarity ready")
            
            self.logger.debug("Verifying implication_checker module...")
            assert hasattr(implication_checker, 'check_syntactic_implication')
            self.logger.success("implication_checker ready")
            
            self.logger.debug("Verifying unsat_generator module...")
            assert hasattr(unsat_generator, 'generate_mixed_sat_unsat_set')
            self.logger.success("unsat_generator ready")
            
            self.logger.success("All 6 advanced modules verified")
            
        except Exception as e:
            self.logger.error(f"Failed to verify advanced modules: {e}")
            self.logger.warning("Advanced features will be disabled")
            self.enable_research_features = False
        
        self.logger.dedent()
    
    def generate_suite(self) -> Dict:
        """
        Generate complete benchmark suite with all advanced features.
        
        Returns:
            Suite statistics dictionary
        """
        self.logger.section(f"BENCHMARK SUITE GENERATION: {self.suite.suite_name}")
        self.logger.info(f"Version: {self.suite.version}")
        self.logger.info(f"Framework Version: {self.suite.framework_version}")
        self.logger.info(f"Output: {self.output_root}")
        self.logger.info(f"Advanced Features: {'ENABLED' if self.enable_research_features else 'DISABLED'}")
        
        suite_stats = {
            'suite_name': self.suite.suite_name,
            'version': self.suite.version,
            'generated_at': datetime.now().isoformat(),
            'research_features_enabled': self.enable_research_features,
            'models': [],
            'total_constraints': 0,
            'total_sat': 0,
            'total_unsat': 0,
            'total_valid': 0,
            'total_unknown': 0
        }
        
        # Process each model
        for model_idx, model_spec in enumerate(self.suite.models, 1):
            self.logger.section(
                f"MODEL {model_idx}/{len(self.suite.models)}: {model_spec.name}",
                char="-"
            )
            self.logger.info(f"XMI: {model_spec.xmi}")
            
            model_stats = self._process_model(model_spec, model_idx)
            suite_stats['models'].append(model_stats)
            
            # Aggregate statistics
            suite_stats['total_constraints'] += model_stats['total_constraints']
            suite_stats['total_valid'] += model_stats['total_valid']
            suite_stats['total_sat'] += model_stats['total_sat']
            suite_stats['total_unsat'] += model_stats['total_unsat']
            suite_stats['total_unknown'] += model_stats['total_unknown']
        
        # Save suite summary
        summary_path = self.output_root / f"{self.suite.suite_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(suite_stats, f, indent=2)
        
        self.logger.success(f"Suite summary saved to: {summary_path}")
        
        # Final statistics
        self.logger.section("SUITE GENERATION COMPLETE")
        self.logger.statistics({
            'Total Constraints': suite_stats['total_constraints'],
            'Valid': suite_stats['total_valid'],
            'SAT': suite_stats['total_sat'],
            'UNSAT': suite_stats['total_unsat'],
            'Unknown': suite_stats['total_unknown'],
            'SAT Ratio': f"{suite_stats['total_sat'] / suite_stats['total_constraints'] * 100:.1f}%" if suite_stats['total_constraints'] > 0 else "0%",
            'UNSAT Ratio': f"{suite_stats['total_unsat'] / suite_stats['total_constraints'] * 100:.1f}%" if suite_stats['total_constraints'] > 0 else "0%"
        }, title="FINAL STATISTICS")
        
        return suite_stats
    
    def _process_model(self, model_spec: ModelSpec, model_idx: int) -> Dict:
        """Process single model with all its profiles."""
        model_stats = {
            'model': model_spec.name,
            'xmi': model_spec.xmi,
            'profiles': [],
            'total_constraints': 0,
            'total_valid': 0,
            'total_sat': 0,
            'total_unsat': 0,
            'total_unknown': 0
        }
        
        try:
            # Load metamodel
            self.logger.step("Loading metamodel from XMI...")
            self.logger.indent()
            
            extractor = MetamodelExtractor(model_spec.xmi)
            metamodel = extractor.get_metamodel()
            
            self.logger.success(f"Loaded {len(metamodel.classes)} classes")
            self.logger.debug(f"  - Attributes: {sum(len(c.attributes) for c in metamodel.classes.values())}")
            self.logger.debug(f"  - Associations: {len(metamodel.get_all_associations())}")
            
            self.logger.dedent()
            
            # Initialize engine and verifier
            self.logger.step("Initializing generation engine...")
            engine = BenchmarkEngineV2(
                metamodel,
                verification_enabled=self.suite.verification.enable
            )
            self.logger.success("Engine initialized")
            
            verifier = None
            if self.suite.verification.enable:
                self.logger.step("Initializing constraint verifier...")
                verifier = FrameworkConstraintVerifier(metamodel, model_spec.xmi)
                self.logger.success("Verifier initialized")
            
            # Process each profile
            for prof_idx, prof_spec in enumerate(model_spec.profiles, 1):
                self.logger.subsection(
                    f"PROFILE {prof_idx}/{len(model_spec.profiles)}: {prof_spec.name}",
                    char="·"
                )
                
                prof_stats = self._generate_profile(
                    model_spec, prof_spec, engine, verifier, metamodel
                )
                
                model_stats['profiles'].append(prof_stats)
                model_stats['total_constraints'] += prof_stats['constraint_count']
                model_stats['total_valid'] += prof_stats.get('valid_count', 0)
                model_stats['total_sat'] += prof_stats.get('sat_count', 0)
                model_stats['total_unsat'] += prof_stats.get('unsat_count', 0)
                model_stats['total_unknown'] += prof_stats.get('unknown_count', 0)
        
        except Exception as e:
            self.logger.error(f"Failed to process model: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            model_stats['error'] = str(e)
        
        return model_stats
    
    def _generate_profile(
        self,
        model_spec: ModelSpec,
        prof_spec: ProfileSpec,
        engine: BenchmarkEngineV2,
        verifier: Optional[FrameworkConstraintVerifier],
        metamodel: Metamodel
    ) -> Dict:
        """Generate constraints for a single profile with advanced features."""
        # Set random seed
        random.seed(prof_spec.seed)
        self.logger.debug(f"Random seed set to: {prof_spec.seed}")
        
        # Build profile
        profile = self._build_profile_from_spec(prof_spec, metamodel)
        
        # Create output directory
        output_dir = self.output_root / model_spec.name / prof_spec.name
        output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Output directory: {output_dir}")
        
        prof_stats = {
            'profile': prof_spec.name,
            'seed': prof_spec.seed,
            'complexity': prof_spec.complexity_profile,
            'output_dir': str(output_dir),
            'research_features_applied': []
        }
        
        try:
            # ============================================================
            # STEP 1: Generate base SAT constraints
            # ============================================================
            self.logger.step(f"Generating {prof_spec.constraints} base constraints...")
            self.logger.indent()
            
            constraints = engine.generate(profile, progress_callback=None)
            prof_stats['constraint_count'] = len(constraints)
            prof_stats['initial_sat_count'] = len(constraints)
            
            self.logger.success(f"Generated {len(constraints)} base constraints")
            self.logger.dedent()
            
            # ============================================================
            # STEP 2: Metadata Enrichment
            # ============================================================
            if self.enable_research_features:
                self.logger.step("Applying metadata enrichment...")
                self.logger.indent()
                
                enriched_constraints = []
                for i, constraint in enumerate(constraints):
                    enriched = metadata_enricher.enrich_constraint_metadata(constraint)
                    enriched_constraints.append(enriched)
                    
                    if i < 3:  # Show first 3 for debugging
                        self.logger.debug(f"  Constraint {i+1}:")
                        self.logger.debug(f"    - Operators: {enriched.metadata.get('operators_used', [])}")
                        self.logger.debug(f"    - Difficulty: {enriched.metadata.get('difficulty', 'unknown')}")
                        self.logger.debug(f"    - Nav Depth: {enriched.metadata.get('navigation_depth', 0)}")
                
                constraints = enriched_constraints
                prof_stats['research_features_applied'].append('metadata_enrichment')
                self.logger.success(f"Enriched {len(constraints)} constraints with metadata")
                self.logger.dedent()
            
            # ============================================================
            # STEP 3: UNSAT Generation
            # ============================================================
            if self.enable_research_features:
                self.logger.step("Generating UNSAT constraints via mutation...")
                self.logger.indent()
                
                # Target UNSAT ratio from profile spec
                target_unsat_ratio = prof_spec.unsat_ratio or 0.4
                self.logger.debug(f"Target UNSAT ratio: {target_unsat_ratio*100:.1f}%")
                
                # Use generate_mixed_sat_unsat_set function
                all_constraints, unsat_map = unsat_generator.generate_mixed_sat_unsat_set(
                    constraints, metamodel, unsat_ratio=target_unsat_ratio
                )
                
                unsat_count = len([c for c in all_constraints if c.metadata.get('is_unsat', False)])
                sat_count = len(all_constraints) - unsat_count
                
                # Count mutation strategies used
                mutation_counts = {}
                for strategy in unsat_map.values():
                    mutation_counts[strategy] = mutation_counts.get(strategy, 0) + 1
                
                self.logger.success(f"Generated {unsat_count} UNSAT constraints")
                self.logger.debug(f"  Total: {sat_count} SAT + {unsat_count} UNSAT = {len(all_constraints)}")
                if mutation_counts:
                    self.logger.debug(f"  Mutation strategies: {mutation_counts}")
                
                prof_stats['sat_count_after_mutation'] = sat_count
                prof_stats['unsat_count_after_mutation'] = unsat_count
                prof_stats['constraint_count'] = len(all_constraints)
                prof_stats['research_features_applied'].append('unsat_generation')
                
                self.logger.info(f"Total constraints: {len(all_constraints)} ({sat_count} SAT + {unsat_count} UNSAT)")
                self.logger.dedent()
                
                constraints = all_constraints
            
            # ============================================================
            # STEP 3.5: Constraint Compatibility Checking (Batch Mode)
            # ============================================================
            # NOTE: Runs silently - uses greedy algorithm to find maximal compatible subset
            if verifier and len(constraints) > 0:
                sat_constraints_only = [c for c in constraints if not c.metadata.get('is_unsat', False)]
                
                if len(sat_constraints_only) > 0:
                    try:
                        # Quick compatibility check (silent)
                        compat_results = verifier.verify_batch(sat_constraints_only, silent=True)
                        is_consistent = any(r.solver_result == 'sat' for r in compat_results)
                        
                        if not is_consistent:
                            # Model UNSAT - find compatible subset using greedy algorithm
                            compatible_sat = self._find_compatible_subset_batch(sat_constraints_only, verifier)
                            
                            if len(compatible_sat) < len(sat_constraints_only):
                                removed = len(sat_constraints_only) - len(compatible_sat)
                                
                                # Rebuild constraints list
                                unsat_constraints_only = [c for c in constraints if c.metadata.get('is_unsat', False)]
                                constraints = compatible_sat + unsat_constraints_only
                                prof_stats['constraint_count'] = len(constraints)
                                prof_stats['conflicts_removed'] = removed
                    except Exception:
                        pass
            
            # ============================================================
            # STEP 4: Structural Similarity (Tree-Edit) & Deduplication
            # ============================================================
            if self.enable_research_features:
                self.logger.step("Computing structural similarity (tree-edit)...")
                self.logger.indent()
                
                before_count = len(constraints)
                
                # Compute pairwise similarities
                similarities = []
                for i in range(len(constraints)):
                    for j in range(i + 1, len(constraints)):
                        sim = constraint_similarity.ast_similarity(constraints[i], constraints[j])
                        similarities.append((i, j, sim))
                
                self.logger.debug(f"Computed {len(similarities)} pairwise similarities")
                
                # Simple deduplication: remove constraints with similarity > threshold
                threshold = prof_spec.similarity_threshold or 0.85
                self.logger.debug(f"Deduplication threshold: {threshold}")
                
                # Mark duplicates
                to_remove = set()
                for i, j, sim in similarities:
                    if sim > threshold:
                        to_remove.add(j)  # Keep first occurrence, remove second
                
                dedup_constraints = [c for idx, c in enumerate(constraints) if idx not in to_remove]
                
                removed = before_count - len(dedup_constraints)
                if removed > 0:
                    self.logger.warning(f"Removed {removed} duplicate constraints (>{threshold*100:.0f}% similar)")
                    constraints = dedup_constraints
                    prof_stats['constraint_count'] = len(constraints)
                    prof_stats['duplicates_removed'] = removed
                else:
                    self.logger.success(f"No duplicates found (threshold: {threshold*100:.0f}%)")
                
                prof_stats['research_features_applied'].append('ast_similarity')
                self.logger.dedent()
            
            # ============================================================
            # STEP 5: Semantic Similarity & Clustering
            # ============================================================
            if self.enable_research_features:
                self.logger.step("Computing semantic similarity...")
                self.logger.indent()
                
                try:
                    # Compute embeddings
                    ocl_list = [c.ocl for c in constraints]
                    embeddings = constraint_similarity.compute_embeddings_batch(ocl_list)
                    self.logger.debug(f"Computed {len(embeddings)} semantic embeddings")
                    
                    # Cluster constraints using similarity threshold
                    clustering_threshold = 0.75  # Constraints with >75% similarity clustered together
                    clusters = constraint_similarity.cluster_by_semantic_similarity(
                        constraints, threshold=clustering_threshold
                    )
                    
                    num_clusters = len(clusters)
                    self.logger.success(f"Clustered into {num_clusters} semantic groups")
                    
                    # Build cluster assignment map (constraint index -> cluster id)
                    cluster_assignment = {}
                    for cluster_id, cluster_indices in enumerate(clusters):
                        for idx in cluster_indices:
                            cluster_assignment[idx] = cluster_id
                    
                    # Log cluster distribution
                    cluster_sizes = [len(c) for c in clusters]
                    self.logger.debug(f"  Cluster sizes: {cluster_sizes}")
                    
                    # Add cluster info to metadata
                    for idx, constraint in enumerate(constraints):
                        constraint.metadata['semantic_cluster'] = cluster_assignment.get(idx, -1)
                    
                    prof_stats['semantic_clusters'] = num_clusters
                    prof_stats['research_features_applied'].append('semantic_similarity')
                except Exception as e:
                    self.logger.warning(f"Semantic clustering failed: {e}")
                    self.logger.debug("Continuing without semantic clustering")
                
                self.logger.dedent()
            
            # ============================================================
            # STEP 6: Implication Checking
            # ============================================================
            if self.enable_research_features:
                self.logger.step("Checking constraint implications...")
                self.logger.indent()
                
                # Check implications between constraints (syntactic, optionally solver-based)
                implications = {f"{c.pattern_id}_{c.context}": [] for c in constraints}
                impl_count = 0
                use_solver = bool(
                    verifier
                    and self.suite.verification.enable
                    and getattr(self.suite.verification, 'implication_use_z3', False)
                )
                
                relations = implication_checker.find_implications(
                    constraints,
                    use_z3=use_solver,
                    verifier=verifier,
                    same_context_only=True
                )
                
                for i, j, relation in relations:
                    if relation == implication_checker.ImplicationRelation.C1_IMPLIES_C2:
                        implications[f"{constraints[i].pattern_id}_{constraints[i].context}"].append(
                            f"{constraints[j].pattern_id}_{constraints[j].context}"
                        )
                        impl_count += 1
                    elif relation == implication_checker.ImplicationRelation.C2_IMPLIES_C1:
                        implications[f"{constraints[j].pattern_id}_{constraints[j].context}"].append(
                            f"{constraints[i].pattern_id}_{constraints[i].context}"
                        )
                        impl_count += 1
                    elif relation == implication_checker.ImplicationRelation.EQUIVALENT:
                        implications[f"{constraints[i].pattern_id}_{constraints[i].context}"].append(
                            f"{constraints[j].pattern_id}_{constraints[j].context}"
                        )
                        implications[f"{constraints[j].pattern_id}_{constraints[j].context}"].append(
                            f"{constraints[i].pattern_id}_{constraints[i].context}"
                        )
                        impl_count += 2
                
                self.logger.success(f"Found {impl_count} implication relationships")
                
                if impl_count > 0:
                    # Show some examples
                    shown = 0
                    for constraint_id, implies_list in implications.items():
                        if implies_list and shown < 3:
                            self.logger.debug(f"  {constraint_id} implies {len(implies_list)} constraints")
                            shown += 1
                
                # Add implication info to metadata
                for constraint in constraints:
                    constraint_id = f"{constraint.pattern_id}_{constraint.context}"
                    constraint.metadata['implies'] = implications.get(constraint_id, [])
                
                prof_stats['implication_count'] = impl_count
                prof_stats['research_features_applied'].append('implication_checking')
                self.logger.dedent()
            
            # ============================================================
            # STEP 7: Verification
            # ============================================================
            verif_results = []
            if verifier and self.suite.verification.enable:
                self.logger.step("Verifying constraints with Z3...")
                self.logger.indent()
                
                # IMPORTANT: Only verify SAT constraints for global consistency
                # UNSAT constraints are INTENTIONALLY contradictory and would make the model UNSAT
                sat_constraints_for_verification = [c for c in constraints if not c.metadata.get('is_unsat', False)]
                unsat_constraints_skipped = [c for c in constraints if c.metadata.get('is_unsat', False)]
                
                if unsat_constraints_skipped:
                    self.logger.debug(f"Skipping {len(unsat_constraints_skipped)} UNSAT constraints from global consistency check")
                    self.logger.debug(f"Verifying {len(sat_constraints_for_verification)} SAT constraints for global consistency")
                
                verif_results = verifier.verify_batch(sat_constraints_for_verification, silent=True)
                
                # Count results
                valid_count = sum(1 for r in verif_results if r.is_valid)
                sat_count = sum(1 for r in verif_results if r.solver_result == 'sat')
                unsat_count = sum(1 for r in verif_results if r.solver_result == 'unsat')
                unknown_count = sum(1 for r in verif_results if r.solver_result == 'unknown')
                
                # IMPORTANT: verification results only include SAT constraints
                # Add intentionally-UNSAT constraints to unsat_count for final statistics
                total_unsat_constraints = unsat_count + len(unsat_constraints_skipped)
                
                prof_stats.update({
                    'valid_count': valid_count,
                    'sat_count': sat_count,
                    'unsat_count': total_unsat_constraints,  # Verified UNSAT + intentionally UNSAT
                    'unknown_count': unknown_count,
                    'verified_count': len(sat_constraints_for_verification),
                    'skipped_unsat_count': len(unsat_constraints_skipped),
                    'intentional_unsat_count': len(unsat_constraints_skipped)  # For clarity
                })
                
                self.logger.success(f"Verification complete:")
                self.logger.info(f"  Verified: {len(sat_constraints_for_verification)} SAT constraints")
                self.logger.info(f"  Skipped: {len(unsat_constraints_skipped)} UNSAT constraints (intentionally contradictory)")
                self.logger.info(f"  Valid: {valid_count}/{len(sat_constraints_for_verification)} ({valid_count/len(sat_constraints_for_verification)*100:.1f}%)" if len(sat_constraints_for_verification) > 0 else "  Valid: 0/0 (N/A)")
                self.logger.info(f"  Result: {sat_count} SAT, {total_unsat_constraints} UNSAT, {unknown_count} Unknown")
                self.logger.dedent()
            
            # ============================================================
            # STEP 8: Save Outputs
            # ============================================================
            self.logger.step("Saving outputs...")
            self.logger.indent()
            
            # Save traditional formats
            self._save_benchmark(
                output_dir, model_spec, prof_spec, profile,
                constraints, verif_results
            )
            self.logger.success("Saved constraints.json and constraints.ocl")
            
            # Save separate SAT and UNSAT files if advanced features enabled
            if self.enable_research_features:
                sat_constraints = [c for c in constraints if not c.metadata.get('is_unsat', False)]
                unsat_constraints = [c for c in constraints if c.metadata.get('is_unsat', False)]
                
                if sat_constraints:
                    self._save_sat_unsat_files(output_dir, sat_constraints, 'sat', model_spec, prof_spec)
                    self.logger.success(f"Saved constraints_sat.ocl and constraints_sat.json ({len(sat_constraints)} SAT)")
                
                if unsat_constraints:
                    self._save_sat_unsat_files(output_dir, unsat_constraints, 'unsat', model_spec, prof_spec)
                    self.logger.success(f"Saved constraints_unsat.ocl and constraints_unsat.json ({len(unsat_constraints)} UNSAT)")
            
            # Save manifest.jsonl (ML-friendly)
            if self.enable_research_features:
                from pathlib import Path as PathLib
                manifest_path = output_dir / "manifest.jsonl"
                manifest_generator.generate_manifest(
                    constraints=constraints,
                    model_name=model_spec.name,
                    profile_name=prof_spec.name,
                    output_path=PathLib(manifest_path)
                )
                prof_stats['research_features_applied'].append('manifest_generation')
                self.logger.success(f"Saved manifest.jsonl ({len(constraints)} records)")
            
            self.logger.dedent()
            
            prof_stats['status'] = 'success'
            self.logger.success(f"Profile '{prof_spec.name}' complete")
            
        except Exception as e:
            self.logger.error(f"Profile generation failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            prof_stats['status'] = 'failed'
            prof_stats['error'] = str(e)
        
        return prof_stats
    
    def _build_profile_from_spec(self, prof_spec: ProfileSpec, metamodel: Metamodel) -> BenchmarkProfile:
        """Build BenchmarkProfile from ProfileSpec."""
        # Get difficulty profile
        diff_profile = get_difficulty_profile(prof_spec.complexity_profile)
        
        # Build quantities config
        quantities = QuantitiesConfig(
            invariants=prof_spec.constraints,
            per_class_min=prof_spec.per_class_min or 1,
            per_class_max=prof_spec.per_class_max or 5
        )
        
        # Override families if specified
        if prof_spec.families_pct:
            quantities.families_pct = prof_spec.families_pct
        
        # Build full profile
        from modules.generation.benchmark.bench_config import (
            BenchmarkProfile, CoverageTargets, LibraryConfig, RedundancyConfig
        )
        
        profile = BenchmarkProfile(
            quantities=quantities,
            coverage=CoverageTargets(
                difficulty_mix=prof_spec.difficulty_mix or diff_profile['difficulty_mix']
            ),
            library=LibraryConfig(),
            redundancy=RedundancyConfig(
                similarity_threshold=prof_spec.similarity_threshold or 0.85,
                novelty_boost=prof_spec.novelty_boost if prof_spec.novelty_boost is not None else True
            )
        )
        
        return profile
    
    def _save_benchmark(
        self,
        output_dir: Path,
        model_spec: ModelSpec,
        prof_spec: ProfileSpec,
        profile: BenchmarkProfile,
        constraints: List[OCLConstraint],
        verif_results: List
    ):
        """Save benchmark outputs."""
        # Save constraints as OCL text
        ocl_path = output_dir / "constraints.ocl"
        with open(ocl_path, 'w') as f:
            f.write(f"-- Benchmark: {model_spec.name} / {prof_spec.name}\n")
            f.write(f"-- Generated: {datetime.now().isoformat()}\n")
            f.write(f"-- Seed: {prof_spec.seed}\n")
            f.write(f"-- Complexity: {prof_spec.complexity_profile}\n")
            f.write(f"-- Count: {len(constraints)}\n")
            f.write(f"-- Advanced Features: {'ENABLED' if self.enable_research_features else 'DISABLED'}\n\n")
            
            for i, c in enumerate(constraints, 1):
                f.write(f"-- #{i} {c.pattern_name} in {c.context}\n")
                
                # Add metadata if available
                if self.enable_research_features and c.metadata:
                    if 'difficulty' in c.metadata:
                        f.write(f"-- Difficulty: {c.metadata['difficulty']}\n")
                    if 'mutation_strategy' in c.metadata:
                        f.write(f"-- Type: UNSAT (mutated via {c.metadata['mutation_strategy']})\n")
                    elif 'is_unsat' in c.metadata and c.metadata['is_unsat']:
                        f.write(f"-- Type: UNSAT\n")
                    else:
                        f.write(f"-- Type: SAT\n")
                
                f.write(f"{c.ocl}\n\n")
        
        # Save as JSON with metadata
        json_path = output_dir / "constraints.json"
        data = {
            'metadata': {
                'suite': self.suite.suite_name,
                'model': model_spec.name,
                'profile': prof_spec.name,
                'generated_at': datetime.now().isoformat(),
                'seed': prof_spec.seed,
                'complexity': prof_spec.complexity_profile,
                'framework_version': self.suite.framework_version,
                'git_commit': self.suite.git_commit,
                'constraint_count': len(constraints),
                'research_features_enabled': self.enable_research_features
            },
            'constraints': [c.to_dict() for c in constraints]
        }
        
        if verif_results:
            data['verification'] = {
                'enabled': True,
                'results': [
                    {
                        'constraint_id': r.constraint_id,
                        'is_valid': r.is_valid,
                        'solver_result': r.solver_result,
                        'execution_time': r.execution_time
                    }
                    for r in verif_results
                ]
            }
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Save profile config
        profile_path = output_dir / "profile.json"
        with open(profile_path, 'w') as f:
            json.dump({
                'name': prof_spec.name,
                'seed': prof_spec.seed,
                'constraints': prof_spec.constraints,
                'complexity_profile': prof_spec.complexity_profile,
                'sat_ratio': prof_spec.sat_ratio,
                'unsat_ratio': prof_spec.unsat_ratio,
                'difficulty_mix': prof_spec.difficulty_mix,
                'research_features_enabled': self.enable_research_features
            }, f, indent=2)
    
    def _save_sat_unsat_files(
        self,
        output_dir: Path,
        constraints: List[OCLConstraint],
        constraint_type: str,
        model_spec: ModelSpec,
        prof_spec: ProfileSpec
    ):
        """Save SAT or UNSAT constraints to separate files.
        
        Args:
            output_dir: Output directory
            constraints: List of constraints (all SAT or all UNSAT)
            constraint_type: 'sat' or 'unsat'
            model_spec: Model specification
            prof_spec: Profile specification
        """
        # Save as OCL text
        ocl_path = output_dir / f"constraints_{constraint_type}.ocl"
        with open(ocl_path, 'w') as f:
            f.write(f"-- Benchmark: {model_spec.name} / {prof_spec.name}\n")
            f.write(f"-- Type: {constraint_type.upper()} Constraints Only\n")
            f.write(f"-- Generated: {datetime.now().isoformat()}\n")
            f.write(f"-- Count: {len(constraints)}\n\n")
            
            for i, c in enumerate(constraints, 1):
                f.write(f"-- #{i} {c.pattern_name} in {c.context}\n")
                
                # Add metadata if available
                if c.metadata:
                    if 'difficulty' in c.metadata:
                        f.write(f"-- Difficulty: {c.metadata['difficulty']}\n")
                    if constraint_type == 'unsat' and 'mutation' in c.metadata:
                        f.write(f"-- Mutation Strategy: {c.metadata['mutation']}\n")
                
                f.write(f"{c.ocl}\n\n")
        
        # Save as JSON
        json_path = output_dir / f"constraints_{constraint_type}.json"
        data = {
            'metadata': {
                'type': constraint_type.upper(),
                'model': model_spec.name,
                'profile': prof_spec.name,
                'generated_at': datetime.now().isoformat(),
                'constraint_count': len(constraints)
            },
            'constraints': [c.to_dict() for c in constraints]
        }
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
     
    def _find_compatible_subset_batch(self, constraints: List, verifier) -> List:
        """Find maximal compatible subset using batch greedy algorithm.
        
        This is optimized to minimize verification calls by testing batches
        instead of individual constraints.
        
        Args:
            constraints: List of OCLConstraint objects
            verifier: Constraint verifier
            
        Returns:
            List of compatible constraints (subset of input)
        """
        if len(constraints) <= 1:
            return constraints
        
        # Greedy algorithm: Start with empty set, add constraints one by one
        compatible = []
        
        for constraint in constraints:
            test_set = compatible + [constraint]
            
            try:
                # Test if adding this constraint keeps model SAT
                results = verifier.verify_batch(test_set, silent=True)
                is_consistent = any(r.solver_result == 'sat' for r in results)
                
                if is_consistent:
                    compatible.append(constraint)
                # else: skip this constraint (causes conflict)
            except Exception:
                pass
        
        return compatible
    
