import pytest
from ssr_ocl.lowering.ocl2smt import encode_candidate_to_z3
from ssr_ocl.types import Candidate
from ssr_ocl.config_loader import load_cfg
from z3 import sat, unsat

@pytest.fixture
def test_config():
    return {
        '__scopes': {
            'classes': {
                'Student': 5,
                'Course': 3,
                'Professor': 2
            },
            'enums': {
                'Semester': ['Spring', 'Summer', 'Fall'],
                'Degree': ['UG', 'PG']
            }
        }
    }

def test_size_constraint_greater_than(test_config):
    """Test size() > 0 constraints"""
    ocl = "context University\ninv: self.departments->size() > 0"
    candidate = Candidate(context="University", ocl=ocl)
    solver, model_vars = encode_candidate_to_z3(candidate, test_config)
    
    # Should have departments_size variable
    assert 'departments_size' in model_vars
    
    # Check if constraint is satisfiable (should find counterexample where size <= 0)
    result = solver.check()
    assert result == sat

def test_is_unique_constraint(test_config):
    """Test isUnique constraints"""
    ocl = "context University\ninv: self.students->isUnique(s | s.studentId)"
    candidate = Candidate(context="University", ocl=ocl)
    solver, model_vars = encode_candidate_to_z3(candidate, test_config)
    
    # Should have student ID variables
    assert 'students_0_id' in model_vars
    assert 'students_1_id' in model_vars
    
    result = solver.check()
    assert result == sat  # Should find duplicate IDs

def test_age_constraint(test_config):
    """Test age >= 16 constraints"""
    ocl = "context Student\ninv: self.age >= 16"
    candidate = Candidate(context="Student", ocl=ocl)
    solver, model_vars = encode_candidate_to_z3(candidate, test_config)
    
    assert 'age' in model_vars
    
    result = solver.check()
    assert result == sat  # Should find age < 16

def test_gpa_range_constraint(test_config):
    """Test GPA range constraints"""
    ocl = "context Student\ninv: self.gpa >= 0.0 and self.gpa <= 4.0"
    candidate = Candidate(context="Student", ocl=ocl)
    solver, model_vars = encode_candidate_to_z3(candidate, test_config)
    
    assert 'gpa' in model_vars
    
    result = solver.check()
    assert result == sat  # Should find GPA outside range

def test_null_check_constraint(test_config):
    """Test null check constraints"""
    ocl = "context Student\ninv: self.advisor <> null"
    candidate = Candidate(context="Student", ocl=ocl)
    solver, model_vars = encode_candidate_to_z3(candidate, test_config)
    
    assert 'advisor_present' in model_vars
    
    result = solver.check()
    assert result == sat  # Should find null advisor

def test_includes_constraint(test_config):
    """Test includes constraints"""
    ocl = "context Course\ninv: not self.prerequisites->includes(self)"
    candidate = Candidate(context="Course", ocl=ocl)
    solver, model_vars = encode_candidate_to_z3(candidate, test_config)
    
    assert 'self_value' in model_vars
    assert 'prerequisites_0' in model_vars
    
    result = solver.check()
    assert result == sat  # Should find self-prerequisite

def test_credits_constraint(test_config):
    """Test credits range constraints"""
    ocl = "context Course\ninv: self.credits >= 1 and self.credits <= 10"
    candidate = Candidate(context="Course", ocl=ocl)
    solver, model_vars = encode_candidate_to_z3(candidate, test_config)
    
    assert 'credits' in model_vars
    
    result = solver.check()
    assert result == sat  # Should find credits outside range

def test_forall_constraint(test_config):
    """Test forAll constraints"""
    ocl = "context Course\ninv: self.enrollments->forAll(e | e.student <> null)"
    candidate = Candidate(context="Course", ocl=ocl)
    solver, model_vars = encode_candidate_to_z3(candidate, test_config)
    
    # Should have enrollment elements
    assert 'enrollments_0' in model_vars
    
    result = solver.check()
    assert result == sat

def test_max_seats_constraint(test_config):
    """Test maxSeats constraint"""
    ocl = "context Course\ninv: self.maxSeats > 0"
    candidate = Candidate(context="Course", ocl=ocl)
    solver, model_vars = encode_candidate_to_z3(candidate, test_config)
    
    assert 'maxSeats' in model_vars
    
    result = solver.check()
    assert result == sat  # Should find maxSeats <= 0

def test_experience_years_constraint(test_config):
    """Test experienceYears constraint"""
    ocl = "context Professor\ninv: self.experienceYears >= 0"
    candidate = Candidate(context="Professor", ocl=ocl)
    solver, model_vars = encode_candidate_to_z3(candidate, test_config)
    
    assert 'experienceYears' in model_vars
    
    result = solver.check()
    assert result == sat  # Should find negative experience