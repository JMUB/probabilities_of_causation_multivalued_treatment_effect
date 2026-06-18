# Probabilities of Causation for Multivalued Treatments and Effects

This repository provides Python implementations of the theorems for upper and lower bounds of probabilities of causation with multivalued treatments and effects from the paper ["Probabilities of Causation with Nonbinary Treatment and Effect"](https://arxiv.org/pdf/2208.09568) by Ang Li and Judea Pearl. The following examples from the paper are replicated:

- Choice of Treatment
- Change of Institute
- Effectiveness of Vaccine

## Requirements

- Python 3.7+
- numpy

## Data Format Requirements

The implementation requires two types of probability distributions: **observational** and **experimental** data. Both must be provided as Python dictionaries with specific tuple key formats.

### Observational Probabilities P(x, y)

Observational data represents the joint probability of observing treatment `x` and outcome `y` in a natural setting.

**Format:**
```python
observational_data = {
    ('x1', 'y1'): 0.25,
    ('x1', 'y2'): 0.15,
    ('x2', 'y1'): 0.30,
    ('x2', 'y2'): 0.30,
    # ... more combinations
}
```

**Key format:** `('x', 'y')` - treatment first, outcome second

**Requirements:**
- All combinations of treatments and outcomes should be included (missing values are treated as 0)
- The sum of all probabilities must equal 1.0
- Values must be between 0 and 1

### Experimental Probabilities P(y|do(x))

Experimental data represents the probability of outcome `y` when treatment `x` is forced (interventional data).

**Format:**
```python
experimental_data = {
    ('y1', 'x1'): 0.5,
    ('y2', 'x1'): 0.5,
    ('y1', 'x2'): 0.6,
    ('y2', 'x2'): 0.4,
    # ... more combinations
}
```

**Key format:** `('y', 'x')` - outcome first, treatment second

**⚠️ IMPORTANT:** Note the reversed order compared to observational data!

**Requirements:**
- For each treatment `x`, the probabilities of all outcomes must sum to 1.0
- Values must be between 0 and 1

### Example: Complete Data Setup

```python
# Define treatments and outcomes
treatments = ['x1', 'x2']  # e.g., surgery, chemotherapy
outcomes = ['y1', 'y2']    # e.g., cured, not cured

# Observational data: P(x, y)
observational_data = {
    ('x1', 'y1'): 0.20,
    ('x1', 'y2'): 0.10,
    ('x2', 'y1'): 0.35,
    ('x2', 'y2'): 0.35,
}
# Sum = 0.20 + 0.10 + 0.35 + 0.35 = 1.0 ✓

# Experimental data: P(y|do(x))
experimental_data = {
    ('y1', 'x1'): 0.67,  # P(y1|do(x1)) = 0.67
    ('y2', 'x1'): 0.33,  # P(y2|do(x1)) = 0.33
    ('y1', 'x2'): 0.50,  # P(y1|do(x2)) = 0.50
    ('y2', 'x2'): 0.50,  # P(y2|do(x2)) = 0.50
}
# For x1: 0.67 + 0.33 = 1.0 ✓
# For x2: 0.50 + 0.50 = 1.0 ✓
```

## Basic Usage Example

Here's a complete, runnable example demonstrating how to use the library:

```python
import sys
sys.path.insert(0, 'src/')

from probabilities_of_causation import Probabilities, get_bounds

# Step 1: Define your observational and experimental data
observational_data = {
    ('x1', 'y1'): 0.20,
    ('x1', 'y2'): 0.10,
    ('x2', 'y1'): 0.35,
    ('x2', 'y2'): 0.35,
}

experimental_data = {
    ('y1', 'x1'): 0.67,
    ('y2', 'x1'): 0.33,
    ('y1', 'x2'): 0.50,
    ('y2', 'x2'): 0.50,
}

# Step 2: Initialize the Probabilities class
# This will validate your data and check for completeness
probabilities = Probabilities(observational_data, experimental_data)

# Step 3: Define your query
# Query: What is the probability that a unit would have outcome y1
# if it received treatment x1, AND would have outcome y2 if it received treatment x2?
query = {
    "hypotheticals": [('y1', 'x1'), ('y2', 'x2')],
    "x_p": None,  # No specific treatment condition
    "y_q": None   # No specific outcome condition
}

# Step 4: Compute the bounds
lb, ub = get_bounds(query, probabilities)

# Step 5: Interpret the results
print(f"Query: P(y1(x1), y2(x2))")
print(f"Lower bound: {lb:.4f}")
print(f"Upper bound: {ub:.4f}")
print(f"\nInterpretation: The probability is between {lb:.1%} and {ub:.1%}")
```

**Output:**
```
=== Data Completeness Check ===

✓ All observational probabilities P(x, y) are present
✓ All experimental probabilities P(y|do(x)) are present

=== Marginal Probability Check ===
P(x1) = 0.3000
P(x2) = 0.7000
P(y1) = 0.5500
P(y2) = 0.4500
=== End of Data Completeness Check ===

Applying theorem 8: probability of necessity and sufficiency
Query: P(y1(x1), y2(x2))
Lower bound: 0.0200
Upper bound: 0.3300

Interpretation: The probability is between 2.0% and 33.0%
```

## Query Format

Queries are specified as dictionaries with the following structure:

```python
query = {
    "hypotheticals": [('y1', 'x1'), ('y2', 'x2'), ...],  # Required
    "x_p": None,                                         # Optional
    "y_q": None                                          # Optional
}
```

### Query Components

1. **`hypotheticals`** (Required): List of tuples representing counterfactual statements
   - Format: `[(y, x), ...]` where `y` is the hypothetical outcome and `x` is the hypothetical treatment
   - Each tuple represents: "The unit would have outcome y if it received treatment x"
   - Example: `[('y1', 'x1'), ('y2', 'x2')]` means "y1 if x1 AND y2 if x2"

2. **`x_p`** (Optional): A specific treatment value
   - Used in Theorems 6, 7, 9, and 11

3. **`y_q`** (Optional): A specific outcome value 
   - Used in Theorems 4, 5, 10, and 11


## Available Theorems

The implementation provides 8 theorems (Theorems 4-11 from the paper) for computing bounds on different types of probabilities of causation.

### Single Hypothetical Term (Theorems 4-7)

- **Theorem 4: Probability of Preservation** $P({y_i}_{x_j}, y_i)$
  - The probability that a unit with outcome $y_i$ would remain in outcome $y_i$ if exposed to treatment $x_j$
  
- **Theorem 5: Probability of Replacement** $P({y_i}_{x_j}, y_k)$ where $i \neq k$
  - The probability that a unit with outcome $y_k$ would switch to outcome $y_i$ if exposed to treatment $x_j$
  
- **Theorem 6: Probability of Substitute** $P({y_i}_{x_j}, x_k)$ where $j \neq k$
  - The probability that a unit exposed to treatment $x_k$ would have outcome $y_i$ had it been exposed to treatment $x_j$ instead
  
- **Theorem 7: Probability of Necessity** $P({y_i}_{x_j}, y_k, x_p)$ where $j \neq p$
  - The probability that a unit with treatment $x_p$ and outcome $y_k$ would have had outcome $y_i$ if it had received treatment $x_j$

### Multiple Hypothetical Terms (Theorems 8-11)

- **Theorem 8: Probability of Necessity and Sufficiency** $P({y_{i1}}_{x_{j1}}, ..., {y_{ik}}_{x_{jk}})$
  - Generalization of PNS to multiple hypothetical terms
  
- **Theorem 9: Probability of Substitute (k terms)** $P({y_{i1}}_{x_{j1}}, ..., {y_{ik}}_{x_{jk}}, x_p)$
  - Probability of substitute with $k$ hypothetical terms and treatment $x_p$
  
- **Theorem 10: Probability of Replacement (k terms)** $P({y_{i1}}_{x_{j1}}, ..., {y_{ik}}_{x_{jk}}, y_q)$
  - Probability of replacement with $k$ hypothetical terms and outcome $y_q$
  
- **Theorem 11: Probability of Necessity (k terms)** $P({y_{i1}}_{x_{j1}}, ..., {y_{ik}}_{x_{jk}}, x_p, y_q)$
  - Probability of necessity with $k$ hypothetical terms, treatment $x_p$, and outcome $y_q$

### Automatic Theorem Selection

The `get_bounds()` function automatically selects the appropriate theorem based on your query structure. You don't need to manually specify which theorem to use.

## Examples

The `examples/` directory contains Jupyter notebooks that replicate the examples from the paper:

1. **choice_of_treatment_Li_and_Pearl_2024.ipynb** - Medical treatment decision scenario
2. **change_of_institute_Li_and_Pearl_2024.ipynb** - Job training institute selection
3. **effectiveness_of_vaccine_Li_and_Pearl_2024.ipynb** - Vaccine effectiveness evaluation


## Data Validation

The `Probabilities` class automatically performs several validation checks:

1. **Probability Distribution Validation**
   - Ensures observational probabilities sum to 1.0
   - Ensures experimental probabilities for each treatment sum to 1.0

2. **Data Completeness Check**
   - Warns about missing observational probabilities P(x, y)
   - Warns about missing experimental probabilities P(y|do(x))
   - Displays marginal probabilities for reference

3. **Bounds Validation**
   - Validates that computed bounds are consistent (lower bound ≤ upper bound)
   - Prints warnings if inconsistent bounds are detected

## References

- **Paper:** "Probabilities of Causation with Nonbinary Treatment and Effect" by Ang Li and Judea Pearl
- **arXiv:** https://arxiv.org/pdf/2208.09568
- **AAAI 2024:** https://ojs.aaai.org/index.php/AAAI/article/view/30030
